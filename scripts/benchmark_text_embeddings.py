#!/usr/bin/env python3
"""Benchmark text retrieval with Azure embeddings and jina-clip-v2 on CPU.

Default mode is read-only planning. ``--apply`` is required because Azure makes
external requests and Jina may download/load several gigabytes of model data.
The script never writes vectors to Qdrant; it evaluates against local post text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPO_ROOT / "data/staging/google-drive/extracted/data"
DEFAULT_BENCHMARK = REPO_ROOT / "data/eval/text_embedding_benchmark_v1.json"
DEFAULT_CACHE_DIR = REPO_ROOT / "data/eval/cache"
DEFAULT_RESULTS_DIR = REPO_ROOT / "data/eval/results"
GIB = 1024**3
np: Any = None


class BenchmarkError(RuntimeError):
    """The benchmark configuration or input is unsafe/invalid."""


class TextProvider(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class CorpusRow:
    post_id: str
    text: str


@dataclass(frozen=True)
class QueryRow:
    query_id: str
    category: str
    text: str
    relevant_ids: frozenset[str]
    token_coverage: str
    evaluation_scope: str
    exclude_from_main: bool
    title_overlap: str
    expected_empty: bool


class AzureProvider:
    name = "azure"

    def __init__(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "API"))
        from src.rag_noiquy.embedding.embedding_service import EmbeddingService

        self._service = EmbeddingService(check_embedding_ctx_length=False)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._service.embed_documents(texts)


class JinaCpuProvider:
    name = "jina-cpu"

    def __init__(self) -> None:
        os.environ["EMBED_DEVICE"] = "cpu"
        sys.path.insert(0, str(REPO_ROOT / "embedding_server"))
        from encoder import JinaClipEncoder

        self._encoder = JinaClipEncoder()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._encoder.encode_texts(texts)


class JinaRunPodProvider:
    """Cùng model với `jina-cpu`, nhưng trọng số nằm trên GPU của RunPod.

    Đây là cách duy nhất đo được Jina trên máy này: bản CPU đạt đỉnh 5,05 GiB
    trong khi máy chỉ còn khoảng 5,3 GiB và không có swap. Phía máy chỉ giữ vài
    chục KB JSON.

    Đổi lại, `first_query_seconds` của provider này KHÔNG so sánh được với
    Azure: nó gánh cold start (đo được ~190 giây) nên là số của hạ tầng, không
    phải của model. Chỉ `warm_query_p50_seconds` mới so được, và cũng chỉ trong
    giới hạn "một job hàng đợi" chứ không phải "một lời gọi HTTP".
    """

    name = "jina-runpod"

    def __init__(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "API"))
        from src.rag_video_anh.embedding.runpod_transport import build_runpod_embedding_service

        self._service = build_runpod_embedding_service()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._service.embed_texts(texts)


PROVIDERS: dict[str, Callable[[], TextProvider]] = {
    "azure": AzureProvider,
    "jina-cpu": JinaCpuProvider,
    "jina-runpod": JinaRunPodProvider,
}


def load_numpy() -> None:
    global np
    try:
        import numpy as numpy_module
    except ImportError as exc:
        raise BenchmarkError(
            "Thiếu numpy trong Python hiện tại. Chạy bằng môi trường dự án: "
            "conda run -n nhhoang python scripts/benchmark_text_embeddings.py ..."
        ) from exc
    np = numpy_module


def load_dotenv_file(path: Path) -> None:
    """Load missing keys without printing secret values or requiring dotenv."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def resolve_corpus_root(root: Path) -> Path:
    root = root.resolve()
    if any(root.glob("*/post.json")):
        return root
    nested = root / "data"
    if nested.is_dir() and any(nested.glob("*/post.json")):
        return nested
    raise BenchmarkError(f"Không tìm thấy <post_id>/post.json dưới {root}")


def load_corpus(root: Path) -> list[CorpusRow]:
    rows: list[CorpusRow] = []
    seen: set[str] = set()
    for post_file in sorted(resolve_corpus_root(root).glob("*/post.json")):
        data = json.loads(post_file.read_text(encoding="utf-8"))
        post_id = str(data.get("id") or post_file.parent.name).strip()
        text = " ".join(str(data.get("message") or "").split())
        if not post_id or not text:
            continue
        if post_id in seen:
            raise BenchmarkError(f"Trùng facebook_post_id trong corpus: {post_id}")
        seen.add(post_id)
        rows.append(CorpusRow(post_id=post_id, text=text))
    if not rows:
        raise BenchmarkError("Corpus không có post message hợp lệ")
    return rows


def load_queries(path: Path, corpus_ids: set[str], *, accept_draft: bool) -> list[QueryRow]:
    data = json.loads(path.read_text(encoding="utf-8"))
    status = str(data.get("meta", {}).get("status", ""))
    if status.startswith("draft") and not accept_draft:
        raise BenchmarkError(
            "Bộ nhãn còn là draft. Duyệt nhãn rồi đổi status, hoặc truyền --accept-draft để chạy thử."
        )
    rows: list[QueryRow] = []
    seen: set[str] = set()
    for item in data.get("queries", []):
        query_id = str(item.get("id", "")).strip()
        text = " ".join(str(item.get("query", "")).split())
        relevant = frozenset(str(value) for value in item.get("relevant_facebook_post_ids", []))
        expected_empty = bool(item.get("expected_empty", False))
        if not query_id or not text:
            raise BenchmarkError(f"Query thiếu id/text: {query_id or '<unknown>'}")
        if not relevant and not expected_empty:
            raise BenchmarkError(f"Query dương thiếu qrel: {query_id}")
        if relevant and expected_empty:
            raise BenchmarkError(f"Query âm lại có qrel: {query_id}")
        if query_id in seen:
            raise BenchmarkError(f"Trùng query id: {query_id}")
        missing = relevant - corpus_ids
        if missing:
            raise BenchmarkError(f"{query_id} tham chiếu post không tồn tại: {sorted(missing)}")
        seen.add(query_id)
        rows.append(
            QueryRow(
                query_id=query_id,
                category=str(item.get("category") or "unknown"),
                text=text,
                relevant_ids=relevant,
                token_coverage=str(item.get("discriminative_token_coverage") or "unknown"),
                evaluation_scope=str(item.get("evaluation_scope") or "vector_text"),
                exclude_from_main=bool(item.get("exclude_from_main_vector_metrics", False)),
                title_overlap=str(item.get("title_overlap") or "normal"),
                expected_empty=expected_empty,
            )
        )
    if sum(not row.expected_empty for row in rows) < 30:
        raise BenchmarkError("Cần ít nhất 30 query dương")
    return rows


def memory_info() -> dict[str, int]:
    values: dict[str, int] = {}
    for raw in Path("/proc/meminfo").read_text().splitlines():
        key, value = raw.split(":", 1)
        values[key] = int(value.strip().split()[0]) * 1024
    return {
        "ram_total_bytes": values.get("MemTotal", 0),
        "ram_available_bytes": values.get("MemAvailable", 0),
        "swap_total_bytes": values.get("SwapTotal", 0),
        "swap_free_bytes": values.get("SwapFree", 0),
    }


def require_env(provider: str, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if not os.getenv(key, "").strip()]
    if missing:
        raise BenchmarkError(f"Thiếu cấu hình {provider}: {', '.join(missing)}")


def validate_provider_preflight(provider: str, *, allow_oom_risk: bool) -> None:
    if provider == "azure":
        require_env(provider, ("EMBEDDING_API_KEY", "EMBEDDING_BASE_URL", "EMBEDDING_MODEL"))
        return
    if provider == "jina-runpod":
        # RUNPOD_ENDPOINT_ID là worker video; nhầm hai biến này thì job vào đúng
        # hàng đợi sai và trả về lỗi rất khó lần.
        require_env(provider, ("RUNPOD_API_KEY", "JINA_RUNPOD_ENDPOINT_ID"))
        return
    if provider != "jina-cpu":
        raise BenchmarkError(f"Provider không hỗ trợ: {provider}")
    memory = memory_info()
    if memory["ram_total_bytes"] < 12 * GIB and not allow_oom_risk:
        raise BenchmarkError(
            "Máy có dưới 12 GiB RAM. Số đo thật của repo cho thấy jina-clip-v2 "
            "đạt đỉnh 5,05 GiB với fp32 và 4,89 GiB với bfloat16; cgroup 3 GiB "
            "sẽ bảo vệ dịch vụ nhưng sẽ giết benchmark. Không tạo swap. Hãy "
            "chạy trên máy RAM lớn hơn hoặc chỉ dùng --allow-oom-risk trong "
            "maintenance window và dưới cgroup đủ lớn."
        )


def provider_signature(provider: str) -> str:
    """Nhận dạng đầy đủ của một provider: tên chưa đủ.

    Cache cũ chỉ khoá theo tên provider và văn bản đầu vào, nên chạy `azure` ở
    512 chiều sau khi đã chạy ở 1.536 sẽ nạp lại đúng file cũ và báo ra số của
    lần trước — sai mà không có triệu chứng nào. Model và số chiều phải nằm
    trong khoá.

    Nạp `.env` ngay tại đây: script khác gọi hàm này mà quên nạp thì tên cache
    ra `unknown-default`, và cùng một cache lại mang hai tên khác nhau tuỳ chỗ gọi.
    """
    load_dotenv_file(REPO_ROOT / ".env")
    if provider == "azure":
        model = os.getenv("EMBEDDING_MODEL", "").strip() or "unknown"
        dims = os.getenv("EMBEDDING_DIMENSIONS", "").strip() or "default"
        return f"{provider}-{model}-{dims}"
    if provider in {"jina-cpu", "jina-runpod"}:
        model = os.getenv("MEDIA_IMAGE_EMBEDDING_MODEL", "").strip() or "jina-clip-v2"
        return f"{provider}-{model}-1024"
    return provider


def cache_path_for(cache_dir: Path, provider: str, vector_signature: str) -> Path:
    """Một chỗ duy nhất đặt tên cache, để script khác không tự ghép sai."""
    return cache_dir / f"text-{provider_signature(provider)}-{vector_signature}.npz"


def vector_fingerprint(corpus: list[CorpusRow], queries: list[QueryRow]) -> str:
    """Fingerprint only provider inputs so qrel edits do not invalidate paid caches."""
    digest = hashlib.sha256()
    for row in corpus:
        digest.update(row.post_id.encode())
        digest.update(b"\0")
        digest.update(row.text.encode())
        digest.update(b"\0")
    for row in queries:
        digest.update(row.query_id.encode())
        digest.update(b"\0")
        digest.update(row.text.encode())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def benchmark_fingerprint(corpus: list[CorpusRow], queries: list[QueryRow]) -> str:
    """Fingerprint labels/eval scopes separately from reusable embedding inputs."""
    digest = hashlib.sha256(vector_fingerprint(corpus, queries).encode())
    for row in queries:
        digest.update(row.evaluation_scope.encode())
        digest.update(b"\0")
        digest.update(row.token_coverage.encode())
        digest.update(b"\0")
        digest.update(row.title_overlap.encode())
        digest.update(b"\0")
        digest.update(str(row.expected_empty).encode())
        digest.update(b"\0")
        for post_id in sorted(row.relevant_ids):
            digest.update(post_id.encode())
            digest.update(b"\0")
    return digest.hexdigest()[:16]


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not vectors.shape[1]:
        raise BenchmarkError(f"Ma trận vector không hợp lệ: {vectors.shape}")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _aggregate_positive(rows: list[dict]) -> dict:
    if not rows:
        return {"queries": 0, "recall_at_5": 0.0, "mrr_at_10": 0.0, "ndcg_at_10": 0.0}
    return {
        "queries": len(rows),
        "recall_at_5": statistics.fmean(row["recall_at_5"] for row in rows),
        "mrr_at_10": statistics.fmean(row["reciprocal_rank_at_10"] for row in rows),
        "ndcg_at_10": statistics.fmean(row["ndcg_at_10"] for row in rows),
    }


def evaluate(
    corpus: list[CorpusRow],
    queries: list[QueryRow],
    doc_vectors: np.ndarray,
    query_vectors: np.ndarray,
    *,
    rejection_threshold: float | None = None,
) -> dict:
    doc_vectors = normalize(doc_vectors)
    query_vectors = normalize(query_vectors)
    if len(doc_vectors) != len(corpus) or len(query_vectors) != len(queries):
        raise BenchmarkError("Số vector không khớp corpus/query")
    scores = query_vectors @ doc_vectors.T
    per_query: list[dict] = []
    for query_index, query in enumerate(queries):
        order = np.argsort(-scores[query_index], kind="stable")[:10]
        ranking = [corpus[int(index)].post_id for index in order]
        top_score = float(scores[query_index, int(order[0])])
        row = {
            "id": query.query_id,
            "category": query.category,
            "evaluation_scope": query.evaluation_scope,
            "token_coverage": query.token_coverage,
            "title_overlap": query.title_overlap,
            "expected_empty": query.expected_empty,
            "top_score": top_score,
            "top_10": ranking,
        }
        if query.expected_empty:
            row.update(
                {
                    "recall_at_5": None,
                    "reciprocal_rank_at_10": None,
                    "ndcg_at_10": None,
                    "accepted_at_threshold": (
                        top_score >= rejection_threshold if rejection_threshold is not None else None
                    ),
                }
            )
        else:
            relevant_ranks = [
                rank for rank, post_id in enumerate(ranking, 1) if post_id in query.relevant_ids
            ]
            recall5 = len(set(ranking[:5]) & query.relevant_ids) / len(query.relevant_ids)
            reciprocal_rank = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0
            dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
            ideal_count = min(len(query.relevant_ids), 10)
            ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
            row.update(
                {
                    "recall_at_5": recall5,
                    "reciprocal_rank_at_10": reciprocal_rank,
                    "ndcg_at_10": dcg / ideal_dcg if ideal_dcg else 0.0,
                    "accepted_at_threshold": (
                        top_score >= rejection_threshold if rejection_threshold is not None else None
                    ),
                }
            )
        per_query.append(row)

    main_rows = [
        row
        for row, query in zip(per_query, queries, strict=True)
        if not query.expected_empty and not query.exclude_from_main
    ]
    metadata_rows = [
        row
        for row, query in zip(per_query, queries, strict=True)
        if not query.expected_empty and query.evaluation_scope == "metadata_filter"
    ]
    negative_rows = [row for row in per_query if row["expected_empty"]]
    token_groups = {
        coverage: _aggregate_positive([row for row in main_rows if row["token_coverage"] == coverage])
        for coverage in ("all", "partial", "none", "unparsed")
        if any(row["token_coverage"] == coverage for row in main_rows)
    }
    no_title_overlap = [row for row in main_rows if row["title_overlap"] != "high"]
    negative_scores = [row["top_score"] for row in negative_rows]
    rejection = {
        "queries": len(negative_rows),
        "threshold": rejection_threshold,
        "top_score_min": min(negative_scores) if negative_scores else None,
        "top_score_p50": statistics.median(negative_scores) if negative_scores else None,
        "top_score_p95": percentile(negative_scores, 0.95) if negative_scores else None,
        "top_score_max": max(negative_scores) if negative_scores else None,
        "false_positive_rate": (
            statistics.fmean(float(row["accepted_at_threshold"]) for row in negative_rows)
            if negative_rows and rejection_threshold is not None
            else None
        ),
        "warning": "Chọn threshold trên development split riêng; không tune và báo điểm trên cùng 5 câu âm.",
    }
    main = _aggregate_positive(main_rows)
    return {
        "recall_at_5": main["recall_at_5"],
        "mrr_at_10": main["mrr_at_10"],
        "ndcg_at_10": main["ndcg_at_10"],
        "main_vector_metrics": main,
        "by_token_coverage": token_groups,
        "without_high_title_overlap": _aggregate_positive(no_title_overlap),
        "metadata_filter_required": _aggregate_positive(metadata_rows),
        "rejection": rejection,
        "queries": per_query,
    }


def embed_provider(
    provider: TextProvider,
    corpus: list[CorpusRow],
    queries: list[QueryRow],
    *,
    batch_size: int,
    cache_path: Path,
    rejection_threshold: float | None,
) -> dict:
    if cache_path.is_file():
        cached = np.load(cache_path)
        doc_vectors = cached["documents"]
        query_vectors = cached["queries"]
        timings = {"cache_hit": True}
    else:
        query_vectors_list: list[list[float]] = []
        query_latencies: list[float] = []
        for query in queries:
            started = time.perf_counter()
            vectors = provider.embed([query.text])
            query_latencies.append(time.perf_counter() - started)
            query_vectors_list.append(vectors[0])

        document_vectors_list: list[list[float]] = []
        document_started = time.perf_counter()
        texts = [row.text for row in corpus]
        for start in range(0, len(texts), batch_size):
            document_vectors_list.extend(provider.embed(texts[start : start + batch_size]))
        document_seconds = time.perf_counter() - document_started

        doc_vectors = np.asarray(document_vectors_list, dtype=np.float32)
        query_vectors = np.asarray(query_vectors_list, dtype=np.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, documents=doc_vectors, queries=query_vectors)
        timings = {
            "cache_hit": False,
            "first_query_seconds": query_latencies[0],
            "warm_query_p50_seconds": statistics.median(query_latencies[1:]),
            "warm_query_p95_seconds": percentile(query_latencies[1:], 0.95),
            "documents_total_seconds": document_seconds,
            "documents_per_second": len(corpus) / document_seconds if document_seconds else 0.0,
        }
    metrics = evaluate(
        corpus, queries, doc_vectors, query_vectors, rejection_threshold=rejection_threshold
    )
    return {
        "provider": provider.name,
        "dimension": int(doc_vectors.shape[1]),
        "timings": timings,
        "metrics": metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--providers", nargs="+", choices=tuple(PROVIDERS), default=["azure", "jina-cpu"]
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--accept-draft", action="store_true", help="chạy thử dù nhãn chưa được duyệt")
    parser.add_argument(
        "--rejection-threshold",
        type=float,
        help="cosine threshold đã chốt trên development split; bỏ trống để chỉ báo score",
    )
    parser.add_argument(
        "--allow-oom-risk",
        action="store_true",
        help="bỏ chặn RAM của jina-cpu; chỉ dùng trong maintenance window có cgroup",
    )
    parser.add_argument("--apply", action="store_true", help="thực sự gọi provider/nạp model")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise BenchmarkError("--batch-size phải >= 1")
    load_dotenv_file(REPO_ROOT / ".env")
    corpus_root = resolve_corpus_root(args.corpus_root)
    source_post_files = sum(1 for _ in corpus_root.glob("*/post.json"))
    corpus = load_corpus(corpus_root)
    # Dry-run vẫn cho phép đọc draft để in kế hoạch; apply buộc người chạy xác nhận rõ.
    queries = load_queries(
        args.benchmark,
        {row.post_id for row in corpus},
        accept_draft=args.accept_draft or not args.apply,
    )
    memory = memory_info()
    plan = {
        "applied": args.apply,
        "source_post_files": source_post_files,
        "corpus_documents": len(corpus),
        "skipped_empty_messages": source_post_files - len(corpus),
        "queries": len(queries),
        "positive_queries": sum(not row.expected_empty for row in queries),
        "negative_queries": sum(row.expected_empty for row in queries),
        "metadata_filter_queries": sum(row.evaluation_scope == "metadata_filter" for row in queries),
        "qrels": sum(len(row.relevant_ids) for row in queries),
        "providers": args.providers,
        "batch_size": args.batch_size,
        "vector_fingerprint": vector_fingerprint(corpus, queries),
        "benchmark_fingerprint": benchmark_fingerprint(corpus, queries),
        "memory_gib": {key: round(value / GIB, 2) for key, value in memory.items()},
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.apply:
        print("Dry-run: chưa gọi Azure, chưa tải/nạp Jina và chưa ghi cache/result.")
        return 0

    for provider_name in args.providers:
        validate_provider_preflight(provider_name, allow_oom_risk=args.allow_oom_risk)

    load_numpy()
    vector_signature = plan["vector_fingerprint"]
    benchmark_signature = plan["benchmark_fingerprint"]
    results = []
    for provider_name in args.providers:
        provider: TextProvider = PROVIDERS[provider_name]()
        cache_path = cache_path_for(args.cache_dir, provider_name, vector_signature)
        results.append(
            embed_provider(
                provider,
                corpus,
                queries,
                batch_size=args.batch_size,
                cache_path=cache_path,
                rejection_threshold=args.rejection_threshold,
            )
        )

    output = {
        "meta": plan,
        "results": results,
        "warning": "Draft labels were used." if args.accept_draft else None,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.results_dir / f"text-embedding-{benchmark_signature}.json"
    result_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result_path.resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
