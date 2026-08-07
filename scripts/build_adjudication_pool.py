#!/usr/bin/env python3
"""Dựng pool review mù để hoàn thiện qrel cho bộ đo nhúng văn bản.

76 nhãn hiện có đều đúng nhưng **không đầy đủ**, nên Recall/MRR/nDCG đang là cận
dưới: baseline Azure đã ghi ví dụ TXT-03 trả đúng một bài GEN 15 ở hạng 1 mà bị
chấm 0 vì ID đó chưa nằm trong qrel.

Không được bổ sung nhãn theo kết quả của riêng một model — làm vậy là thiên vị
đúng model mình lấy pool. Cách chuẩn (TREC pooling) là **hợp** top-K của mọi
model đang so, **xáo** thứ tự, **giấu** hạng và nguồn, rồi mới đưa người đọc chấm.

`text-retrieval-pool-v1.json` dựng ngày 05/08 chỉ có Azure, và chính nó ghi
"Do not finalize from Azure-only pool". Script này dựng bản hợp cả hai, giữ
nguyên tên trường của v1 để quy trình chấm không phải đổi.

Chỉ đọc cache vector có sẵn trên đĩa: không gọi provider nào, không tốn tiền.
Ghi hai file:

* ``<tên>.json``      — bản cho người chấm. Không có hạng, không có điểm, không
  có tên model.
* ``<tên>.key.json``  — bản đối chiếu: model nào tìm ra ứng viên nào ở hạng mấy,
  và nhãn đó đã có sẵn trong qrel hay chưa. **Không mở trước khi chấm xong.**
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import benchmark_text_embeddings as bench  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "data/eval/review_pools/text-retrieval-pool-v2.json"
# 10 là độ sâu pool tiêu chuẩn cho bộ nhỏ: đủ bắt gần hết bài đúng bị sót mà
# không đẩy số lượt chấm tay vượt quá một buổi ngồi.
DEFAULT_DEPTH = 10
# Đoạn trích của v1 chỉ ~90 ký tự — quá ngắn để phán đoán một bài có trả lời
# được câu hỏi hay không. 500 ký tự đủ thấy chủ đề mà vẫn đọc lướt được.
EXCERPT_CHARS = 500
# Cố định để chạy lại ra đúng thứ tự cũ. Đổi seed là đẩy bài đã chấm sang vị trí
# khác và người chấm phải làm lại từ đầu.
SEED = 20260806
NHAN = ("relevant", "partially_relevant", "not_relevant")


def doc_metadata_bai(corpus_root: Path) -> dict[str, dict]:
    """created_time và permalink giúp người chấm phân biệt các đợt cùng tên."""
    ra: dict[str, dict] = {}
    for path in corpus_root.glob("*/post.json"):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(d, dict) and d.get("id"):
            ra[str(d["id"])] = {
                "created_time": d.get("created_time"),
                "source_url": d.get("permalink_url"),
            }
    return ra


def build(depth: int, providers: list[str], accept_draft: bool) -> tuple[dict, dict]:
    bench.load_numpy()
    np = bench.np
    corpus_root = bench.resolve_corpus_root(bench.DEFAULT_CORPUS)
    corpus = bench.load_corpus(corpus_root)
    queries = bench.load_queries(bench.DEFAULT_BENCHMARK, {r.post_id for r in corpus}, accept_draft=accept_draft)
    chu_ky = bench.vector_fingerprint(corpus, queries)
    meta_bai = doc_metadata_bai(corpus_root)
    van_ban = {r.post_id: r.text for r in corpus}

    vectors = {}
    for ten in providers:
        duong_dan = bench.cache_path_for(bench.DEFAULT_CACHE_DIR, ten, chu_ky)
        if not duong_dan.is_file():
            raise bench.BenchmarkError(
                f"Thiếu cache '{duong_dan.name}'. Chạy benchmark cho provider '{ten}' trước; "
                "script này cố ý không tự gọi provider."
            )
        c = np.load(duong_dan)
        chuan = lambda m: m / np.linalg.norm(m, axis=1, keepdims=True)  # noqa: E731
        vectors[ten] = (chuan(c["queries"]), chuan(c["documents"]))

    rng = random.Random(SEED)
    review, key = [], []
    for vi_tri, q in enumerate(queries):
        if q.expected_empty:
            continue  # câu ngoài phạm vi không có bài đúng nào để chấm

        # Hợp qrel hiện có vào pool: người chấm gặp lại nhãn cũ mà không biết,
        # nên độ nhất quán của chính họ đo được sau khi đối chiếu key.
        nguon: dict[str, dict[str, int]] = {pid: {} for pid in q.relevant_ids}
        for ten, (Q, D) in vectors.items():
            xep = (Q[vi_tri] @ D.T).argsort()[::-1][:depth]
            for hang, idx in enumerate(xep, start=1):
                nguon.setdefault(corpus[int(idx)].post_id, {})[ten] = hang

        thu_tu = sorted(nguon)          # sắp trước rồi mới xáo để tái lập được
        rng.shuffle(thu_tu)

        muc_review, muc_key = [], []
        for post_id in thu_tu:
            m = meta_bai.get(post_id, {})
            noi_dung = van_ban.get(post_id, "")
            muc_review.append({
                "facebook_post_id": post_id,
                "created_time": m.get("created_time"),
                "source_url": m.get("source_url"),
                "excerpt": noi_dung[:EXCERPT_CHARS] + ("…" if len(noi_dung) > EXCERPT_CHARS else ""),
                "judgment": None,
                "review_note": "",
            })
            muc_key.append({
                "facebook_post_id": post_id,
                "tim_ra_boi": nguon[post_id],           # {provider: hạng}
                "chi_co_trong_qrel": not nguon[post_id],  # nhãn cũ, không model nào lôi lên trong top-K
                "da_co_trong_qrel": post_id in q.relevant_ids,
            })

        review.append({
            "query_id": q.query_id,
            "query": q.text,
            "category": q.category,
            "expected_empty": q.expected_empty,
            "evaluation_scope": q.evaluation_scope,
            "instructions": f"Đặt judgment là một trong {NHAN} mà không dùng hạng của provider.",
            "candidates": muc_review,
        })
        key.append({"query_id": q.query_id, "candidates": muc_key, "qrel_hien_tai": sorted(q.relevant_ids)})

    meta = {
        "version": 2,
        "created_at": "2026-08-06",
        "status": "awaiting_blind_human_judgment",
        "pool_method": f"union(current qrels, {', '.join(f'{p} top-{depth}' for p in providers)}); "
                       "candidate order deterministically shuffled",
        "provider_rank_and_score_hidden": True,
        "supersedes": "text-retrieval-pool-v1.json (Azure-only; chính nó ghi 'do not finalize')",
        "judgment_labels": list(NHAN),
        "vector_fingerprint": chu_ky,
        "seed": SEED,
        "so_cau_hoi": len(review),
        "so_luot_cham": sum(len(r["candidates"]) for r in review),
        "qrel_hien_tai": sum(len(k["qrel_hien_tai"]) for k in key),
    }
    return {"meta": meta, "queries": review}, {"meta": meta, "queries": key}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    p.add_argument("--providers", nargs="+", default=["azure", "jina-runpod"])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--accept-draft", action="store_true")
    p.add_argument("--force", action="store_true", help="ghi đè pool đã có (người chấm mất bài đã làm)")
    args = p.parse_args()

    key_path = args.out.with_suffix(".key.json")
    for path in (args.out, key_path):
        if path.exists() and not args.force:
            raise SystemExit(f"{path} đã tồn tại. Dùng --force nếu thật sự muốn ghi đè.")

    review, key = build(args.depth, args.providers, args.accept_draft)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(review["meta"], ensure_ascii=False, indent=2))
    print(f"\nbản chấm     : {args.out}")
    print(f"bản đối chiếu: {key_path}  ← KHÔNG mở trước khi chấm xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
