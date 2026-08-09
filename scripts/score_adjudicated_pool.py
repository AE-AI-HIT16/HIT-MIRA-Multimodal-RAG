#!/usr/bin/env python3
"""Chấm lại metric trên qrel đã adjudication — quy tắc chốt TRƯỚC khi có nhãn.

Viết ngày 06/08/2026, lúc pool v2 còn trắng 451 lượt chấm. Đó là chủ ý: khi
chưa ai nhìn thấy kết quả thì không thể chọn công thức theo kết quả. Mọi định
nghĩa dưới đây được ghim thêm bằng test (`API/tests/test_adjudication_scoring.py`),
nên sửa công thức mà quên sửa test là gãy CI chứ không trôi lặng lẽ.

Quy tắc đã khoá:

* ``nDCG@10`` dùng **graded relevance**, gain 2/1/0, chiết khấu ``log2(i+1)``.
* ``Recall@5`` và ``MRR@10`` **luôn báo cả hai** chế độ, không có cờ để chọn:
  - **strict**  — chỉ ``relevant`` được tính là đúng;
  - **lenient** — ``relevant`` và ``partially_relevant`` đều tính là đúng.
* Pool bị khoá bằng SHA-256 phủ *thứ tự ứng viên*. Thêm, bớt hay xáo lại giữa
  chừng là script từ chối chạy.
* Thiếu dù một nhãn cũng từ chối chạy: chấm nửa chừng rồi công bố số là cách
  êm ái nhất để tự lừa mình.

76 nhãn cũ nằm trộn trong pool được dùng để đo **intra-rater consistency** khi
chỉ một người chấm. Đó không phải inter-annotator agreement và không được gọi
như vậy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import benchmark_text_embeddings as bench  # noqa: E402

POOL_DIR = REPO_ROOT / "data/eval/review_pools"
DEFAULT_POOL = POOL_DIR / "text-retrieval-pool-v2.json"

GAIN = {"relevant": 2, "partially_relevant": 1, "not_relevant": 0}
STRICT = {"relevant"}
LENIENT = {"relevant", "partially_relevant"}
RECALL_K = 5
RANK_K = 10


class PoolError(RuntimeError):
    """Pool không dùng được để chấm điểm."""


# ------------------------------------------------------------------ metric


def recall_at_k(xep_hang: list[str], dung: set[str], k: int = RECALL_K) -> float:
    """Phần bài đúng lọt vào top-k. Câu không có bài đúng nào bị bỏ khỏi trung bình."""
    if not dung:
        raise ValueError("recall không định nghĩa được khi không có bài đúng nào")
    return len(set(xep_hang[:k]) & dung) / len(dung)


def mrr_at_k(xep_hang: list[str], dung: set[str], k: int = RANK_K) -> float:
    """Nghịch đảo hạng của bài đúng ĐẦU TIÊN; 0 nếu không có bài đúng nào trong top-k."""
    for i, post_id in enumerate(xep_hang[:k], start=1):
        if post_id in dung:
            return 1.0 / i
    return 0.0


def ndcg_at_k(xep_hang: list[str], gain: dict[str, int], k: int = RANK_K) -> float:
    """nDCG với gain phân mức. IDCG lấy từ toàn bộ gain đã chấm, không chỉ trong top-k."""
    dcg = sum(gain.get(pid, 0) / math.log2(i + 1) for i, pid in enumerate(xep_hang[:k], start=1))
    ly_tuong = sorted((g for g in gain.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ly_tuong, start=1))
    return dcg / idcg if idcg else 0.0


# ------------------------------------------------------------------ pool


def bam_thu_tu(pool: dict) -> str:
    canon = [[q["query_id"], [u["facebook_post_id"] for u in q["candidates"]]] for q in pool["queries"]]
    return hashlib.sha256(json.dumps(canon, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def kiem_pool(pool: dict, lock: dict) -> None:
    thuc_te = bam_thu_tu(pool)
    if thuc_te != lock["sha256_thu_tu_ung_vien"]:
        raise PoolError(
            "Danh sách hoặc thứ tự ứng viên đã đổi so với lúc khoá.\n"
            f"  khoá   : {lock['sha256_thu_tu_ung_vien']}\n"
            f"  hiện tại: {thuc_te}\n"
            "Pool phải bất biến trong lúc review; dựng lại pool là phải chấm lại từ đầu."
        )
    thieu = [
        f"{q['query_id']}/{u['facebook_post_id']}"
        for q in pool["queries"]
        for u in q["candidates"]
        if u.get("judgment") not in GAIN
    ]
    if thieu:
        raise PoolError(
            f"Còn {len(thieu)}/{lock['so_luot_cham']} lượt chưa chấm hoặc sai nhãn "
            f"(hợp lệ: {sorted(GAIN)}). Ví dụ: {thieu[:5]}"
        )


def qrel_tu_pool(pool: dict) -> dict[str, dict]:
    ra: dict[str, dict] = {}
    for q in pool["queries"]:
        gain = {u["facebook_post_id"]: GAIN[u["judgment"]] for u in q["candidates"]}
        ra[q["query_id"]] = {
            "gain": gain,
            "strict": {p for p, g in gain.items() if g == GAIN["relevant"]},
            "lenient": {p for p, g in gain.items() if g > 0},
        }
    return ra


def do_nhat_quan(pool: dict, key: dict) -> dict:
    """So nhãn mới với 76 nhãn cũ đã trộn vào pool mà người chấm không biết."""
    cu = {q["query_id"]: {u["facebook_post_id"] for u in q["candidates"] if u["da_co_trong_qrel"]}
          for q in key["queries"]}
    moi = {q["query_id"]: {u["facebook_post_id"]: u["judgment"] for u in q["candidates"]}
           for q in pool["queries"]}
    giu_strict = giu_lenient = tong = 0
    lech: list[str] = []
    for qid, ids in cu.items():
        for pid in ids:
            nhan = moi.get(qid, {}).get(pid)
            if nhan is None:
                continue
            tong += 1
            giu_strict += nhan in STRICT
            giu_lenient += nhan in LENIENT
            if nhan not in LENIENT:
                lech.append(f"{qid}/{pid} -> {nhan}")
    return {
        "chu_thich": "intra-rater consistency nếu cùng một người chấm; KHÔNG phải inter-annotator agreement",
        "so_nhan_cu_gap_lai": tong,
        "van_giu_relevant": giu_strict,
        "van_giu_relevant_hoac_partial": giu_lenient,
        "ty_le_strict": round(giu_strict / tong, 4) if tong else None,
        "ty_le_lenient": round(giu_lenient / tong, 4) if tong else None,
        "bi_ha_xuong_not_relevant": lech,
    }


# ------------------------------------------------------------------ chạy


def xep_hang_theo_provider(providers: list[str], query_ids: list[str]) -> dict[str, dict[str, list[str]]]:
    bench.load_numpy()
    np = bench.np
    corpus = bench.load_corpus(bench.resolve_corpus_root(bench.DEFAULT_CORPUS))
    queries = bench.load_queries(bench.DEFAULT_BENCHMARK, {r.post_id for r in corpus}, accept_draft=True)
    chu_ky = bench.vector_fingerprint(corpus, queries)
    vi_tri = {q.query_id: i for i, q in enumerate(queries)}

    ra: dict[str, dict[str, list[str]]] = {}
    for ten in providers:
        duong_dan = bench.cache_path_for(bench.DEFAULT_CACHE_DIR, ten, chu_ky)
        if not duong_dan.is_file():
            raise PoolError(f"Thiếu cache '{duong_dan.name}' cho provider '{ten}'.")
        c = np.load(duong_dan)
        chuan = lambda m: m / np.linalg.norm(m, axis=1, keepdims=True)  # noqa: E731
        Q, D = chuan(c["queries"]), chuan(c["documents"])
        ra[ten] = {
            qid: [corpus[int(i)].post_id for i in (Q[vi_tri[qid]] @ D.T).argsort()[::-1][:RANK_K]]
            for qid in query_ids
            if qid in vi_tri
        }
    return ra


def tinh_diem(xep: dict[str, list[str]], qrel: dict[str, dict]) -> dict:
    def tb(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    ket = {}
    for che_do in ("strict", "lenient"):
        r, m = [], []
        for qid, hang in xep.items():
            dung = qrel[qid][che_do]
            if not dung:
                continue  # không có bài đúng thì recall/MRR vô nghĩa, bỏ khỏi trung bình
            r.append(recall_at_k(hang, dung))
            m.append(mrr_at_k(hang, dung))
        ket[che_do] = {"so_cau": len(r), f"recall_at_{RECALL_K}": tb(r), f"mrr_at_{RANK_K}": tb(m)}
    n = [ndcg_at_k(hang, qrel[qid]["gain"]) for qid, hang in xep.items() if any(qrel[qid]["gain"].values())]
    ket["graded"] = {"so_cau": len(n), f"ndcg_at_{RANK_K}": tb(n)}
    return ket


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    p.add_argument("--providers", nargs="+", default=["azure", "jina-runpod"])
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data/eval/results/adjudicated-metrics.json")
    args = p.parse_args()

    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    lock = json.loads(args.pool.with_suffix(".lock.json").read_text(encoding="utf-8"))
    key_path = args.pool.with_suffix(".key.json")

    kiem_pool(pool, lock)
    qrel = qrel_tu_pool(pool)
    xep = xep_hang_theo_provider(args.providers, list(qrel))

    ket = {
        "pool": args.pool.name,
        "sha256_thu_tu_ung_vien": lock["sha256_thu_tu_ung_vien"],
        "rubric": lock["rubric"],
        "metric_rules": lock["metric_rules"],
        "qrel_sau_adjudication": {
            "strict": sum(len(v["strict"]) for v in qrel.values()),
            "lenient": sum(len(v["lenient"]) for v in qrel.values()),
        },
        "nhat_quan": do_nhat_quan(pool, json.loads(key_path.read_text(encoding="utf-8"))),
        "providers": {ten: tinh_diem(x, qrel) for ten, x in xep.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ket, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(ket, ensure_ascii=False, indent=2))
    print(f"\nĐã ghi {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
