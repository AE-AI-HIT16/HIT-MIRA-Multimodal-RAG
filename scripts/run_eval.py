"""Chạy tập đánh giá và in báo cáo Recall@k / MRR / latency — T-71, T-72.

    python scripts/run_eval.py                # chạy, in báo cáo
    python scripts/run_eval.py --k 10         # đổi k
    python scripts/run_eval.py --rewrite      # bật viết lại truy vấn nhánh nội quy
    python scripts/run_eval.py --out bao_cao.json

Cần một nguồn nhúng đang sống (pod GPU hoặc API Jina) vì mỗi câu hỏi phải được
nhúng thật — xem embedding_server/README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "API"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from src.eval.runner import chay  # noqa: E402
from src.rag_noiquy.retrieval.retrieval_service import build_retrieval_service  # noqa: E402
from src.rag_video_anh.retrieval.retrieval_service import build_video_retrieval_service  # noqa: E402

TAP_MAC_DINH = REPO_ROOT / "data" / "eval" / "eval_queries.json"
BAO_CAO_MAC_DINH = REPO_ROOT / "data" / "eval" / "report.json"


def _dong(nhan: str, so: dict, muc_tieu: float | None = None) -> str:
    dat = ""
    if muc_tieu is not None:
        dat = "  ĐẠT" if so["recall_at_k"] >= muc_tieu else "  CHƯA ĐẠT"
    return (
        f"  {nhan:20} n={so['so_truy_van']:<3} "
        f"Recall@k={so['recall_at_k']:.3f}  MRR={so['mrr']:.3f}{dat}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tap", default=str(TAP_MAC_DINH), help="đường dẫn eval_queries.json")
    parser.add_argument("--k", type=int, default=None, help="top-k (mặc định lấy từ tập)")
    parser.add_argument("--rewrite", action="store_true", help="bật viết lại truy vấn nhánh nội quy")
    parser.add_argument("--out", default=str(BAO_CAO_MAC_DINH), help="nơi ghi báo cáo JSON")
    args = parser.parse_args()

    tap = json.loads(Path(args.tap).read_text(encoding="utf-8"))
    print(f"Tập đánh giá: {len(tap['truy_van'])} truy vấn\n")

    bao_cao = chay(
        tap,
        media_service=build_video_retrieval_service(),
        noi_quy_service=build_retrieval_service(),
        k=args.k,
        rewrite=args.rewrite,
        tien_trinh=lambda dong: print(f"  {dong}", flush=True),
    )
    ket = bao_cao.to_dict()
    muc_tieu = ket["muc_tieu"]

    print("\n" + "=" * 72)
    print(f"CHỈ SỐ CHÍNH (k={ket['k']}) — mục tiêu Recall@k ≥ {muc_tieu.get('recall_at_k')}, MRR ≥ {muc_tieu.get('mrr')}")
    print("=" * 72)
    chinh = ket["chi_so_chinh"]
    dat = ket["dat_muc_tieu"]
    print(f"  Recall@{ket['k']:<2} {chinh['recall_at_k']:.3f}   {'ĐẠT' if dat['recall_at_k'] else 'CHƯA ĐẠT'}")
    print(f"  MRR       {chinh['mrr']:.3f}   {'ĐẠT' if dat['mrr'] else 'CHƯA ĐẠT'}")
    print(f"  (trên {chinh['so_truy_van']} truy vấn trong miền)")

    print("\nTHEO TỪNG NHÓM")
    for loai, so in ket["theo_loai"].items():
        if so["so_truy_van"]:
            ghi_chu = "   <- phạm vi v2, không tính vào chỉ số chính" if loai == "media_image_ocr" else ""
            print(_dong(loai, so) + ghi_chu)

    lat = ket["latency"]
    print(f"\nLATENCY (mục tiêu trung bình ≤ {muc_tieu.get('latency_trung_binh')}s)")
    print(
        f"  trung bình {lat['trung_binh']:.2f}s   p95 {lat['p95']:.2f}s   cao nhất {lat['cao_nhat']:.2f}s"
        f"   {'ĐẠT' if dat['latency_trung_binh'] else 'CHƯA ĐẠT'}"
    )
    print("  (đo trong tiến trình, chưa gồm chi phí HTTP — là cận dưới)")

    ng = ket["nguong_ngoai_mien"]
    print("\nNGƯỠNG 'KHÔNG TÌM THẤY' (số liệu cho T-33)")
    print(f"  câu ngoài miền : cao nhất {ng['ngoai_mien_cao_nhat']:.3f}   trung bình {ng['ngoai_mien_trung_binh']:.3f}")
    print(f"  câu trong miền : thấp nhất {ng['trong_mien_thap_nhat']:.3f}   trung bình {ng['trong_mien_trung_binh']:.3f}")
    if ng["ngoai_mien_cao_nhat"] >= ng["trong_mien_thap_nhat"]:
        print("  => CHỒNG LẤN: không một ngưỡng điểm nào tách được hai nhóm.")
    else:
        giua = (ng["ngoai_mien_cao_nhat"] + ng["trong_mien_thap_nhat"]) / 2
        print(f"  => tách được; ngưỡng đề xuất ≈ {giua:.3f}")

    if ket["bi_loai_vi_thieu_nhan"]:
        print(f"\nCẢNH BÁO — loại vì thiếu nhãn: {', '.join(ket['bi_loai_vi_thieu_nhan'])}")
    loi = [c for c in ket["chi_tiet"] if c["loi"]]
    if loi:
        print(f"\nLỖI ({len(loi)} truy vấn):")
        for c in loi[:5]:
            print(f"  {c['id']}: {c['loi'][:90]}")

    Path(args.out).write_text(json.dumps(ket, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nđã ghi báo cáo: {Path(args.out).relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
