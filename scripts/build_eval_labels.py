"""Giải luật từ khoá trong `eval_queries.yaml` thành danh sách đáp án đúng.

T-70 / US-601.1. Đọc `data/eval/eval_queries.yaml`, quét payload của cả ba
collection bằng **khớp chuỗi**, rồi ghi `data/eval/eval_queries.json` — tập đánh
giá đã cố định nhãn, chạy lại lúc nào cũng ra đúng một kết quả.

Vì sao gán nhãn bằng khớp chuỗi chứ không bằng tìm kiếm vector: nhãn phải độc
lập với thứ đang được đo. Lấy top-k của chính hệ thống làm đáp án đúng thì
Recall@k luôn bằng 1.0 và phép đo không nói lên điều gì.

    python scripts/build_eval_labels.py           # xem trước, không ghi
    python scripts/build_eval_labels.py --apply   # ghi eval_queries.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "API"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

THU_MUC = REPO_ROOT / "data" / "eval"
NGUON = THU_MUC / "eval_queries.yaml"
DICH = THU_MUC / "eval_queries.json"

# Mỗi mục khai `truong` để nói rõ nhãn dò ở đâu. Tách caption khỏi OCR là cố ý:
# vector trong `media_clip` là vector của HÌNH ẢNH, nên nhãn theo caption đo
# đúng thứ v1 làm, còn nhãn theo OCR đo khoảng trống dành cho v2.
TRUONG_DO = {
    "caption": ("caption",),
    "ocr": ("ocr_text",),
    "text": ("text", "page_content"),
}
COLLECTION_THEO_LOAI = {
    "media_image": "media_clip",
    "media_image_ocr": "media_clip",
    "media_transcript": "video_transcript",
    "regulation": "rag_documents",
}
# Chỉ số chính chỉ tính trên các loại này; phần còn lại báo cáo riêng.
LOAI_VAO_CHI_SO_CHINH = ("media_image", "media_transcript", "regulation")


def chuan_hoa(gia_tri: Any) -> str:
    """NFC + thường hoá, để 'Tết' và 'tết' là một."""
    return unicodedata.normalize("NFC", str(gia_tri or "")).lower()


def _khoa_don_vi(collection: str, payload: dict[str, Any], point_id: Any) -> str:
    """Định danh ổn định của một đơn vị truy hồi, khớp với thứ API trả về."""
    if collection == "rag_documents":
        return str(payload.get("chunk_id") or point_id)
    return str(payload.get("unit_id") or point_id)


def tai_collection(client: Any, collection: str) -> list[tuple[str, dict[str, str]]]:
    """Trả về (khoá đơn vị, {tên trường: nội dung đã chuẩn hoá}) cho cả collection."""
    moi_truong = sorted({ten for bo in TRUONG_DO.values() for ten in bo})
    ket_qua: list[tuple[str, dict[str, str]]] = []
    offset = None
    while True:
        diem, offset = client.scroll(
            collection, limit=1000, offset=offset, with_payload=True, with_vectors=False
        )
        for point in diem:
            payload = point.payload or {}
            noi_dung = {ten: chuan_hoa(payload.get(ten)) for ten in moi_truong}
            ket_qua.append((_khoa_don_vi(collection, payload, point.id), noi_dung))
        if offset is None:
            break
    return ket_qua


def _doc_tu_khoa(tu_khoa: Any) -> list[list[str]]:
    """Đọc `tu_khoa` thành dạng VÀ-của-HOẶC: [[a, b], [c]] nghĩa là (a hoặc b) và c.

    Một chuỗi là điều kiện bắt buộc; một danh sách con là nhóm đồng nghĩa, chỉ
    cần khớp một từ. Nhóm đồng nghĩa là bắt buộc chứ không phải tiện nghi: đo
    lần đầu bằng một từ khoá duy nhất, câu "ảnh các bạn đá bóng" chấm trượt cả
    "Các cầu thủ đang đứng trên sân cỏ nhân tạo" — ảnh đúng, chỉ là caption dùng
    từ khác. Nhãn hẹp như vậy dìm Recall xuống dưới mức thật và biến phép đo
    thành đo cách viết caption, chứ không đo chất lượng truy xuất.
    """
    dieu_kien: list[list[str]] = []
    for muc in tu_khoa:
        if isinstance(muc, (list, tuple)):
            dieu_kien.append([chuan_hoa(tu) for tu in muc])
        else:
            dieu_kien.append([chuan_hoa(muc)])
    return dieu_kien


def giai_nhan(muc: dict[str, Any], kho: dict[str, list[tuple[str, dict[str, str]]]]) -> list[str]:
    """Đáp án đúng = mọi đơn vị chứa ĐỦ mọi từ khoá, trong đúng trường đã khai.

    `media_image_ocr` cố ý loại những đơn vị đã khớp ở caption: mục đích của nó
    là đo riêng phần CHỈ có trong chữ trên ảnh, tức phần vector hình ảnh không
    mang. Để lẫn vào thì hai nhóm chồng lên nhau và không tách được kết luận.
    """
    collection = COLLECTION_THEO_LOAI[muc["loai"]]
    truong = TRUONG_DO[muc.get("truong") or "text"]
    dieu_kien = _doc_tu_khoa(muc.get("tu_khoa") or [])
    if not dieu_kien:
        return []
    dap_an = []
    for khoa, noi_dung in kho[collection]:
        gop = " ".join(noi_dung.get(ten, "") for ten in truong)
        if not all(any(tu in gop for tu in nhom) for nhom in dieu_kien):
            continue
        if muc["loai"] == "media_image_ocr" and all(
            any(tu in noi_dung.get("caption", "") for tu in nhom) for nhom in dieu_kien
        ):
            continue
        dap_an.append(khoa)
    return dap_an


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="ghi thật ra eval_queries.json")
    args = parser.parse_args()

    import yaml
    from qdrant_client import QdrantClient

    cau_hinh = yaml.safe_load(NGUON.read_text(encoding="utf-8"))
    client = QdrantClient(
        url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY") or None
    )
    kho = {ten: tai_collection(client, ten) for ten in ("media_clip", "video_transcript", "rag_documents")}
    for ten, muc_list in kho.items():
        print(f"{ten:18} {len(muc_list)} đơn vị")

    ra: list[dict[str, Any]] = []
    khong_co_nhan: list[str] = []
    for muc in cau_hinh["truy_van"]:
        if muc["loai"] == "ngoai_mien":
            ra.append({**muc, "dap_an": []})
            continue
        dap_an = giai_nhan(muc, kho)
        if not dap_an:
            # US-601.1 AC-2: mục thiếu nhãn bị loại, và phải có cảnh báo.
            khong_co_nhan.append(muc["id"])
        ra.append({**muc, "dap_an": dap_an})

    print("\nid      loại              số đáp án  câu hỏi")
    for muc in ra:
        if muc["loai"] == "ngoai_mien":
            continue
        canh_bao = "  <-- KHÔNG CÓ NHÃN" if not muc["dap_an"] else ""
        print(f"{muc['id']:8}{muc['loai']:18}{len(muc['dap_an']):>6}     {muc['cau_hoi'][:44]}{canh_bao}")

    trong_mien = [m for m in ra if m["loai"] != "ngoai_mien"]
    print(
        f"\n{len(ra)} mục: {len(trong_mien)} trong miền "
        f"({len(trong_mien) - len(khong_co_nhan)} có nhãn, {len(khong_co_nhan)} bị loại), "
        f"{len(ra) - len(trong_mien)} ngoài miền"
    )
    if khong_co_nhan:
        print(f"CẢNH BÁO — không giải ra nhãn: {', '.join(khong_co_nhan)}")

    if not args.apply:
        print("\n(xem trước — thêm --apply để ghi)")
        return 0

    DICH.write_text(
        json.dumps({"meta": cau_hinh["meta"], "truy_van": ra}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nđã ghi {DICH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
