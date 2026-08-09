"""So vector self-host với vector đang nằm trong Qdrant.

**Đây là câu hỏi đắt nhất của cả việc chuyển sang tự host.** 2.409 điểm trong
Qdrant được nhúng bằng api.jina.ai. Nếu bản tự host cho ra vector khác — dù chỉ
khác vì một bước tiền xử lý ảnh mà API làm ngầm — thì toàn bộ index cũ thành vô
giá trị: hai không gian vector trộn trong một collection làm điểm xếp hạng mất
hết ý nghĩa. Biết trước thì chỉ phải chạy nốt 38 video; biết sau thì chạy lại
cả 59 mà không hiểu vì sao kết quả tìm kiếm ngày càng lạ.

Script chỉ đọc, không sửa gì, nên không cần `--apply`.

    # Kiểm bản CPU cục bộ (không cần dựng server)
    /home/ubuntu/.venvs/embedding_server/bin/python scripts/check_embedding_parity.py

    # Kiểm endpoint đã deploy (RunPod hoặc bản local đang chạy)
    python scripts/check_embedding_parity.py --url https://xxx.runpod.net/v1/embeddings

Đọc kết quả:

* cosine >= 0.998  → cùng không gian, index cũ dùng tiếp được.
* 0.99 – 0.998     → vẫn là cùng model, nhưng lệch nhiều hơn mức đã đo được;
                     nên xem lại tiền xử lý trước khi tin.
* < 0.99           → KHÁC không gian. Phải nhúng lại toàn bộ collection.

Ngưỡng 0.998 lấy từ số đo thật ngày 02/08/2026, không phải chọn cho đẹp:
fp32 trên CPU đối chiếu với vector do api.jina.ai sinh, 4 ảnh + 4 đoạn lời
thoại, thấp nhất là 0.999841 (ảnh) và 0.998703 (lời thoại). Đặt 0.999 thì
chính đoạn lời thoại dài nhất trượt, trong khi khác biệt đó không đủ để đổi
thứ hạng của bất kỳ kết quả nào.

Điều làm phép thử này có sức phân định: hai model KHÁC nhau cho cosine quanh 0
ở không gian 1024 chiều, chứ không phải 0.99. Nên mọi con số bắt đầu bằng 0.99
đều đã trả lời xong câu hỏi "có phải cùng model không".
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "API"))
sys.path.insert(0, str(REPO_ROOT / "embedding_server"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

# Xem phần "Đọc kết quả" ở docstring: hai ngưỡng này lấy từ số đo thật.
NGUONG_DAT = 0.998
NGUONG_NGO = 0.99


def _client():
    from qdrant_client import QdrantClient

    return QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ.get("QDRANT_API_KEY") or None,
        timeout=120,
    )


def _cosine(a: list[float], b: list[float]) -> float:
    import numpy as np

    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(va @ vb / denom) if denom else 0.0


class _BoNhung:
    """Gọi bản tự host: trực tiếp trong tiến trình, hoặc qua HTTP."""

    def __init__(self, url: str | None, api_key: str) -> None:
        self.url = url
        self.api_key = api_key
        self._encoder = None
        if url is None:
            from encoder import JinaClipEncoder

            self._encoder = JinaClipEncoder()

    def nhung_anh(self, anh_base64: list[str]) -> list[list[float]]:
        if self._encoder is not None:
            return self._encoder.encode_images(anh_base64)
        return self._qua_http({"input": [{"image": item} for item in anh_base64]})

    def nhung_text(self, texts: list[str]) -> list[list[float]]:
        if self._encoder is not None:
            return self._encoder.encode_texts(texts)
        return self._qua_http(
            {"input": [{"text": text} for text in texts], "task": "retrieval.query"}
        )

    def _qua_http(self, phan_them: dict[str, Any]) -> list[list[float]]:
        import httpx

        payload = {
            "model": os.getenv("EMBED_MODEL_NAME", "jina-clip-v2"),
            "embedding_type": "float",
            "dimensions": 1024,
            "normalized": True,
            **phan_them,
        }
        response = httpx.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=300.0,
        )
        response.raise_for_status()
        rows = sorted(response.json()["data"], key=lambda row: int(row.get("index", 0)))
        return [row["embedding"] for row in rows]


def kiem_anh(client, bo_nhung: _BoNhung, so_luong: int) -> list[float]:
    """Lấy ảnh thật từ MinIO, tiền xử lý y hệt lúc index, rồi nhúng lại."""
    from src.rag_video_anh.embedding.embedding_service import ImageEmbeddingService
    from src.rag_video_anh.pipeline.minio_storage import MinioStorage

    # Dùng lại đúng `_image_as_base64` của service thật. Tự viết lại bước thu
    # nhỏ ở đây là tự tạo ra một khác biệt không có thật rồi đi đo nó.
    service = ImageEmbeddingService()
    storage = MinioStorage()

    points, _ = client.scroll(
        collection_name="media_clip",
        limit=so_luong * 3,  # chừa dư, một số object có thể đã biến mất
        with_vectors=True,
        with_payload=True,
    )

    diem_so: list[float] = []
    with tempfile.TemporaryDirectory() as thu_muc:
        for point in points:
            if len(diem_so) >= so_luong:
                break
            payload = point.payload or {}
            bucket = payload.get("bucket_name")
            key = payload.get("frame_object_key")
            if not bucket or not key:
                continue
            duong_dan = Path(thu_muc) / "anh.jpg"
            try:
                storage.client.fget_object(bucket, key, str(duong_dan))
            except Exception as exc:
                print(f"  bỏ qua {key}: {exc.__class__.__name__}")
                continue

            moi = bo_nhung.nhung_anh([service._image_as_base64(duong_dan)])[0]
            cu = point.vector if isinstance(point.vector, list) else list(point.vector or [])
            cos = _cosine(cu, moi)
            diem_so.append(cos)
            print(f"  {cos:.6f}  {payload.get('media_kind', '?'):11} {key[-52:]}")
    return diem_so


def kiem_text(client, bo_nhung: _BoNhung, so_luong: int) -> list[float]:
    points, _ = client.scroll(
        collection_name="video_transcript",
        limit=so_luong,
        with_vectors=True,
        with_payload=True,
    )
    diem_so: list[float] = []
    for point in points:
        text = str((point.payload or {}).get("text") or "").strip()
        if not text:
            continue
        moi = bo_nhung.nhung_text([text])[0]
        cu = point.vector if isinstance(point.vector, list) else list(point.vector or [])
        cos = _cosine(cu, moi)
        diem_so.append(cos)
        print(f"  {cos:.6f}  {text[:56]}")
    return diem_so


def _tom_tat(ten: str, diem_so: list[float]) -> bool:
    if not diem_so:
        print(f"{ten:18} không đo được điểm nào")
        return False
    thap_nhat = min(diem_so)
    trung_binh = sum(diem_so) / len(diem_so)
    if thap_nhat >= NGUONG_DAT:
        ket_luan = "ĐẠT — cùng không gian, index cũ dùng tiếp được"
    elif thap_nhat >= NGUONG_NGO:
        ket_luan = "NGỜ — gần giống nhưng lệch, xem kỹ trước khi tin"
    else:
        ket_luan = "TRƯỢT — khác không gian, phải nhúng lại toàn bộ"
    print(f"{ten:18} n={len(diem_so):<3} thấp nhất={thap_nhat:.6f} trung bình={trung_binh:.6f}  {ket_luan}")
    return thap_nhat >= NGUONG_DAT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=None, help="endpoint /v1/embeddings; bỏ trống thì chạy model ngay trong tiến trình")
    parser.add_argument("--api-key", default=os.getenv("JINA_API_KEY", "local"))
    parser.add_argument("--images", type=int, default=5, help="số ảnh lấy mẫu từ media_clip")
    parser.add_argument("--texts", type=int, default=5, help="số đoạn lời thoại lấy mẫu từ video_transcript")
    args = parser.parse_args()

    client = _client()
    bo_nhung = _BoNhung(args.url, args.api_key)
    print(f"Đối chiếu với: {args.url or 'model chạy trong tiến trình'}\n")

    print("media_clip (ảnh + keyframe)")
    diem_anh = kiem_anh(client, bo_nhung, args.images) if args.images > 0 else []
    print("\nvideo_transcript (lời thoại)")
    diem_text = kiem_text(client, bo_nhung, args.texts) if args.texts > 0 else []

    print("\n" + "=" * 76)
    dat_anh = _tom_tat("media_clip", diem_anh)
    dat_text = _tom_tat("video_transcript", diem_text)
    print("=" * 76)
    # Mã thoát khác 0 để cắm được vào CI hoặc `&&` mà không phải đọc log.
    return 0 if (dat_anh and dat_text) else 1


if __name__ == "__main__":
    raise SystemExit(main())
