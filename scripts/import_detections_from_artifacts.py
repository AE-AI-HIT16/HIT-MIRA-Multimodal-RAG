#!/usr/bin/env python3
"""Nhập lại kết quả YOLO từ artifact RunPod trong MinIO vào PostgreSQL.

VÌ SAO CÓ SCRIPT RIÊNG thay vì dùng `import_media_outputs.py`:

Script kia nhập TẤT CẢ (frames + OCR + captions + detections + transcript) và
lặp qua MỌI frame, nên frame nào vắng mặt trong file JSON vẫn bị upsert với giá
trị `None`. Với bộ artifact hiện tại điều đó có hai hậu quả đo được:

  * `captions.json` trong cả 60 ZIP đều RỖNG (RunPod chỉ tách artifact, caption
    chạy sau ở máy này) → chạy nó là ghi `None` đè lên 3.289 caption keyframe.
  * `transcript.json` còn một bản `full_text` đúng bằng "<" — ảo giác của video
    câm mà `_co_noi_dung_doc_duoc` đã được vá để chặn. Nhập lại là mở lại cửa đó.

Script này chỉ chạm `object_results`/`detected_objects`, và chỉ chạm những frame
THỰC SỰ có mặt trong `detections.json`. Frame vắng mặt được giữ nguyên — đó
chính là lỗi đã xoá 17.262 object lần trước, nên ở đây nó là quy tắc.

MẶC ĐỊNH LÀ CHẠY THỬ. Muốn ghi DB phải truyền --apply:

    python scripts/import_detections_from_artifacts.py                  # chỉ đếm
    python scripts/import_detections_from_artifacts.py --apply --limit 2
    python scripts/import_detections_from_artifacts.py --apply

Cập nhật Qdrant là bước RIÊNG sau đó (payload-only, không nhúng lại):

    python scripts/index_video_retrieval_units.py <video_media_id> --payload-only
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "API"
sys.path.insert(0, str(API_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv(PROJECT_ROOT / ".env")

# Dùng lại đúng hai hàm của script nhập gốc thay vì chép: bảng quy đổi trạng
# thái và cách đọc bbox mà lệch nhau giữa hai script thì dữ liệu nhập vào sẽ
# khác nhau tuỳ ai chạy cái nào — đúng kiểu trôi dạt âm thầm.
from import_media_outputs import bbox_from_polygon, status  # noqa: E402

from src.configuration import AppConfig  # noqa: E402
from src.rag_video_anh.pipeline.minio_storage import MinioStorage  # noqa: E402
from src.rag_video_anh.repository import RepositoryUnitOfWork  # noqa: E402
from src.rag_video_anh.repository.schemas import DetectedObjectCreate  # noqa: E402

TIEN_TO_ARTIFACT = "runpod-artifacts/"


def liet_ke_artifact(storage: MinioStorage, bucket: str) -> list[str]:
    """Trả về khoá của mọi ZIP artifact, sắp xếp cho lượt chạy tất định."""
    khoa = [
        obj.object_name
        for obj in storage.client.list_objects(bucket, prefix=TIEN_TO_ARTIFACT, recursive=True)
        if obj.object_name.endswith(".zip")
    ]
    return sorted(khoa)


def doc_zip(storage: MinioStorage, bucket: str, khoa: str) -> zipfile.ZipFile:
    phan_hoi = storage.client.get_object(bucket, khoa)
    try:
        return zipfile.ZipFile(io.BytesIO(phan_hoi.read()))
    finally:
        phan_hoi.close()
        phan_hoi.release_conn()


def doc_json(zf: zipfile.ZipFile, ten: str) -> dict | None:
    ung_vien = [n for n in zf.namelist() if n.endswith(ten)]
    if not ung_vien:
        return None
    return json.loads(zf.read(ung_vien[0]))


def nhap_mot_video(khoa: str, zf: zipfile.ZipFile, apply: bool, dem: Counter) -> None:
    manifest = doc_json(zf, "manifest.json")
    detections = doc_json(zf, "detections.json")
    if manifest is None or detections is None:
        dem["zip_thieu_file"] += 1
        return

    video_media_id = str(manifest.get("media_id") or khoa.split("/")[1])

    # frame_id trong artifact ("<media_id>_f000005") không có trong DB; cầu nối
    # giữa hai bên là frame_index.
    index_theo_frame_id = {
        str(f["frame_id"]): int(f["frame_index"])
        for f in manifest.get("frames", [])
        if f.get("frame_id") is not None and f.get("frame_index") is not None
    }

    with RepositoryUnitOfWork() as uow:
        assert uow.media is not None and uow.results is not None
        media_id_theo_index = {
            f.frame_index: f.media_id
            for f in uow.media.list_frames_for_video(video_media_id)
            if f.frame_index is not None and f.media_id is not None
        }

        for muc in detections.get("results", []):
            frame_id = str(muc.get("frame_id") or "")
            frame_index = index_theo_frame_id.get(frame_id)
            if frame_index is None:
                dem["frame_khong_co_trong_manifest"] += 1
                continue
            frame_media_id = media_id_theo_index.get(frame_index)
            if frame_media_id is None:
                dem["frame_chua_dang_ky_trong_db"] += 1
                continue

            objects = [
                DetectedObjectCreate(
                    label=d.get("label"),
                    confidence=d.get("confidence"),
                    **bbox_from_polygon(d.get("bbox")),
                )
                for d in muc.get("detections", [])
            ]
            dem["frame_da_xu_ly"] += 1
            dem["object_da_nhap"] += len(objects)
            if not objects:
                dem["frame_that_su_khong_co_vat"] += 1

            if not apply:
                continue
            uow.results.upsert_object_result(
                frame_media_id,
                status=status(muc.get("status"), status(detections.get("status"))),
                model=(muc.get("inference_meta") or {}).get("model")
                or detections.get("model")
                or "unknown-detection",
                objects=objects,
            )
    dem["video_da_xu_ly"] += 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nhập detection từ artifact MinIO vào PostgreSQL.")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ xử lý N video đầu tiên.")
    parser.add_argument("--apply", action="store_true", help="Ghi DB thật. Không có cờ này thì chỉ đếm.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = AppConfig()
    storage = MinioStorage(config=config)
    bucket = storage.bucket_name

    khoa_list = liet_ke_artifact(storage, bucket)
    if args.limit is not None:
        khoa_list = khoa_list[: args.limit]
    print(f"Tìm thấy {len(khoa_list)} ZIP artifact trong '{bucket}/{TIEN_TO_ARTIFACT}'.")

    dem: Counter = Counter()
    for thu_tu, khoa in enumerate(khoa_list, start=1):
        try:
            zf = doc_zip(storage, bucket, khoa)
            nhap_mot_video(khoa, zf, args.apply, dem)
        except Exception as exc:  # noqa: BLE001
            dem["zip_hong"] += 1
            print(f"  LỖI {khoa}: {exc.__class__.__name__}: {exc}", flush=True)
            continue
        if thu_tu % 10 == 0:
            print(f"  ... {thu_tu}/{len(khoa_list)} video | {dict(dem)}", flush=True)

    print()
    for khoa_dem, gia_tri in sorted(dem.items()):
        print(f"  {khoa_dem}: {gia_tri}")
    if not args.apply:
        print("\nChưa ghi DB. Thêm --apply để thực hiện.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
