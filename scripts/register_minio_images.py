"""Đăng ký ảnh đã có trên MinIO thành hàng media trong PostgreSQL.

Song song với `register_minio_videos.py`, khác hai điểm:
  * tạo hàng media_type='image';
  * đọc `events/<post>/post.json` để lấy nội dung bài thật (message, link,
    ngày đăng) thay vì ghi chuỗi giữ chỗ.

MẶC ĐỊNH LÀ CHẠY THỬ. Muốn ghi thật phải truyền --apply:

    python scripts/register_minio_images.py                 # chỉ xem trước
    python scripts/register_minio_images.py --apply         # ghi vào DB

Sau khi đăng ký, chạy phân tích và index bằng:

    python -m src.rag_video_anh.pipeline.image_processing_worker <media_id>
    python scripts/index_image_units.py --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "API"
sys.path.insert(0, str(API_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.rag_video_anh.pipeline.minio_storage import MinioStorage  # noqa: E402
from src.rag_video_anh.repository import (  # noqa: E402
    MediaCreate,
    MediaType,
    PostCreate,
    RepositoryUnitOfWork,
)
from src.rag_video_anh.repository.models import MediaModel  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def event_id_from_object_key(object_key: str) -> str:
    parts = object_key.strip("/").split("/")
    if len(parts) >= 2 and parts[0] == "events":
        return parts[1]
    return Path(object_key).parent.name or Path(object_key).stem


def list_minio_images(storage: MinioStorage, prefix: str = "") -> list[str]:
    return sorted(
        obj.object_name
        for obj in storage.client.list_objects(storage.bucket_name, prefix=prefix, recursive=True)
        if any(obj.object_name.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)
    )


def load_post_metadata(storage: MinioStorage, event_id: str) -> dict:
    """Đọc post.json của bài; thiếu file cũng không được làm hỏng cả lượt chạy."""
    object_key = f"events/{event_id}/post.json"
    try:
        response = storage.client.get_object(storage.bucket_name, object_key)
        try:
            return json.loads(response.read().decode("utf-8"))
        finally:
            response.close()
            response.release_conn()
    except Exception as exc:
        print(f"  ! không đọc được {object_key}: {exc.__class__.__name__}")
        return {}


def _parse_created_time(value) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def register_images(prefix: str = "events/", apply: bool = False) -> dict:
    storage = MinioStorage()
    object_keys = list_minio_images(storage, prefix=prefix)
    created = 0
    skipped = 0
    posts_touched: set[str] = set()
    metadata_cache: dict[str, dict] = {}

    with RepositoryUnitOfWork() as uow:
        if uow.session is None or uow.posts is None or uow.media is None:
            raise RuntimeError("RepositoryUnitOfWork did not expose session/posts/media")

        for object_key in object_keys:
            exists = uow.session.scalar(
                select(MediaModel)
                .where(
                    MediaModel.media_type == MediaType.IMAGE.value,
                    MediaModel.bucket_name == storage.bucket_name,
                    MediaModel.object_key == object_key,
                )
                .limit(1)
            )
            if exists is not None:
                skipped += 1
                continue

            event_id = event_id_from_object_key(object_key)
            posts_touched.add(event_id)
            created += 1
            if not apply:
                continue

            if event_id not in metadata_cache:
                metadata_cache[event_id] = load_post_metadata(storage, event_id)
            metadata = metadata_cache[event_id]

            post = uow.posts.upsert_by_facebook_id(
                PostCreate(
                    facebook_post_id=event_id,
                    content=metadata.get("message") or metadata.get("story"),
                    author="facebook-crawler",
                    post_url=metadata.get("permalink_url"),
                    created_time=_parse_created_time(metadata.get("created_time")),
                )
            )
            uow.media.create_media(
                MediaCreate(
                    post_id=post.post_id,
                    media_type=MediaType.IMAGE.value,
                    bucket_name=storage.bucket_name,
                    object_key=object_key,
                )
            )

    return {
        "found": len(object_keys),
        "created": created,
        "skipped": skipped,
        "posts": len(posts_touched),
        "applied": apply,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Đăng ký ảnh trên MinIO vào bảng media của PostgreSQL.")
    parser.add_argument("--prefix", default="events/", help="Chỉ quét object dưới prefix này. Mặc định: events/")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Ghi thật vào DB. Không truyền cờ này thì chỉ xem trước, không đổi gì.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = register_images(prefix=args.prefix, apply=args.apply)
    action = "Đã tạo" if summary["applied"] else "SẼ tạo (chạy thử)"
    print(f"Tìm thấy {summary['found']} ảnh trên MinIO thuộc {summary['posts']} bài.")
    print(f"{action} {summary['created']} hàng media; bỏ qua {summary['skipped']} hàng đã có.")
    if not summary["applied"]:
        print("Chưa ghi gì vào DB. Thêm --apply để thực hiện.")


if __name__ == "__main__":
    main()
