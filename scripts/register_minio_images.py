"""Đăng ký ảnh đã có trên MinIO thành hàng media trong PostgreSQL.

Song song với `register_minio_videos.py`, khác mỗi chỗ tạo hàng
media_type='image'. Phần đọc `events/<post>/post.json` để lấy nội dung bài
thật (message, link, ngày đăng) nằm trong `minio_registration.py`.

MẶC ĐỊNH LÀ CHẠY THỬ. Muốn ghi thật phải truyền --apply:

    python scripts/register_minio_images.py                 # chỉ xem trước
    python scripts/register_minio_images.py --apply         # ghi vào DB

Sau khi đăng ký, chạy phân tích và index bằng:

    python -m src.rag_video_anh.pipeline.image_processing_worker <media_id>
    python scripts/index_image_units.py --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "API"
sys.path.insert(0, str(API_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
load_dotenv(PROJECT_ROOT / ".env")

from minio_registration import (  # noqa: E402
    IMAGE_EXTENSIONS,
    PostMetadataResolver,
    event_id_from_object_key,
    list_objects_by_extension,
)

from src.rag_video_anh.pipeline.minio_storage import MinioStorage  # noqa: E402
from src.rag_video_anh.repository import MediaCreate, MediaType, RepositoryUnitOfWork  # noqa: E402
from src.rag_video_anh.repository.models import MediaModel  # noqa: E402


def register_images(prefix: str = "events/", apply: bool = False) -> dict:
    storage = MinioStorage()
    object_keys = list_objects_by_extension(storage, prefix, IMAGE_EXTENSIONS)
    created = 0
    skipped = 0
    posts_touched: set[str] = set()
    missing_metadata: set[str] = set()

    with RepositoryUnitOfWork() as uow:
        if uow.session is None or uow.posts is None or uow.media is None:
            raise RuntimeError("RepositoryUnitOfWork did not expose session/posts/media")

        resolver = PostMetadataResolver(storage, uow)

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

            post = resolver.upsert(event_id)
            if not resolver.has_metadata(event_id):
                missing_metadata.add(event_id)
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
        "missing_metadata": sorted(missing_metadata),
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
    elif summary["missing_metadata"]:
        print(f"  ! {len(summary['missing_metadata'])} bài không đọc được post.json: {summary['missing_metadata']}")


if __name__ == "__main__":
    main()
