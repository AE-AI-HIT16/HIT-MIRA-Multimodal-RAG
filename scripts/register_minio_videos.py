"""Đăng ký video đã có trên MinIO thành hàng media trong PostgreSQL.

Hàng đợi worker đọc PostgreSQL chứ không đọc MinIO, nên mỗi object video phải
có một hàng media tương ứng.

Giống `register_minio_images.py`, script đọc `events/<post>/post.json` để lấy
nội dung bài thật thay vì ghi chuỗi giữ chỗ. Metadata post được upsert kể cả
khi hàng media đã tồn tại, nên chạy lại sẽ vá được những bài đăng ký bằng bản
script cũ.

MẶC ĐỊNH LÀ CHẠY THỬ. Muốn ghi thật phải truyền --apply:

    python scripts/register_minio_videos.py            # chỉ xem trước
    python scripts/register_minio_videos.py --apply    # ghi vào DB
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
    VIDEO_EXTENSIONS,
    PostMetadataResolver,
    event_id_from_object_key,
    list_objects_by_extension,
)

from src.rag_video_anh.pipeline.minio_storage import MinioStorage  # noqa: E402
from src.rag_video_anh.repository import MediaCreate, MediaType, RepositoryUnitOfWork  # noqa: E402
from src.rag_video_anh.repository.models import MediaModel  # noqa: E402


def register_videos(prefix: str = "events/", apply: bool = False) -> dict:
    storage = MinioStorage()
    object_keys = list_objects_by_extension(storage, prefix, VIDEO_EXTENSIONS)
    created = 0
    skipped = 0
    posts_touched: set[str] = set()
    missing_metadata: set[str] = set()

    with RepositoryUnitOfWork() as uow:
        if uow.session is None or uow.posts is None or uow.media is None:
            raise RuntimeError("RepositoryUnitOfWork did not expose session/posts/media")

        resolver = PostMetadataResolver(storage, uow)

        for object_key in object_keys:
            event_id = event_id_from_object_key(object_key)
            posts_touched.add(event_id)

            exists = uow.session.scalar(
                select(MediaModel)
                .where(
                    MediaModel.media_type == MediaType.VIDEO.value,
                    MediaModel.bucket_name == storage.bucket_name,
                    MediaModel.object_key == object_key,
                )
                .limit(1)
            )
            if exists is None:
                created += 1
            else:
                skipped += 1

            if not apply:
                continue

            # Upsert post kể cả khi media đã có, để vá nội dung giữ chỗ của bản cũ.
            post = resolver.upsert(event_id)
            if not resolver.has_metadata(event_id):
                missing_metadata.add(event_id)
            if exists is None:
                uow.media.create_media(
                    MediaCreate(
                        post_id=post.post_id,
                        media_type=MediaType.VIDEO.value,
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
    parser = argparse.ArgumentParser(description="Đăng ký video trên MinIO vào bảng media của PostgreSQL.")
    parser.add_argument("--prefix", default="events/", help="Chỉ quét object dưới prefix này. Mặc định: events/")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Ghi thật vào DB. Không truyền cờ này thì chỉ xem trước, không đổi gì.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = register_videos(prefix=args.prefix, apply=args.apply)
    action = "Đã tạo" if summary["applied"] else "SẼ tạo (chạy thử)"
    print(f"Tìm thấy {summary['found']} video trên MinIO thuộc {summary['posts']} bài.")
    print(f"{action} {summary['created']} hàng media; bỏ qua {summary['skipped']} hàng đã có.")
    if summary["applied"]:
        print(f"Đã làm mới metadata cho {summary['posts']} bài.")
        if summary["missing_metadata"]:
            print(f"  ! {len(summary['missing_metadata'])} bài không đọc được post.json: {summary['missing_metadata']}")
    else:
        print("Chưa ghi gì vào DB. Thêm --apply để thực hiện.")


if __name__ == "__main__":
    main()
