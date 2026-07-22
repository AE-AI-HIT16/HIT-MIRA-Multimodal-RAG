"""Register existing MinIO video objects in PostgreSQL.

The worker queue reads PostgreSQL, not MinIO. Run this script after uploading
or discovering videos in MinIO so every object has a media row.
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
load_dotenv(PROJECT_ROOT / ".env")

from src.rag_video_anh.pipeline.minio_storage import MinioStorage  # noqa: E402
from src.rag_video_anh.repository import MediaCreate, MediaType, PostCreate, RepositoryUnitOfWork  # noqa: E402
from src.rag_video_anh.repository.models import MediaModel  # noqa: E402


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def event_id_from_object_key(object_key: str) -> str:
    parts = object_key.strip("/").split("/")
    if len(parts) >= 2 and parts[0] == "events":
        return parts[1]
    return Path(object_key).stem


def list_minio_videos(storage: MinioStorage, prefix: str = "") -> list[str]:
    return sorted(
        obj.object_name
        for obj in storage.client.list_objects(storage.bucket_name, prefix=prefix, recursive=True)
        if any(obj.object_name.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)
    )


def register_videos(prefix: str = "", dry_run: bool = False) -> tuple[int, int, list[str]]:
    storage = MinioStorage()
    object_keys = list_minio_videos(storage, prefix=prefix)
    created = 0
    skipped = 0

    with RepositoryUnitOfWork() as uow:
        assert uow.session is not None
        assert uow.posts is not None
        assert uow.media is not None

        for object_key in object_keys:
            exists = uow.session.scalar(
                select(MediaModel)
                .where(
                    MediaModel.media_type == MediaType.VIDEO.value,
                    MediaModel.bucket_name == storage.bucket_name,
                    MediaModel.object_key == object_key,
                )
                .limit(1)
            )
            if exists is not None:
                skipped += 1
                continue

            if dry_run:
                created += 1
                continue

            event_id = event_id_from_object_key(object_key)
            post = uow.posts.upsert_by_facebook_id(
                PostCreate(
                    facebook_post_id=event_id,
                    content=f"Registered from MinIO object {object_key}",
                    author="minio-sync",
                )
            )
            uow.media.create_media(
                MediaCreate(
                    post_id=post.post_id,
                    media_type=MediaType.VIDEO.value,
                    bucket_name=storage.bucket_name,
                    object_key=object_key,
                )
            )
            created += 1
    return created, skipped, object_keys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Register MinIO video objects in PostgreSQL media table.")
    parser.add_argument("--prefix", default="", help="Only scan MinIO objects below this prefix.")
    parser.add_argument("--dry-run", action="store_true", help="Show how many media rows would be created.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    created, skipped, object_keys = register_videos(prefix=args.prefix, dry_run=args.dry_run)
    action = "Would create" if args.dry_run else "Created"
    print(f"Found {len(object_keys)} video object(s) in MinIO.")
    print(f"{action} {created} media row(s); skipped {skipped} existing row(s).")
    for object_key in object_keys:
        print(object_key)


if __name__ == "__main__":
    main()
