"""Phần dùng chung của hai script đăng ký media từ MinIO vào PostgreSQL.

Ảnh và video nằm cùng một bộ crawl nên cách đọc `events/<post>/post.json`,
cách suy ra post_id từ object key và cách dựng bản ghi post phải giống hệt
nhau — tách ra đây để hai script không trôi khác nhau.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.rag_video_anh.pipeline.minio_storage import MinioStorage
from src.rag_video_anh.repository import PostCreate

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def event_id_from_object_key(object_key: str) -> str:
    """events/<post_id>/media/photo_01.jpg -> <post_id>"""
    parts = object_key.strip("/").split("/")
    if len(parts) >= 2 and parts[0] == "events":
        return parts[1]
    return Path(object_key).parent.name or Path(object_key).stem


def list_objects_by_extension(storage: MinioStorage, prefix: str, extensions: set[str]) -> list[str]:
    return sorted(
        obj.object_name
        for obj in storage.client.list_objects(storage.bucket_name, prefix=prefix, recursive=True)
        if Path(obj.object_name).suffix.lower() in extensions
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


def parse_created_time(value) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def build_post_create(event_id: str, metadata: dict) -> PostCreate:
    """Dựng PostCreate từ post.json; không có metadata thì chỉ giữ facebook_post_id."""
    return PostCreate(
        facebook_post_id=event_id,
        content=metadata.get("message") or metadata.get("story"),
        author="facebook-crawler",
        post_url=metadata.get("permalink_url"),
        created_time=parse_created_time(metadata.get("created_time")),
    )


class PostMetadataResolver:
    """Upsert post theo event_id, đọc post.json đúng một lần cho mỗi bài."""

    def __init__(self, storage: MinioStorage, uow) -> None:
        self.storage = storage
        self.uow = uow
        self._metadata: dict[str, dict] = {}
        self._posts: dict[str, object] = {}

    def upsert(self, event_id: str):
        if event_id in self._posts:
            return self._posts[event_id]
        if event_id not in self._metadata:
            self._metadata[event_id] = load_post_metadata(self.storage, event_id)
        post = self.uow.posts.upsert_by_facebook_id(build_post_create(event_id, self._metadata[event_id]))
        self._posts[event_id] = post
        return post

    def has_metadata(self, event_id: str) -> bool:
        return bool(self._metadata.get(event_id))
