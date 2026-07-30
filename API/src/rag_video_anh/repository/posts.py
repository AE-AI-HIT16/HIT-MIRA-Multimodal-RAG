"""Repository for Facebook post metadata."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.rag_video_anh.repository.models import PostModel
from src.rag_video_anh.repository.schemas import PostCreate, PostRecord


class PostRepository:
    """Read/write access for the posts table."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, post: PostCreate) -> PostRecord:
        row = PostModel(**post.__dict__)
        self.session.add(row)
        self.session.flush()
        return self._to_record(row)

    def upsert_by_facebook_id(self, post: PostCreate) -> PostRecord:
        row = self.session.scalar(select(PostModel).where(PostModel.facebook_post_id == post.facebook_post_id))
        if row is None:
            return self.create(post)

        for field_name, value in post.__dict__.items():
            setattr(row, field_name, value)
        self.session.flush()
        return self._to_record(row)

    def get(self, post_id) -> PostRecord | None:
        row = self.session.get(PostModel, post_id)
        return self._to_record(row) if row else None

    def get_by_facebook_id(self, facebook_post_id: str) -> PostRecord | None:
        row = self.session.scalar(select(PostModel).where(PostModel.facebook_post_id == facebook_post_id))
        return self._to_record(row) if row else None

    @staticmethod
    def _to_record(row: PostModel) -> PostRecord:
        return PostRecord(
            post_id=row.post_id,
            facebook_post_id=row.facebook_post_id,
            content=row.content,
            author=row.author,
            post_url=row.post_url,
            created_time=row.created_time,
            crawl_time=row.crawl_time,
        )
