"""Unit-of-work facade for Repository usage."""

from __future__ import annotations

from types import TracebackType

from sqlalchemy.orm import Session

from src.rag_video_anh.repository.database import DatabaseSessionManager, get_session_manager
from src.rag_video_anh.repository.datasets import DatasetRepository
from src.rag_video_anh.repository.jobs import ProcessingJobRepository
from src.rag_video_anh.repository.media import MediaRepository
from src.rag_video_anh.repository.posts import PostRepository
from src.rag_video_anh.repository.results import AIResultRepository


class RepositoryUnitOfWork:
    """Transaction boundary exposing all repository groups."""

    def __init__(
        self,
        manager: DatabaseSessionManager | None = None,
        *,
        session: Session | None = None,
    ) -> None:
        # Dùng chung engine của cả tiến trình: dựng mới mỗi lần thì pool kết nối
        # cứ chồng lên nhau cho tới khi PostgreSQL từ chối (xem get_session_manager).
        self.manager = manager or get_session_manager()
        self._external_session = session
        self.session: Session | None = session
        self.datasets: DatasetRepository | None = None
        self.posts: PostRepository | None = None
        self.media: MediaRepository | None = None
        self.jobs: ProcessingJobRepository | None = None
        self.results: AIResultRepository | None = None

    def __enter__(self) -> RepositoryUnitOfWork:
        if self.session is None:
            self.session = self.manager.session_factory()
        self.datasets = DatasetRepository(self.session)
        self.posts = PostRepository(self.session)
        self.media = MediaRepository(self.session)
        self.jobs = ProcessingJobRepository(self.session)
        self.results = AIResultRepository(self.session)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._external_session is not None or self.session is None:
            return
        try:
            if exc_type is None:
                self.session.commit()
            else:
                self.session.rollback()
        finally:
            self.session.close()
            self.session = None

    def commit(self) -> None:
        if self.session is None:
            raise RuntimeError("RepositoryUnitOfWork is not active")
        self.session.commit()

    def rollback(self) -> None:
        if self.session is None:
            raise RuntimeError("RepositoryUnitOfWork is not active")
        self.session.rollback()
