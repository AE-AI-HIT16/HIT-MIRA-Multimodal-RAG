"""Database session management for the Repository layer."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.rag_video_anh.repository.models import Base

DEFAULT_DATABASE_URL = "postgresql+psycopg://hit:hit@localhost:5432/hit_mira"


def get_database_url(default: str = DEFAULT_DATABASE_URL) -> str:
    """Return the configured SQLAlchemy database URL."""

    return os.getenv("DATABASE_URL", default)


class DatabaseSessionManager:
    """Owns the SQLAlchemy engine and provides transaction-scoped sessions."""

    def __init__(
        self,
        database_url: str | None = None,
        *,
        engine: Engine | None = None,
        echo: bool = False,
        expire_on_commit: bool = False,
    ) -> None:
        # pool_pre_ping: worker chạy hàng giờ, kết nối nằm không dễ bị phía
        # PostgreSQL đóng; kiểm tra trước khi dùng rẻ hơn là để lỗi bật ra giữa
        # một lô đang chạy dở.
        self.engine = engine or create_engine(
            database_url or get_database_url(),
            echo=echo,
            future=True,
            pool_pre_ping=True,
        )
        self.session_factory = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=expire_on_commit,
            future=True,
        )

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Yield one committing/rolling-back unit of database work."""

        db_session = self.session_factory()
        try:
            yield db_session
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()

    def create_all(self) -> None:
        """Create mapped tables. Prefer migrations in production."""

        Base.metadata.create_all(self.engine)

    def drop_all(self) -> None:
        """Drop mapped tables. Intended for tests and local reset only."""

        Base.metadata.drop_all(self.engine)


_default_manager: DatabaseSessionManager | None = None
_default_manager_lock = Lock()


def get_session_manager() -> DatabaseSessionManager:
    """Trả manager dùng chung cho cả tiến trình.

    Mỗi DatabaseSessionManager dựng một engine kèm pool kết nối riêng, và
    engine thì không tự đóng. Nếu mỗi UnitOfWork lại dựng một cái mới thì chỉ
    vài trăm lượt là PostgreSQL trả 'sorry, too many clients already' — đã xảy
    ra thật khi chạy caption 6 luồng. Một engine dùng chung là cách SQLAlchemy
    khuyến nghị: pool lo phần tái sử dụng kết nối.
    """
    global _default_manager
    if _default_manager is None:
        with _default_manager_lock:
            if _default_manager is None:
                _default_manager = DatabaseSessionManager()
    return _default_manager


def reset_session_manager() -> None:
    """Bỏ manager dùng chung và đóng engine của nó (dùng cho test/đổi cấu hình)."""
    global _default_manager
    with _default_manager_lock:
        if _default_manager is not None:
            _default_manager.engine.dispose()
        _default_manager = None
