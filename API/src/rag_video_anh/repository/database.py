"""Database session management for the Repository layer."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

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
        self.engine = engine or create_engine(database_url or get_database_url(), echo=echo, future=True)
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
