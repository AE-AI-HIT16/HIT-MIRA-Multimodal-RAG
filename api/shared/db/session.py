"""Kết nối DB + session factory (hạ tầng — CHO SẴN).

Package top-level khi chạy từ api/: `app`, `pipeline`, `shared`.
Chạy dev:  cd api && uvicorn app.main:app --reload
"""
from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    """Base cho mọi ORM model (shared/db/models.py)."""


engine = create_engine(settings.database_url, future=True, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, class_=Session, autoflush=False, expire_on_commit=False)


def get_session() -> Iterator[Session]:
    """FastAPI dependency: mở/đóng 1 session mỗi request."""
    with SessionLocal() as session:
        yield session
