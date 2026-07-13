"""Pytest fixtures (CHO SẴN) — DB sqlite in-memory + TestClient.

Chạy:  cd api && pytest
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import shared.db.models  # noqa: F401  (đăng ký bảng vào Base.metadata)
from app.main import app
from shared.db.session import Base, get_session


@pytest.fixture
def engine():
    eng = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def db(engine):
    TestSession = sessionmaker(bind=engine)
    with TestSession() as session:
        yield session


@pytest.fixture
def client(engine):
    TestSession = sessionmaker(bind=engine)

    def override_get_session():
        with TestSession() as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session
    yield TestClient(app)
    app.dependency_overrides.clear()
