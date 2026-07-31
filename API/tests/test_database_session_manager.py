"""Engine phải dùng chung cho cả tiến trình.

Bản cũ dựng một DatabaseSessionManager — tức một engine kèm pool kết nối riêng
— cho MỖI UnitOfWork, và engine thì không tự đóng. Chạy caption 6 luồng được
134 ảnh thì PostgreSQL trả 'sorry, too many clients already'.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from src.rag_video_anh.repository.database import (
    DatabaseSessionManager,
    get_session_manager,
    reset_session_manager,
)
from src.rag_video_anh.repository.unit_of_work import RepositoryUnitOfWork


def test_session_manager_is_shared_across_calls() -> None:
    reset_session_manager()
    try:
        first = get_session_manager()
        second = get_session_manager()

        assert first is second
        assert first.engine is second.engine
    finally:
        reset_session_manager()


def test_unit_of_work_reuses_one_engine_instead_of_creating_one_each_time() -> None:
    """Mỗi UnitOfWork một engine thì vài trăm lượt là hết chỗ kết nối."""
    reset_session_manager()
    try:
        engines = {id(RepositoryUnitOfWork().manager.engine) for _ in range(25)}

        assert len(engines) == 1
    finally:
        reset_session_manager()


def test_explicit_manager_still_wins_over_the_shared_one() -> None:
    """Test và script vẫn phải trỏ được vào DB riêng."""
    reset_session_manager()
    own = DatabaseSessionManager(
        engine=create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    )
    try:
        uow = RepositoryUnitOfWork(manager=own)

        assert uow.manager is own
        assert uow.manager.engine is not get_session_manager().engine
    finally:
        own.engine.dispose()
        reset_session_manager()


def test_reset_disposes_the_shared_engine() -> None:
    reset_session_manager()
    first = get_session_manager()

    reset_session_manager()

    assert get_session_manager() is not first
    reset_session_manager()
