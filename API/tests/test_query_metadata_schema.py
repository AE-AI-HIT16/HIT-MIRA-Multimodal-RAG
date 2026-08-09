"""Schema catalog và taxonomy cho truy vấn sự kiện lặp lại."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from src.rag_video_anh.repository import DatasetCreate, PostCreate, RepositoryUnitOfWork
from src.rag_video_anh.repository.models import (
    Base,
    DatasetModel,
    EventAliasModel,
    EventOccurrenceModel,
    EventSeriesModel,
    MediaModel,
    PostEventOccurrenceModel,
    PostModel,
)


@pytest.fixture()
def session() -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _foreign_keys(dbapi_connection, _connection_record) -> None:
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    db_session = Session(engine)
    yield db_session
    db_session.close()
    engine.dispose()


def _dataset(session: Session):
    with RepositoryUnitOfWork(session=session) as uow:
        assert uow.datasets is not None
        return uow.datasets.upsert_by_location(
            DatasetCreate(
                name="raw/google-drive/data",
                bucket_name="hit-mira-media",
                object_prefix="raw/google-drive/data",
            )
        )


def _post(session: Session, facebook_id: str = "facebook-post-1") -> PostModel:
    dataset = _dataset(session)
    row = PostModel(facebook_post_id=facebook_id, dataset_id=dataset.dataset_id)
    session.add(row)
    session.flush()
    return row


def _series(session: Session) -> EventSeriesModel:
    row = EventSeriesModel(slug="tuyen-thanh-vien", canonical_name="Tuyển thành viên")
    session.add(row)
    session.flush()
    return row


def test_dataset_upsert_theo_bucket_va_prefix(session: Session) -> None:
    first = _dataset(session)
    second = _dataset(session)

    assert first.dataset_id == second.dataset_id
    assert session.scalar(select(func.count()).select_from(DatasetModel)) == 1


def test_post_khong_co_event_van_hop_le(session: Session) -> None:
    post = _post(session)
    session.commit()

    assert post.event_links == []


def test_occurrence_khoa_theo_label_khong_khoa_theo_nam(session: Session) -> None:
    series = _series(session)
    session.add_all(
        [
            EventOccurrenceModel(series_id=series.series_id, label="HIT-15", event_year=2025),
            EventOccurrenceModel(series_id=series.series_id, label="HIT-15-DOT-1", event_year=2025),
        ]
    )
    session.commit()

    assert session.scalar(select(func.count()).select_from(EventOccurrenceModel)) == 2


def test_occurrence_trung_label_trong_cung_series_bi_chan(session: Session) -> None:
    series = _series(session)
    session.add_all(
        [
            EventOccurrenceModel(series_id=series.series_id, label="HIT-15"),
            EventOccurrenceModel(series_id=series.series_id, label="HIT-15"),
        ]
    )

    with pytest.raises(IntegrityError):
        session.commit()


def test_alias_chi_duoc_tro_vao_mot_cap(session: Session) -> None:
    series = _series(session)
    occurrence = EventOccurrenceModel(series_id=series.series_id, label="HIT-15", event_year=2025)
    session.add(occurrence)
    session.flush()
    session.add(
        EventAliasModel(
            series_id=series.series_id,
            occurrence_id=occurrence.occurrence_id,
            alias="tuyển thành viên 2024",
            normalized_alias="tuyen thanh vien 2024",
        )
    )

    with pytest.raises(IntegrityError):
        session.commit()


def test_chi_mot_occurrence_chinh_moi_post(session: Session) -> None:
    post = _post(session)
    series = _series(session)
    first = EventOccurrenceModel(series_id=series.series_id, label="HIT-15")
    second = EventOccurrenceModel(series_id=series.series_id, label="HIT-15-DOT-1")
    session.add_all([first, second])
    session.flush()
    session.add_all(
        [
            PostEventOccurrenceModel(
                post_id=post.post_id,
                occurrence_id=first.occurrence_id,
                is_primary=True,
            ),
            PostEventOccurrenceModel(
                post_id=post.post_id,
                occurrence_id=second.occurrence_id,
                is_primary=True,
            ),
        ]
    )

    with pytest.raises(IntegrityError):
        session.commit()


def test_media_bucket_object_la_unique_nhung_hash_khong_unique(session: Session) -> None:
    post = _post(session)
    same_hash = "a" * 64
    session.add_all(
        [
            MediaModel(
                post_id=post.post_id,
                media_type="image",
                bucket_name="hit-mira-media",
                object_key="raw/a.jpg",
                content_sha256=same_hash,
            ),
            MediaModel(
                post_id=post.post_id,
                media_type="image",
                bucket_name="hit-mira-media",
                object_key="raw/b.jpg",
                content_sha256=same_hash,
            ),
        ]
    )
    session.commit()
    assert session.scalar(select(func.count()).select_from(MediaModel)) == 2

    session.add(
        MediaModel(
            media_id=uuid.uuid4(),
            post_id=post.post_id,
            media_type="image",
            bucket_name="hit-mira-media",
            object_key="raw/a.jpg",
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()


def test_upsert_post_khong_xoa_dataset_khi_caller_cu_khong_biet_dataset(session: Session) -> None:
    dataset = _dataset(session)
    with RepositoryUnitOfWork(session=session) as uow:
        assert uow.posts is not None
        created = uow.posts.upsert_by_facebook_id(
            PostCreate(facebook_post_id="facebook-post-1", dataset_id=dataset.dataset_id)
        )
        updated = uow.posts.upsert_by_facebook_id(
            PostCreate(facebook_post_id="facebook-post-1", content="nội dung mới")
        )

    assert created.post_id == updated.post_id
    assert updated.dataset_id == dataset.dataset_id
