"""SQLAlchemy ORM models matching schema.sql."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import CHAR, TypeDecorator


class GUID(TypeDecorator):
    """Platform-independent UUID type.

    PostgreSQL uses native UUID. SQLite stores canonical 36-character strings,
    which keeps lightweight tests possible without changing repository code.
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PostgreSQLUUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if not isinstance(value, uuid.UUID):
            value = uuid.UUID(str(value))
        if dialect.name == "postgresql":
            return value
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None or isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))


class Base(DeclarativeBase):
    pass


class PostModel(Base):
    __tablename__ = "posts"

    post_id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    facebook_post_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    content: Mapped[str | None] = mapped_column(Text)
    author: Mapped[str | None] = mapped_column(String(255))
    post_url: Mapped[str | None] = mapped_column(Text)
    created_time: Mapped[datetime | None] = mapped_column(DateTime)
    crawl_time: Mapped[datetime | None] = mapped_column(DateTime, server_default=func.current_timestamp())

    media_items: Mapped[list[MediaModel]] = relationship(back_populates="post", cascade="all, delete-orphan")


class MediaModel(Base):
    __tablename__ = "media"
    __table_args__ = (Index("idx_media_type", "media_type"),)

    media_id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    post_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("posts.post_id", ondelete="CASCADE"), nullable=False)
    media_type: Mapped[str] = mapped_column(String(20), nullable=False)
    bucket_name: Mapped[str | None] = mapped_column(String(100), default="mira-data")
    object_key: Mapped[str] = mapped_column(Text, nullable=False)
    parent_media_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), ForeignKey("media.media_id"))
    created_at: Mapped[datetime | None] = mapped_column(DateTime, server_default=func.current_timestamp())

    post: Mapped[PostModel] = relationship(back_populates="media_items")
    parent: Mapped[MediaModel | None] = relationship(remote_side=[media_id], back_populates="children")
    children: Mapped[list[MediaModel]] = relationship(back_populates="parent")
    video: Mapped[VideoModel | None] = relationship(back_populates="media", uselist=False, cascade="all, delete-orphan")
    frame: Mapped[FrameModel | None] = relationship(
        back_populates="media",
        uselist=False,
        cascade="all, delete-orphan",
        foreign_keys="FrameModel.media_id",
    )
    ocr_result: Mapped[OcrResultModel | None] = relationship(back_populates="media", uselist=False, cascade="all, delete-orphan")
    caption_result: Mapped[CaptionResultModel | None] = relationship(back_populates="media", uselist=False, cascade="all, delete-orphan")
    object_result: Mapped[ObjectResultModel | None] = relationship(back_populates="media", uselist=False, cascade="all, delete-orphan")
    embedding: Mapped[EmbeddingModel | None] = relationship(back_populates="media", uselist=False, cascade="all, delete-orphan")
    jobs: Mapped[list[ProcessingJobModel]] = relationship(back_populates="media", cascade="all, delete-orphan")


class VideoModel(Base):
    __tablename__ = "videos"

    video_id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    media_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("media.media_id", ondelete="CASCADE"), unique=True, nullable=False)
    duration: Mapped[float | None] = mapped_column(Float)
    fps: Mapped[float | None] = mapped_column(Float)
    raw_frames: Mapped[int | None] = mapped_column(Integer)
    selected_keyframes: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, server_default=func.current_timestamp())

    media: Mapped[MediaModel] = relationship(back_populates="video")
    transcripts: Mapped[TranscriptModel | None] = relationship(back_populates="video", uselist=False, cascade="all, delete-orphan")


class FrameModel(Base):
    __tablename__ = "frames"
    __table_args__ = (Index("idx_frames_video", "video_media_id"),)

    frame_id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    media_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("media.media_id", ondelete="CASCADE"), unique=True, nullable=False)
    video_media_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("media.media_id"), nullable=False)
    timestamp: Mapped[float | None] = mapped_column(Float)
    frame_index: Mapped[int | None] = mapped_column(Integer)

    media: Mapped[MediaModel] = relationship(back_populates="frame", foreign_keys=[media_id])
    video_media: Mapped[MediaModel] = relationship(foreign_keys=[video_media_id])


class OcrResultModel(Base):
    __tablename__ = "ocr_results"

    ocr_id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    media_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("media.media_id", ondelete="CASCADE"), unique=True, nullable=False)
    ocr_status: Mapped[str | None] = mapped_column(String(20), default="PENDING")
    ocr_text: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, server_default=func.current_timestamp())

    media: Mapped[MediaModel] = relationship(back_populates="ocr_result")


class CaptionResultModel(Base):
    __tablename__ = "caption_results"

    caption_id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    media_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("media.media_id", ondelete="CASCADE"), unique=True, nullable=False)
    caption_status: Mapped[str | None] = mapped_column(String(20), default="PENDING")
    caption_text: Mapped[str | None] = mapped_column(Text)
    caption_model: Mapped[str | None] = mapped_column(String(100))
    vision_metadata: Mapped[dict | None] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, server_default=func.current_timestamp())

    media: Mapped[MediaModel] = relationship(back_populates="caption_result")


class ObjectResultModel(Base):
    __tablename__ = "object_results"

    object_result_id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    media_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("media.media_id", ondelete="CASCADE"), unique=True, nullable=False)
    status: Mapped[str | None] = mapped_column(String(20), default="PENDING")
    model: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime | None] = mapped_column(DateTime, server_default=func.current_timestamp())

    media: Mapped[MediaModel] = relationship(back_populates="object_result")
    objects: Mapped[list[DetectedObjectModel]] = relationship(back_populates="object_result", cascade="all, delete-orphan")


class DetectedObjectModel(Base):
    __tablename__ = "detected_objects"

    object_id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    object_result_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("object_results.object_result_id", ondelete="CASCADE"),
        nullable=False,
    )
    label: Mapped[str | None] = mapped_column(String(100))
    confidence: Mapped[float | None] = mapped_column(Float)
    x1: Mapped[float | None] = mapped_column(Float)
    y1: Mapped[float | None] = mapped_column(Float)
    x2: Mapped[float | None] = mapped_column(Float)
    y2: Mapped[float | None] = mapped_column(Float)

    object_result: Mapped[ObjectResultModel] = relationship(back_populates="objects")


class TranscriptModel(Base):
    __tablename__ = "transcripts"

    transcript_id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("videos.video_id", ondelete="CASCADE"), unique=True, nullable=False)
    status: Mapped[str | None] = mapped_column(String(20), default="PENDING")
    language: Mapped[str | None] = mapped_column(String(20))
    model: Mapped[str | None] = mapped_column(String(100))
    full_text: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, server_default=func.current_timestamp())

    video: Mapped[VideoModel] = relationship(back_populates="transcripts")
    segments: Mapped[list[TranscriptSegmentModel]] = relationship(back_populates="transcript", cascade="all, delete-orphan")


class TranscriptSegmentModel(Base):
    __tablename__ = "transcript_segments"

    segment_id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    transcript_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("transcripts.transcript_id", ondelete="CASCADE"), nullable=False)
    start_time: Mapped[float | None] = mapped_column(Float)
    end_time: Mapped[float | None] = mapped_column(Float)
    text: Mapped[str | None] = mapped_column(Text)

    transcript: Mapped[TranscriptModel] = relationship(back_populates="segments")


class ProcessingJobModel(Base):
    __tablename__ = "processing_jobs"
    __table_args__ = (Index("idx_jobs_worker", "task_type", "status"),)

    job_id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    media_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("media.media_id", ondelete="CASCADE"), nullable=False)
    task_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str | None] = mapped_column(String(20), default="PENDING")
    retry_count: Mapped[int | None] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, server_default=func.current_timestamp())
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)

    media: Mapped[MediaModel] = relationship(back_populates="jobs")


class EmbeddingModel(Base):
    __tablename__ = "embeddings"

    embedding_id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    media_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("media.media_id", ondelete="CASCADE"), unique=True, nullable=False)
    vector_db_id: Mapped[str | None] = mapped_column(String(255))
    model: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime | None] = mapped_column(DateTime, server_default=func.current_timestamp())

    media: Mapped[MediaModel] = relationship(back_populates="embedding")
