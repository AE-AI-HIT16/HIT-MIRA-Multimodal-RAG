"""Repository DTOs and status vocabulary for media metadata persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID


class ProcessingStatus(str, Enum):
    """Status values persisted by result tables and worker jobs."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    FAILED = "FAILED"


class MediaType(str, Enum):
    """Supported media rows from schema.sql."""

    IMAGE = "image"
    VIDEO = "video"
    FRAME = "frame"


class TaskType(str, Enum):
    """Worker task types from processing_jobs.task_type."""

    KEYFRAME = "keyframe"
    OCR = "ocr"
    CAPTION = "caption"
    OBJECT_DETECTION = "object_detection"
    ASR = "asr"
    EMBEDDING = "embedding"


@dataclass(frozen=True)
class PostCreate:
    facebook_post_id: str
    content: str | None = None
    author: str | None = None
    post_url: str | None = None
    created_time: datetime | None = None


@dataclass(frozen=True)
class PostRecord(PostCreate):
    post_id: UUID | None = None
    crawl_time: datetime | None = None


@dataclass(frozen=True)
class MediaCreate:
    post_id: UUID
    media_type: str
    object_key: str
    bucket_name: str = "mira-data"
    parent_media_id: UUID | None = None


@dataclass(frozen=True)
class MediaRecord(MediaCreate):
    media_id: UUID | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class VideoMetadataCreate:
    media_id: UUID
    duration: float | None = None
    fps: float | None = None
    raw_frames: int | None = None
    selected_keyframes: int | None = None


@dataclass(frozen=True)
class VideoMetadataRecord(VideoMetadataCreate):
    video_id: UUID | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class FrameCreate:
    video_media_id: UUID
    object_key: str
    timestamp: float | None = None
    frame_index: int | None = None
    bucket_name: str = "mira-data"


@dataclass(frozen=True)
class FrameRecord(FrameCreate):
    frame_id: UUID | None = None
    media_id: UUID | None = None
    post_id: UUID | None = None
    parent_media_id: UUID | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class OcrResultRecord:
    media_id: UUID
    ocr_status: str = ProcessingStatus.PENDING.value
    ocr_text: str | None = None
    ocr_id: UUID | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class CaptionResultRecord:
    media_id: UUID
    caption_status: str = ProcessingStatus.PENDING.value
    caption_text: str | None = None
    caption_model: str | None = None
    vision_metadata: dict | None = None
    caption_id: UUID | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class DetectedObjectCreate:
    label: str | None = None
    confidence: float | None = None
    x1: float | None = None
    y1: float | None = None
    x2: float | None = None
    y2: float | None = None


@dataclass(frozen=True)
class DetectedObjectRecord(DetectedObjectCreate):
    object_id: UUID | None = None
    object_result_id: UUID | None = None


@dataclass(frozen=True)
class ObjectResultRecord:
    media_id: UUID
    status: str = ProcessingStatus.PENDING.value
    model: str | None = None
    object_result_id: UUID | None = None
    created_at: datetime | None = None
    objects: list[DetectedObjectRecord] = field(default_factory=list)


@dataclass(frozen=True)
class TranscriptSegmentCreate:
    start_time: float | None = None
    end_time: float | None = None
    text: str | None = None


@dataclass(frozen=True)
class TranscriptSegmentRecord(TranscriptSegmentCreate):
    segment_id: UUID | None = None
    transcript_id: UUID | None = None


@dataclass(frozen=True)
class TranscriptRecord:
    video_id: UUID
    status: str = ProcessingStatus.PENDING.value
    language: str | None = None
    model: str | None = None
    full_text: str | None = None
    transcript_id: UUID | None = None
    created_at: datetime | None = None
    segments: list[TranscriptSegmentRecord] = field(default_factory=list)


@dataclass(frozen=True)
class ProcessingJobCreate:
    media_id: UUID
    task_type: str


@dataclass(frozen=True)
class ProcessingJobRecord(ProcessingJobCreate):
    job_id: UUID | None = None
    status: str = ProcessingStatus.PENDING.value
    retry_count: int = 0
    error_message: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


@dataclass(frozen=True)
class EmbeddingRecord:
    media_id: UUID
    vector_db_id: str | None = None
    model: str | None = None
    embedding_id: UUID | None = None
    created_at: datetime | None = None
