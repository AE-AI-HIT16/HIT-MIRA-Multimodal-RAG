"""Repository layer for PostgreSQL metadata persistence.

AI services should depend on these repositories instead of issuing SQL.
Binary media stays in MinIO; this package stores object references, metadata,
processing status, and AI result records.
"""

from src.rag_video_anh.repository.database import DatabaseSessionManager, get_database_url
from src.rag_video_anh.repository.datasets import DatasetRepository
from src.rag_video_anh.repository.events import EventRepository, normalize_alias, slugify
from src.rag_video_anh.repository.jobs import ProcessingJobRepository
from src.rag_video_anh.repository.media import MediaRepository
from src.rag_video_anh.repository.posts import PostRepository
from src.rag_video_anh.repository.results import AIResultRepository
from src.rag_video_anh.repository.schemas import (
    CaptionResultRecord,
    DatasetCreate,
    DatasetRecord,
    DetectedObjectCreate,
    DetectedObjectRecord,
    EmbeddingRecord,
    EventOccurrenceRecord,
    EventSeriesRecord,
    FrameCreate,
    FrameRecord,
    MediaCreate,
    MediaRecord,
    MediaType,
    ObjectResultRecord,
    OcrResultRecord,
    PostCreate,
    PostEventLinkRecord,
    PostRecord,
    ProcessingJobCreate,
    ProcessingJobRecord,
    ProcessingStatus,
    TaskType,
    TranscriptRecord,
    TranscriptSegmentCreate,
    TranscriptSegmentRecord,
    VideoMetadataCreate,
    VideoMetadataRecord,
)
from src.rag_video_anh.repository.unit_of_work import RepositoryUnitOfWork

__all__ = [
    "AIResultRepository",
    "CaptionResultRecord",
    "DatabaseSessionManager",
    "DatasetCreate",
    "DatasetRecord",
    "DatasetRepository",
    "DetectedObjectCreate",
    "DetectedObjectRecord",
    "EmbeddingRecord",
    "EventOccurrenceRecord",
    "EventRepository",
    "EventSeriesRecord",
    "FrameCreate",
    "FrameRecord",
    "MediaCreate",
    "MediaRecord",
    "MediaRepository",
    "MediaType",
    "ObjectResultRecord",
    "OcrResultRecord",
    "PostCreate",
    "PostEventLinkRecord",
    "PostRecord",
    "PostRepository",
    "ProcessingJobCreate",
    "ProcessingJobRecord",
    "ProcessingJobRepository",
    "ProcessingStatus",
    "RepositoryUnitOfWork",
    "TaskType",
    "TranscriptRecord",
    "TranscriptSegmentCreate",
    "TranscriptSegmentRecord",
    "VideoMetadataCreate",
    "VideoMetadataRecord",
    "get_database_url",
    "normalize_alias",
    "slugify",
]
