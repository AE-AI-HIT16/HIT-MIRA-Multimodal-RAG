"""Data schemas for media ingestion and retrieval."""

from src.rag_video_anh.schemas.media_input import MediaInput, PipelineRequest
from src.rag_video_anh.schemas.media_metadata import (
    MediaMetadata,
    MediaValidationResult,
    PipelineError,
    PipelineStatus,
    ProcessingRoute,
    StageStatus,
)
from src.rag_video_anh.schemas.normalized_media import (
    NormalizationRequest,
    NormalizedMediaMetadata,
    PipelineResult,
)
from src.rag_video_anh.schemas.transcript import (
    AlignedTranscriptContext,
    ASRRequest,
    FrameTranscriptContext,
    TranscriptMappingRequest,
    TranscriptSegment,
    TranscriptSet,
)
from src.rag_video_anh.schemas.video_metadata import (
    CaptionResult,
    CaptionResultSet,
    DetectionBox,
    DetectionRequest,
    DetectionResult,
    DetectionResultSet,
    KeyFrame,
    KeyframeExtractionRequest,
    KeyFrameSet,
    OCRResult,
    OCRResultSet,
    TextSpan,
)

__all__ = [
    "MediaInput",
    "PipelineRequest",
    "MediaMetadata",
    "MediaValidationResult",
    "PipelineError",
    "PipelineStatus",
    "ProcessingRoute",
    "StageStatus",
    "NormalizationRequest",
    "NormalizedMediaMetadata",
    "PipelineResult",
    "ASRRequest",
    "AlignedTranscriptContext",
    "FrameTranscriptContext",
    "TranscriptMappingRequest",
    "TranscriptSegment",
    "TranscriptSet",
    "CaptionResult",
    "CaptionResultSet",
    "DetectionBox",
    "DetectionRequest",
    "DetectionResult",
    "DetectionResultSet",
    "KeyFrame",
    "KeyFrameSet",
    "KeyframeExtractionRequest",
    "OCRResult",
    "OCRResultSet",
    "TextSpan",
]
