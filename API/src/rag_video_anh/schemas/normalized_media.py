"""Schemas for normalized media."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.rag_video_anh.schemas.media_input import MediaInput
from src.rag_video_anh.schemas.media_metadata import (
    MediaValidationResult,
    PipelineError,
    PipelineStatus,
    ProcessingRoute,
)
from src.rag_video_anh.schemas.transcript import AlignedTranscriptContext, TranscriptSet
from src.rag_video_anh.schemas.video_metadata import (
    CaptionResultSet,
    DetectionResultSet,
    KeyFrameSet,
    OCRResultSet,
)


class NormalizedMediaMetadata(BaseModel):
    """Canonical merged metadata object for downstream persistence/indexing."""

    media_id: str
    media_summary: dict[str, Any] = Field(default_factory=dict)
    visual_summary: dict[str, Any] = Field(default_factory=dict)
    text_summary: dict[str, Any] = Field(default_factory=dict)
    audio_summary: dict[str, Any] = Field(default_factory=dict)
    frames: list[dict[str, Any]] = Field(default_factory=list)
    global_entities: list[dict[str, Any]] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    schema_version: str = "v1"


class NormalizationRequest(BaseModel):
    """Input contract for metadata normalization."""

    media_input: MediaInput
    validation_result: MediaValidationResult | None = None
    keyframes: KeyFrameSet | None = None
    ocr_results: OCRResultSet | None = None
    caption_results: CaptionResultSet | None = None
    detection_results: DetectionResultSet | None = None
    transcript_set: TranscriptSet | None = None
    aligned_context: AlignedTranscriptContext | None = None


class PipelineResult(BaseModel):
    """Final result returned by PipelineService."""

    media_input: MediaInput
    validation_result: MediaValidationResult | None = None
    route: ProcessingRoute | None = None
    keyframes: KeyFrameSet | None = None
    ocr_results: OCRResultSet | None = None
    caption_results: CaptionResultSet | None = None
    detection_results: DetectionResultSet | None = None
    transcript_set: TranscriptSet | None = None
    aligned_context: AlignedTranscriptContext | None = None
    normalized_metadata: NormalizedMediaMetadata | None = None
    status: PipelineStatus = PipelineStatus.FAILED
    errors: list[PipelineError] = Field(default_factory=list)
