"""Schemas for common media metadata."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StageStatus(str, Enum):
    """Canonical status vocabulary used by each pipeline stage."""

    PENDING = "pending"
    DONE = "done"
    SKIPPED = "skipped"
    ERROR = "error"
    NOT_FOUND = "not_found"


class PipelineStatus(str, Enum):
    """Canonical status vocabulary for the full media pipeline."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"


class MediaMetadata(BaseModel):
    """Technical and descriptive metadata for one media item."""

    media_id: str
    duration: float | None = None
    frame_rate: float | None = None
    width: int | None = None
    height: int | None = None
    audio_present: bool | None = None
    audio_language: str | None = None
    mime_type: str | None = None
    codec_profile: str | None = None


class ProcessingRoute(BaseModel):
    """Processing branches selected for a validated media item."""

    route_type: str
    requires_keyframes: bool = True
    requires_ocr: bool = True
    requires_caption: bool = True
    requires_detection: bool = True
    requires_asr: bool = True


class MediaValidationResult(BaseModel):
    """Validation result for deciding whether processing may continue."""

    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    technical_profile: MediaMetadata | None = None
    supported_route: ProcessingRoute | None = None


class PipelineError(BaseModel):
    """Structured error emitted by a pipeline stage."""

    stage: str
    message: str
    error_code: str | None = None
    media_id: str | None = None
    correlation_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
