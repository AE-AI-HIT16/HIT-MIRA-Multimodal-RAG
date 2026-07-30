"""Schemas for incoming media."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MediaInput(BaseModel):
    """One media item submitted to the media pipeline."""

    media_id: str
    media_type: str
    source_ref: str | None = None
    mime_type: str | None = None
    duration: float | None = None
    size_bytes: int | None = None
    language_hint: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = None
    media_path: str | None = None


class PipelineRequest(BaseModel):
    """Request object accepted by PipelineService."""

    media_input: MediaInput
    correlation_id: str | None = None
    processing_options: dict[str, Any] = Field(default_factory=dict)
    request_context: dict[str, Any] = Field(default_factory=dict)
