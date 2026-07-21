"""Schemas for video metadata."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.rag_video_anh.schemas.media_metadata import StageStatus


class KeyFrame(BaseModel):
    """Representative frame selected from a media timeline."""

    frame_id: str
    media_id: str
    frame_index: int
    timestamp_ms: int
    timestamp_sec: float
    quality_score: float | None = None
    dedup_score: float | None = None
    selection_reason: str | None = None
    image_payload: Any | None = None
    image_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class KeyFrameSet(BaseModel):
    """Selected keyframes for one media item."""

    media_id: str
    frames: list[KeyFrame] = Field(default_factory=list)
    status: StageStatus = StageStatus.PENDING
    reason: str | None = None
    selection_summary: dict[str, Any] = Field(default_factory=dict)
    coverage_summary: dict[str, Any] = Field(default_factory=dict)


class TextSpan(BaseModel):
    """Recognized text span from a keyframe."""

    text: str
    confidence: float | None = None
    bbox: list[list[float]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OCRResult(BaseModel):
    """Text recognized from one keyframe."""

    frame_id: str
    text_spans: list[TextSpan] = Field(default_factory=list)
    full_text: str = ""
    confidence: float | None = None
    language_hints: list[str] = Field(default_factory=list)
    status: StageStatus = StageStatus.DONE
    reason: str | None = None


class OCRResultSet(BaseModel):
    """OCR results for all keyframes of one media item."""

    media_id: str
    results: list[OCRResult] = Field(default_factory=list)
    status: StageStatus = StageStatus.PENDING
    reason: str | None = None


class CaptionResult(BaseModel):
    """Generated caption for one keyframe."""

    frame_id: str
    caption_text: str = ""
    generation_meta: dict[str, Any] = Field(default_factory=dict)
    prompt_version: str | None = None
    status: StageStatus = StageStatus.DONE
    reason: str | None = None


class CaptionResultSet(BaseModel):
    """Caption results for all keyframes of one media item."""

    media_id: str
    results: list[CaptionResult] = Field(default_factory=list)
    status: StageStatus = StageStatus.PENDING
    reason: str | None = None


class DetectionBox(BaseModel):
    """Detected object geometry and confidence."""

    label: str
    confidence: float | None = None
    bbox: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DetectionResult(BaseModel):
    """Object detection output for one keyframe."""

    frame_id: str
    detections: list[DetectionBox] = Field(default_factory=list)
    inference_meta: dict[str, Any] = Field(default_factory=dict)
    status: StageStatus = StageStatus.DONE
    reason: str | None = None


class DetectionResultSet(BaseModel):
    """Detection outputs for all keyframes of one media item."""

    media_id: str
    results: list[DetectionResult] = Field(default_factory=list)
    status: StageStatus = StageStatus.PENDING
    reason: str | None = None

class KeyframeExtractionRequest(BaseModel):
    """Input contract for keyframe extraction."""

    media_input: Any
    validation_result: Any | None = None
    route: Any | None = None
    extraction_policy: dict[str, Any] = Field(default_factory=dict)


class OCRRequest(BaseModel):
    """Input contract for OCR."""

    media_id: str
    keyframes: KeyFrameSet
    ocr_policy: dict[str, Any] = Field(default_factory=dict)


class CaptionRequest(BaseModel):
    """Input contract for caption generation."""

    media_id: str
    keyframes: KeyFrameSet
    caption_policy: dict[str, Any] = Field(default_factory=dict)


class DetectionRequest(BaseModel):
    """Input contract for object detection."""

    media_id: str
    keyframes: KeyFrameSet
    detection_policy: dict[str, Any] = Field(default_factory=dict)

