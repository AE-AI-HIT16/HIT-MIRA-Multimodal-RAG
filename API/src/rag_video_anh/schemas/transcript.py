"""Schemas for media transcripts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.rag_video_anh.schemas.media_metadata import StageStatus


class TranscriptSegment(BaseModel):
    """Time-aligned transcript segment."""

    segment_id: str
    start_ms: int
    end_ms: int
    start_sec: float
    end_sec: float
    text: str
    confidence: float | None = None
    language: str | None = None


class TranscriptSet(BaseModel):
    """Transcript segments for one media item."""

    media_id: str
    segments: list[TranscriptSegment] = Field(default_factory=list)
    language: str | None = None
    transcription_meta: dict = Field(default_factory=dict)
    status: StageStatus = StageStatus.PENDING
    reason: str | None = None


class FrameTranscriptContext(BaseModel):
    """Transcript context aligned to one keyframe."""

    frame_id: str
    matched_segments: list[TranscriptSegment] = Field(default_factory=list)
    context_window_text: str = ""
    context_span: tuple[float, float] | None = None
    status: StageStatus = StageStatus.NOT_FOUND
    reason: str | None = None


class AlignedTranscriptContext(BaseModel):
    """Frame-level transcript context for one media item."""

    media_id: str
    frame_contexts: list[FrameTranscriptContext] = Field(default_factory=list)
    status: StageStatus = StageStatus.PENDING
    reason: str | None = None

class ASRRequest(BaseModel):
    """Input contract for ASR."""

    media_input: object
    validation_result: object | None = None
    asr_policy: dict = Field(default_factory=dict)


class TranscriptMappingRequest(BaseModel):
    """Input contract for transcript-to-frame mapping."""

    keyframes: object
    transcript_set: TranscriptSet
    mapping_policy: dict = Field(default_factory=dict)

