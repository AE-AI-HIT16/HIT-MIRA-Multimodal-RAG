"""Automatic speech recognition component."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.configuration import AppConfig
from src.log.logger import logger
from src.rag_video_anh.schemas import ASRRequest, StageStatus, TranscriptSegment, TranscriptSet


class AsrService:
    """Transcribe media audio into time-aligned transcript segments."""

    def __init__(self, config: AppConfig | None = None, model: Any | None = None) -> None:
        self.config = config or AppConfig()
        self.pipeline_config = self.config.media_pipeline
        self.model_config = self.config.media_models
        self.model = model
        self._model_loaded = model is not None
        self._load_error: str | None = None

    def transcribe(self, request: ASRRequest) -> TranscriptSet:
        media_input = request.media_input
        validation_result = request.validation_result
        if validation_result is not None and validation_result.technical_profile is not None:
            if validation_result.technical_profile.audio_present is False:
                return TranscriptSet(
                    media_id=media_input.media_id,
                    status=StageStatus.SKIPPED,
                    reason="media does not contain an audio track",
                )

        media_path = self._resolve_media_path(media_input)
        if media_path is None:
            return TranscriptSet(
                media_id=media_input.media_id,
                status=StageStatus.ERROR,
                reason="media path is required for ASR",
            )

        model = self._load_model()
        if model is None:
            return TranscriptSet(
                media_id=media_input.media_id,
                status=StageStatus.SKIPPED,
                reason=self._load_error or "missing dependency 'faster-whisper'",
            )

        try:
            raw_segments, info = model.transcribe(
                str(media_path),
                beam_size=self.model_config.whisper_beam_size,
                language=request.asr_policy.get("language") or self.model_config.asr_language or media_input.language_hint,
            )
            segments = [self._segment(media_input.media_id, index, segment) for index, segment in enumerate(raw_segments)]
            language = getattr(info, "language", None) or media_input.language_hint
            logger.info(f"ASR produced {len(segments)} segment(s) for media_id '{media_input.media_id}'")
            return TranscriptSet(
                media_id=media_input.media_id,
                segments=segments,
                language=language,
                transcription_meta={
                    "model": self.model_config.whisper_model_size,
                    "beam_size": self.model_config.whisper_beam_size,
                    "compute_type": self.model_config.whisper_compute_type,
                },
                status=StageStatus.DONE if segments else StageStatus.NOT_FOUND,
                reason=None if segments else "empty transcript",
            )
        except Exception as exc:
            return TranscriptSet(
                media_id=media_input.media_id,
                status=StageStatus.ERROR,
                reason=f"ASR failed: {exc.__class__.__name__}",
            )

    @staticmethod
    def _resolve_media_path(media_input: Any) -> Path | None:
        value = getattr(media_input, "media_path", None) or getattr(media_input, "source_ref", None)
        if not value:
            value = getattr(media_input, "metadata", {}).get("path")
        if not value:
            return None
        path = Path(str(value))
        return path if path.exists() else None

    def _load_model(self) -> Any | None:
        if self._model_loaded:
            return self.model
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            self._model_loaded = True
            self._load_error = "missing dependency 'faster-whisper'"
            return None
        try:
            self.model = WhisperModel(
                self.model_config.whisper_model_size,
                device=self.model_config.whisper_device,
                compute_type=self.model_config.whisper_compute_type,
            )
            self._model_loaded = True
            return self.model
        except Exception as exc:
            self._model_loaded = True
            self._load_error = f"ASR model load failed: {exc.__class__.__name__}"
            return None

    @staticmethod
    def _segment(media_id: str, index: int, raw_segment: Any) -> TranscriptSegment:
        start_sec = float(getattr(raw_segment, "start", 0.0) or 0.0)
        end_sec = float(getattr(raw_segment, "end", start_sec) or start_sec)
        text = str(getattr(raw_segment, "text", "")).strip()
        confidence = getattr(raw_segment, "confidence", None)
        return TranscriptSegment(
            segment_id=f"{media_id}_s{index:06d}",
            start_ms=int(start_sec * 1000),
            end_ms=int(end_sec * 1000),
            start_sec=start_sec,
            end_sec=end_sec,
            text=text,
            confidence=float(confidence) if confidence is not None else None,
            language=getattr(raw_segment, "language", None),
        )
