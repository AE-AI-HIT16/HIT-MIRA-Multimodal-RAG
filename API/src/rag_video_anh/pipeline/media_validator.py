"""Media validation component."""

from __future__ import annotations

from pathlib import Path

from src.configuration import AppConfig
from src.log.logger import logger
from src.rag_video_anh.schemas import MediaInput, MediaMetadata, MediaValidationResult, ProcessingRoute


class MediaValidatorService:
    """Validate that a media item is usable by the pipeline."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.pipeline_config = self.config.media_pipeline

    def validate(self, media_input: MediaInput) -> MediaValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        media_type = (media_input.media_type or "").strip().lower()
        mime_type = (media_input.mime_type or "").strip().lower()
        source_ref = media_input.media_path or media_input.source_ref or ""
        extension = Path(source_ref).suffix.lower()
        supported_types = {item.lower() for item in self.pipeline_config.supported_media_types}

        if not media_input.media_id.strip():
            errors.append("media_id is required")
        if not media_type:
            errors.append("media_type is required")
        if media_type and media_type != "video":
            errors.append(f"unsupported media_type: {media_input.media_type}")
        if extension and extension not in supported_types:
            errors.append(f"unsupported media extension: {extension}")
        if not extension and mime_type and not mime_type.startswith("video/"):
            errors.append(f"unsupported mime_type: {media_input.mime_type}")
        if self.pipeline_config.minimum_duration is not None and media_input.duration is not None:
            if media_input.duration < self.pipeline_config.minimum_duration:
                errors.append("media duration is below the configured minimum")
        if self.pipeline_config.maximum_duration is not None and media_input.duration is not None:
            if media_input.duration > self.pipeline_config.maximum_duration:
                errors.append("media duration exceeds the configured maximum")
        if self.pipeline_config.minimum_size_bytes is not None and media_input.size_bytes is not None:
            if media_input.size_bytes < self.pipeline_config.minimum_size_bytes:
                errors.append("media size is below the configured minimum")
        if self.pipeline_config.maximum_size_bytes is not None and media_input.size_bytes is not None:
            if media_input.size_bytes > self.pipeline_config.maximum_size_bytes:
                errors.append("media size exceeds the configured maximum")
        if self.pipeline_config.audio_required and media_input.metadata.get("audio_present") is False:
            errors.append("audio track is required by policy")

        technical_profile = MediaMetadata(
            media_id=media_input.media_id,
            duration=media_input.duration,
            frame_rate=media_input.metadata.get("frame_rate"),
            width=media_input.metadata.get("width"),
            height=media_input.metadata.get("height"),
            audio_present=media_input.metadata.get("audio_present"),
            audio_language=media_input.metadata.get("audio_language") or media_input.language_hint,
            mime_type=media_input.mime_type,
            codec_profile=media_input.metadata.get("codec_profile"),
        )
        route = None
        if not errors:
            route = ProcessingRoute(
                route_type="video",
                requires_keyframes=True,
                requires_ocr=self.pipeline_config.enable_ocr,
                requires_caption=self.pipeline_config.enable_caption,
                requires_detection=self.pipeline_config.enable_detection,
                requires_asr=self.pipeline_config.enable_asr and technical_profile.audio_present is not False,
            )

        result = MediaValidationResult(
            is_valid=not errors,
            errors=errors,
            warnings=warnings,
            technical_profile=technical_profile,
            supported_route=route,
        )
        logger.info(f"Media validation completed for media_id '{media_input.media_id}' with valid={result.is_valid}")
        return result
