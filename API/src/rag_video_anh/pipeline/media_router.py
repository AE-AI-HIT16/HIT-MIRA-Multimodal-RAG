"""Media type routing component."""

from __future__ import annotations

from src.configuration import AppConfig
from src.log.logger import logger
from src.rag_video_anh.schemas import MediaValidationResult, ProcessingRoute


class MediaRouterService:
    """Determine which processing branches should execute."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.pipeline_config = self.config.media_pipeline

    def select_route(self, validation_result: MediaValidationResult) -> ProcessingRoute:
        if not validation_result.is_valid:
            raise ValueError("cannot route invalid media")
        if validation_result.supported_route is None:
            raise ValueError("validation result does not include a supported route")

        profile = validation_result.technical_profile
        route = validation_result.supported_route.copy()
        route.requires_ocr = bool(route.requires_ocr and self.pipeline_config.enable_ocr)
        route.requires_caption = bool(route.requires_caption and self.pipeline_config.enable_caption)
        route.requires_detection = bool(route.requires_detection and self.pipeline_config.enable_detection)
        route.requires_asr = bool(
            route.requires_asr
            and self.pipeline_config.enable_asr
            and (profile is None or profile.audio_present is not False)
        )
        logger.info(f"Selected media route '{route.route_type}'")
        return route
