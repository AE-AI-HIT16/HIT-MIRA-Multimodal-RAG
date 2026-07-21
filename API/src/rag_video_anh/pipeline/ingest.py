"""Media ingestion component."""

from __future__ import annotations

from src.configuration import AppConfig
from src.rag_video_anh.pipeline.pipeline_service import PipelineService
from src.rag_video_anh.schemas import MediaInput, PipelineRequest, PipelineResult


class MediaIngestService:
    """Submit an already-resolved media object to the media pipeline."""

    def __init__(self, pipeline_service: PipelineService | None = None, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.pipeline_service = pipeline_service or PipelineService(config=self.config)

    def ingest_media(self, media_input: MediaInput, correlation_id: str | None = None) -> PipelineResult:
        request = PipelineRequest(
            media_input=media_input,
            correlation_id=correlation_id or media_input.correlation_id,
        )
        return self.pipeline_service.process(request)
