"""Media pipeline orchestration service."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
from typing import Any, Callable

from src.configuration import AppConfig
from src.log.logger import logger
from src.rag_video_anh.pipeline.asr_service import AsrService
from src.rag_video_anh.pipeline.detection_service import DetectionService
from src.rag_video_anh.pipeline.keyframe_extractor import KeyframeExtractorService
from src.rag_video_anh.pipeline.media_router import MediaRouterService
from src.rag_video_anh.pipeline.media_validator import MediaValidatorService
from src.rag_video_anh.pipeline.normalizer import NormalizerService
from src.rag_video_anh.pipeline.qwen_vision_service import QwenVisionService
from src.rag_video_anh.pipeline.transcript_mapper import TranscriptMapperService
from src.rag_video_anh.schemas import (
    ASRRequest,
    AlignedTranscriptContext,
    CaptionResultSet,
    DetectionRequest,
    DetectionResultSet,
    KeyFrameSet,
    KeyframeExtractionRequest,
    MediaInput,
    NormalizationRequest,
    OCRResultSet,
    PipelineError,
    PipelineRequest,
    PipelineResult,
    PipelineStatus,
    StageStatus,
    TranscriptMappingRequest,
    TranscriptSet,
)


class PipelineService:
    """Coordinate the full media pipeline and return a canonical result."""

    def __init__(
        self,
        config: AppConfig | None = None,
        validator: MediaValidatorService | None = None,
        router: MediaRouterService | None = None,
        keyframe_extractor: KeyframeExtractorService | None = None,
        vision_service: QwenVisionService | None = None,
        detection_service: DetectionService | None = None,
        asr_service: AsrService | None = None,
        transcript_mapper: TranscriptMapperService | None = None,
        normalizer: NormalizerService | None = None,
    ) -> None:
        self.config = config or AppConfig()
        self.pipeline_config = self.config.media_pipeline
        self.validator = validator or MediaValidatorService(config=self.config)
        self.router = router or MediaRouterService(config=self.config)
        self.keyframe_extractor = keyframe_extractor or KeyframeExtractorService(config=self.config)
        self.vision_service = vision_service or QwenVisionService(config=self.config)
        self.detection_service = detection_service or DetectionService(config=self.config)
        self.asr_service = asr_service or AsrService(config=self.config)
        self.transcript_mapper = transcript_mapper or TranscriptMapperService(config=self.config)
        self.normalizer = normalizer or NormalizerService(config=self.config)

    def process(self, request: PipelineRequest | MediaInput) -> PipelineResult:
        pipeline_request = self._coerce_request(request)
        media_input = pipeline_request.media_input
        correlation_id = pipeline_request.correlation_id or media_input.correlation_id
        errors: list[PipelineError] = []
        started = perf_counter()
        logger.info(f"Starting media pipeline for media_id '{media_input.media_id}'")

        validation_result = self.validator.validate(media_input)
        if not validation_result.is_valid:
            errors.extend(
                PipelineError(
                    stage="validation",
                    message=message,
                    media_id=media_input.media_id,
                    correlation_id=correlation_id,
                )
                for message in validation_result.errors
            )
            return PipelineResult(
                media_input=media_input,
                validation_result=validation_result,
                status=PipelineStatus.FAILED,
                errors=errors,
            )

        try:
            route = self.router.select_route(validation_result)
        except Exception as exc:
            errors.append(self._error("routing", exc, media_input, correlation_id))
            return PipelineResult(
                media_input=media_input,
                validation_result=validation_result,
                status=PipelineStatus.FAILED,
                errors=errors,
            )

        keyframes = self._safe_stage(
            "keyframe_extraction",
            lambda: self.keyframe_extractor.extract(
                KeyframeExtractionRequest(
                    media_input=media_input,
                    validation_result=validation_result,
                    route=route,
                    extraction_policy=pipeline_request.processing_options.get("keyframes", {}),
                )
            ),
            errors,
            media_input,
            correlation_id,
        )
        if keyframes is None:
            keyframes = KeyFrameSet(media_id=media_input.media_id, status=StageStatus.ERROR, reason="keyframe extraction failed")

        ocr_results, caption_results, detection_results, transcript_set = self._run_enrichment(
            pipeline_request,
            route,
            keyframes,
            validation_result,
            errors,
            correlation_id,
        )

        aligned_context = self._safe_stage(
            "transcript_mapping",
            lambda: self.transcript_mapper.map(
                TranscriptMappingRequest(
                    keyframes=keyframes,
                    transcript_set=transcript_set,
                    mapping_policy=pipeline_request.processing_options.get("transcript_mapping", {}),
                )
            ),
            errors,
            media_input,
            correlation_id,
        )
        if aligned_context is None:
            aligned_context = AlignedTranscriptContext(
                media_id=media_input.media_id,
                status=StageStatus.ERROR,
                reason="transcript mapping failed",
            )

        normalized_metadata = self._safe_stage(
            "normalization",
            lambda: self.normalizer.normalize(
                NormalizationRequest(
                    media_input=media_input,
                    validation_result=validation_result,
                    keyframes=keyframes,
                    ocr_results=ocr_results,
                    caption_results=caption_results,
                    detection_results=detection_results,
                    transcript_set=transcript_set,
                    aligned_context=aligned_context,
                )
            ),
            errors,
            media_input,
            correlation_id,
        )

        status = self._pipeline_status(
            errors,
            keyframes,
            ocr_results,
            caption_results,
            detection_results,
            transcript_set,
            aligned_context,
            normalized_metadata,
        )
        logger.info(
            f"Completed media pipeline for media_id '{media_input.media_id}' with status '{status.value}' in {perf_counter() - started:.2f}s"
        )
        return PipelineResult(
            media_input=media_input,
            validation_result=validation_result,
            route=route,
            keyframes=keyframes,
            ocr_results=ocr_results,
            caption_results=caption_results,
            detection_results=detection_results,
            transcript_set=transcript_set,
            aligned_context=aligned_context,
            normalized_metadata=normalized_metadata,
            status=status,
            errors=errors,
        )

    def _run_enrichment(
        self,
        request: PipelineRequest,
        route,
        keyframes: KeyFrameSet,
        validation_result,
        errors: list[PipelineError],
        correlation_id: str | None,
    ) -> tuple[OCRResultSet, CaptionResultSet, DetectionResultSet, TranscriptSet]:
        media_input = request.media_input
        tasks: dict[str, Callable[[], Any]] = {}
        if route.requires_ocr or route.requires_caption:
            tasks["vision"] = lambda: self.vision_service.analyze(
                media_id=media_input.media_id,
                keyframes=keyframes,
                ocr_policy=request.processing_options.get("ocr", {}),
                caption_policy=request.processing_options.get("caption", {}),
            )
        if route.requires_detection:
            tasks["detection"] = lambda: self.detection_service.detect(
                DetectionRequest(media_id=media_input.media_id, keyframes=keyframes, detection_policy=request.processing_options.get("detection", {}))
            )
        if route.requires_asr:
            tasks["asr"] = lambda: self.asr_service.transcribe(
                ASRRequest(media_input=media_input, validation_result=validation_result, asr_policy=request.processing_options.get("asr", {}))
            )

        results: dict[str, Any] = {}
        if self.pipeline_config.parallel_enrichment and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
                future_to_stage = {executor.submit(task): stage for stage, task in tasks.items()}
                for future in as_completed(future_to_stage):
                    stage = future_to_stage[future]
                    results[stage] = self._resolve_future(stage, future, errors, media_input, correlation_id)
        else:
            for stage, task in tasks.items():
                results[stage] = self._safe_stage(stage, task, errors, media_input, correlation_id)

        vision_results = results.get("vision")
        ocr_results = vision_results[0] if vision_results and route.requires_ocr else None
        caption_results = vision_results[1] if vision_results and route.requires_caption else None
        return (
            ocr_results or OCRResultSet(media_id=media_input.media_id, status=StageStatus.SKIPPED, reason="OCR disabled by route"),
            caption_results or CaptionResultSet(media_id=media_input.media_id, status=StageStatus.SKIPPED, reason="caption disabled by route"),
            results.get("detection") or DetectionResultSet(media_id=media_input.media_id, status=StageStatus.SKIPPED, reason="detection disabled by route"),
            results.get("asr") or TranscriptSet(media_id=media_input.media_id, status=StageStatus.SKIPPED, reason="ASR disabled by route"),
        )

    def _safe_stage(
        self,
        stage: str,
        callback: Callable[[], Any],
        errors: list[PipelineError],
        media_input: MediaInput,
        correlation_id: str | None,
    ) -> Any | None:
        try:
            stage_started = perf_counter()
            result = callback()
            logger.info(f"Stage '{stage}' completed for media_id '{media_input.media_id}' in {perf_counter() - stage_started:.2f}s")
            return result
        except Exception as exc:
            errors.append(self._error(stage, exc, media_input, correlation_id))
            logger.error(f"Stage '{stage}' failed for media_id '{media_input.media_id}': {exc}")
            return None

    def _resolve_future(self, stage: str, future, errors: list[PipelineError], media_input: MediaInput, correlation_id: str | None) -> Any | None:
        try:
            return future.result(timeout=self.pipeline_config.stage_timeout)
        except Exception as exc:
            errors.append(self._error(stage, exc, media_input, correlation_id))
            logger.error(f"Stage '{stage}' failed for media_id '{media_input.media_id}': {exc}")
            return None

    @staticmethod
    def _coerce_request(request: PipelineRequest | MediaInput) -> PipelineRequest:
        if isinstance(request, PipelineRequest):
            return request
        if isinstance(request, MediaInput):
            return PipelineRequest(media_input=request, correlation_id=request.correlation_id)
        raise TypeError("request must be PipelineRequest or MediaInput")

    @staticmethod
    def _error(stage: str, exc: Exception, media_input: MediaInput, correlation_id: str | None) -> PipelineError:
        return PipelineError(
            stage=stage,
            message=str(exc),
            error_code=exc.__class__.__name__,
            media_id=media_input.media_id,
            correlation_id=correlation_id,
        )

    def _pipeline_status(self, errors: list[PipelineError], *stage_results: Any) -> PipelineStatus:
        if any(result is None for result in stage_results):
            return PipelineStatus.PARTIAL_SUCCESS if self.pipeline_config.partial_success_policy else PipelineStatus.FAILED
        if errors:
            return PipelineStatus.PARTIAL_SUCCESS if self.pipeline_config.partial_success_policy else PipelineStatus.FAILED

        incomplete_stage = False
        for result in stage_results:
            status = getattr(result, "status", None)
            reason = str(getattr(result, "reason", "") or "")
            if status == StageStatus.ERROR:
                incomplete_stage = True
            elif status in {StageStatus.SKIPPED, StageStatus.NOT_FOUND} and "disabled by route" not in reason:
                incomplete_stage = True
        if incomplete_stage:
            return PipelineStatus.PARTIAL_SUCCESS if self.pipeline_config.partial_success_policy else PipelineStatus.FAILED
        return PipelineStatus.SUCCESS
