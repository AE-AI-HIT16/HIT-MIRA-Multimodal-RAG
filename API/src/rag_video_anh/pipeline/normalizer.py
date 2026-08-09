"""Media normalization component."""

from __future__ import annotations

from src.configuration import AppConfig
from src.log.logger import logger
from src.rag_video_anh.schemas import NormalizationRequest, NormalizedMediaMetadata, StageStatus


class NormalizerService:
    """Merge multimodal outputs into the canonical metadata schema."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.pipeline_config = self.config.media_pipeline

    def normalize(self, request: NormalizationRequest) -> NormalizedMediaMetadata:
        media_input = request.media_input
        frame_records = []
        ocr_by_frame = {result.frame_id: result for result in (request.ocr_results.results if request.ocr_results else [])}
        captions_by_frame = {result.frame_id: result for result in (request.caption_results.results if request.caption_results else [])}
        detections_by_frame = {result.frame_id: result for result in (request.detection_results.results if request.detection_results else [])}
        contexts_by_frame = {context.frame_id: context for context in (request.aligned_context.frame_contexts if request.aligned_context else [])}
        keyframes = request.keyframes.frames if request.keyframes else []

        quality_flags: list[str] = []
        for stage_name, stage_result in [
            ("keyframes", request.keyframes),
            ("ocr", request.ocr_results),
            ("caption", request.caption_results),
            ("detection", request.detection_results),
            ("transcript", request.transcript_set),
            ("alignment", request.aligned_context),
        ]:
            if stage_result is None:
                quality_flags.append(f"missing_{stage_name}")
            elif stage_result.status in {StageStatus.ERROR, StageStatus.SKIPPED, StageStatus.NOT_FOUND}:
                quality_flags.append(f"{stage_name}_{stage_result.status.value}")

        for frame in keyframes:
            ocr = ocr_by_frame.get(frame.frame_id)
            caption = captions_by_frame.get(frame.frame_id)
            detection = detections_by_frame.get(frame.frame_id)
            context = contexts_by_frame.get(frame.frame_id)
            frame_records.append(
                {
                    "frame_id": frame.frame_id,
                    "frame_index": frame.frame_index,
                    "timestamp_ms": frame.timestamp_ms,
                    "timestamp_sec": frame.timestamp_sec,
                    "quality_score": frame.quality_score,
                    "dedup_score": frame.dedup_score,
                    "selection_reason": frame.selection_reason,
                    "ocr_text": ocr.full_text if ocr else "",
                    "ocr_confidence": ocr.confidence if ocr else None,
                    "caption": caption.caption_text if caption else "",
                    "detections": [detection_box.dict() for detection_box in detection.detections] if detection else [],
                    "transcript_context": context.context_window_text if context else "",
                    "matched_segment_ids": [segment.segment_id for segment in context.matched_segments] if context else [],
                }
            )

        transcript_segments = request.transcript_set.segments if request.transcript_set else []
        captions = [result.caption_text for result in captions_by_frame.values() if result.caption_text]
        ocr_texts = [result.full_text for result in ocr_by_frame.values() if result.full_text]
        detections = [box for result in detections_by_frame.values() for box in result.detections]
        metadata = NormalizedMediaMetadata(
            media_id=media_input.media_id,
            media_summary={
                "media_type": media_input.media_type,
                "mime_type": media_input.mime_type,
                "duration": media_input.duration,
                "source_ref": media_input.source_ref,
                "metadata": media_input.metadata,
            },
            visual_summary={
                "frame_count": len(frame_records),
                "caption_count": len(captions),
                "detection_count": len(detections),
            },
            text_summary={
                "ocr_text": " ".join(ocr_texts).strip(),
                "caption_text": " ".join(captions).strip(),
            },
            audio_summary={
                "language": request.transcript_set.language if request.transcript_set else media_input.language_hint,
                "segment_count": len(transcript_segments),
                "transcript_text": " ".join(segment.text for segment in transcript_segments if segment.text).strip(),
            },
            frames=frame_records,
            global_entities=[
                {"label": detection.label, "confidence": detection.confidence}
                for detection in detections
            ],
            quality_flags=quality_flags,
            schema_version=self.pipeline_config.canonical_schema_version,
        )
        logger.info(f"Normalized media metadata for media_id '{media_input.media_id}' with schema '{metadata.schema_version}'")
        return metadata
