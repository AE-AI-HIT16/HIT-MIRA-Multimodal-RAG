"""Transcript-to-media mapping component."""

from __future__ import annotations

from src.configuration import AppConfig
from src.log.logger import logger
from src.rag_video_anh.schemas import (
    AlignedTranscriptContext,
    FrameTranscriptContext,
    StageStatus,
    TranscriptMappingRequest,
    TranscriptSegment,
)


class TranscriptMapperService:
    """Align transcript context to keyframes using timestamps."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.pipeline_config = self.config.media_pipeline

    def map(self, request: TranscriptMappingRequest) -> AlignedTranscriptContext:
        keyframes = request.keyframes
        transcript_set = request.transcript_set
        if not getattr(keyframes, "frames", None):
            return AlignedTranscriptContext(
                media_id=transcript_set.media_id,
                status=StageStatus.SKIPPED,
                reason="no keyframes available for transcript mapping",
            )
        if not transcript_set.segments:
            return AlignedTranscriptContext(
                media_id=transcript_set.media_id,
                frame_contexts=[
                    FrameTranscriptContext(
                        frame_id=frame.frame_id,
                        status=StageStatus.NOT_FOUND,
                        reason="no transcript segments available",
                    )
                    for frame in keyframes.frames
                ],
                status=StageStatus.NOT_FOUND,
                reason="no transcript segments available",
            )

        window = int(request.mapping_policy.get("context_window", self.pipeline_config.context_window))
        contexts = [self._frame_context(frame, transcript_set.segments, window) for frame in keyframes.frames]
        matched = sum(1 for context in contexts if context.matched_segments)
        status = StageStatus.DONE if matched else StageStatus.NOT_FOUND
        logger.info(f"Mapped transcript context for {matched}/{len(contexts)} frame(s) in media_id '{transcript_set.media_id}'")
        return AlignedTranscriptContext(
            media_id=transcript_set.media_id,
            frame_contexts=contexts,
            status=status,
            reason=None if matched else "no aligned transcript context",
        )

    def _frame_context(self, frame, segments: list[TranscriptSegment], window: int) -> FrameTranscriptContext:
        matched_index = None
        tolerance = self.pipeline_config.timestamp_tolerance or 0.0
        for index, segment in enumerate(segments):
            if segment.start_sec - tolerance <= frame.timestamp_sec <= segment.end_sec + tolerance:
                matched_index = index
                break
        if matched_index is None:
            return FrameTranscriptContext(
                frame_id=frame.frame_id,
                status=StageStatus.NOT_FOUND,
                reason="no segment contains keyframe timestamp",
            )

        start_index = max(0, matched_index - window)
        end_index = min(len(segments), matched_index + window + 1)
        matched_segments = segments[start_index:end_index]
        text = self.pipeline_config.context_text_joiner.join(segment.text for segment in matched_segments if segment.text).strip()
        return FrameTranscriptContext(
            frame_id=frame.frame_id,
            matched_segments=matched_segments,
            context_window_text=text,
            context_span=(matched_segments[0].start_sec, matched_segments[-1].end_sec),
            status=StageStatus.DONE,
        )
