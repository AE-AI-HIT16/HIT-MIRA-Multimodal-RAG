"""Worker that connects PostgreSQL media jobs, MinIO objects, and the media pipeline."""

from __future__ import annotations

import argparse
import mimetypes
import tempfile
import uuid
from pathlib import Path
from typing import Any

from src.configuration import AppConfig
from src.log.logger import logger
from src.rag_video_anh.pipeline.minio_storage import MinioStorage
from src.rag_video_anh.pipeline.pipeline_service import PipelineService
from src.rag_video_anh.repository import (
    DetectedObjectCreate,
    FrameCreate,
    MediaRecord,
    OcrBoxCreate,
    RepositoryUnitOfWork,
    TaskType,
    TranscriptSegmentCreate,
    VideoMetadataCreate,
)
from src.rag_video_anh.schemas import MediaInput, PipelineResult, PipelineStatus


class VideoProcessingWorker:
    """Process queued video media rows end-to-end.

    PostgreSQL stores only object references. This worker downloads the video
    from MinIO, runs the local pipeline, uploads extracted keyframes back to
    MinIO, then persists AI metadata in PostgreSQL.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        storage: MinioStorage | None = None,
        pipeline: PipelineService | None = None,
        uow_factory: type[RepositoryUnitOfWork] = RepositoryUnitOfWork,
    ) -> None:
        self.config = config or AppConfig()
        self.storage = storage or MinioStorage(config=self.config)
        self.pipeline = pipeline or PipelineService(config=self.config)
        self.uow_factory = uow_factory

    def process_next(self, task_type: str = TaskType.KEYFRAME.value) -> PipelineResult | None:
        """Claim one pending job and process it.

        Returns None when there is no pending job of the requested task type.
        """

        with self.uow_factory() as uow:
            assert uow.jobs is not None
            job = uow.jobs.claim_next(task_type)

        if job is None:
            logger.info(f"No pending '{task_type}' processing job found")
            return None

        try:
            result = self.process_media(job.media_id)
            with self.uow_factory() as uow:
                assert uow.jobs is not None
                if result.status == PipelineStatus.FAILED:
                    uow.jobs.mark_failed(job.job_id, self._pipeline_error_message(result))
                else:
                    uow.jobs.mark_done(job.job_id)
            return result
        except Exception as exc:
            with self.uow_factory() as uow:
                assert uow.jobs is not None
                uow.jobs.mark_failed(job.job_id, str(exc), retry=True, max_retries=self.config.media_pipeline.retry_count)
            raise

    def process_media(self, media_id: str | uuid.UUID) -> PipelineResult:
        """Process one video media row by id without claiming a job."""

        with self.uow_factory() as uow:
            assert uow.media is not None
            media = uow.media.get_media(media_id)
            if media is None:
                raise ValueError(f"media does not exist: {media_id}")
            if media.media_type != "video":
                raise ValueError(f"media_id {media_id} is not a video row")

        suffix = Path(media.object_key).suffix or ".mp4"
        with tempfile.TemporaryDirectory(prefix="hit-mira-video-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            local_video = tmp_path / f"{media.media_id}{suffix}"
            self._download_object(media, local_video)

            result = self.pipeline.process(
                MediaInput(
                    media_id=str(media.media_id),
                    media_type=media.media_type,
                    source_ref=media.object_key,
                    media_path=str(local_video),
                    metadata={
                        "bucket_name": media.bucket_name,
                        "object_key": media.object_key,
                    },
                )
            )
            self._persist_result(media, result, tmp_path)
            return result

    def _persist_result(self, media: MediaRecord, result: PipelineResult, tmp_path: Path) -> None:
        with self.uow_factory() as uow:
            assert uow.media is not None
            assert uow.results is not None

            uow.media.upsert_video_metadata(
                VideoMetadataCreate(
                    media_id=media.media_id,
                    duration=self._duration(result),
                    fps=self._fps(result),
                    raw_frames=self._raw_frames(result),
                    selected_keyframes=len(result.keyframes.frames) if result.keyframes else 0,
                )
            )

            frame_media_ids = self._persist_keyframes(uow, media, result, tmp_path)
            self._persist_ocr(uow, result, frame_media_ids)
            self._persist_captions(uow, result, frame_media_ids)
            self._persist_detections(uow, result, frame_media_ids)
            self._persist_transcript(uow, media, result)

    def _persist_keyframes(
        self,
        uow: RepositoryUnitOfWork,
        media: MediaRecord,
        result: PipelineResult,
        tmp_path: Path,
    ) -> dict[str, uuid.UUID]:
        frame_media_ids: dict[str, uuid.UUID] = {}
        if result.keyframes is None:
            return frame_media_ids

        assert uow.media is not None
        existing_frames = uow.media.list_frames_for_video(media.media_id)
        existing_by_index = {frame.frame_index: frame for frame in existing_frames if frame.frame_index is not None}
        existing_by_key = {frame.object_key: frame for frame in existing_frames}

        for frame in result.keyframes.frames:
            object_key = self._frame_object_key(media, frame)
            existing_frame = existing_by_index.get(frame.frame_index) or existing_by_key.get(object_key)
            if existing_frame is not None and existing_frame.media_id is not None:
                frame_media_ids[frame.frame_id] = existing_frame.media_id
                logger.info(f"Reusing existing keyframe media_id {existing_frame.media_id} for frame_index {frame.frame_index}")
                continue

            local_frame = self._write_keyframe_image(frame, tmp_path)
            self._upload_object(media.bucket_name, local_frame, object_key, content_type="image/jpeg")
            frame_record = uow.media.register_frame(
                FrameCreate(
                    video_media_id=media.media_id,
                    bucket_name=media.bucket_name,
                    object_key=object_key,
                    timestamp=frame.timestamp_sec,
                    frame_index=frame.frame_index,
                )
            )
            frame_media_ids[frame.frame_id] = frame_record.media_id
            existing_by_index[frame.frame_index] = frame_record
            existing_by_key[object_key] = frame_record
        return frame_media_ids

    def _persist_ocr(self, uow: RepositoryUnitOfWork, result: PipelineResult, frame_media_ids: dict[str, uuid.UUID]) -> None:
        if result.ocr_results is None:
            return
        assert uow.results is not None
        if not result.ocr_results.results:
            for frame_media_id in frame_media_ids.values():
                uow.results.upsert_ocr_result(
                    frame_media_id,
                    status=self._persistable_status(result.ocr_results.status.value, result.ocr_results.reason),
                    text=result.ocr_results.reason,
                    model=getattr(self.pipeline.ocr_service, "backend", "paddleocr"),
                    boxes=[],
                )
            return
        for ocr_result in result.ocr_results.results:
            frame_media_id = frame_media_ids.get(ocr_result.frame_id)
            if frame_media_id is None:
                continue
            boxes = [
                OcrBoxCreate(
                    text=span.text,
                    confidence=span.confidence,
                    **self._bbox_kwargs(span.bbox),
                )
                for span in ocr_result.text_spans
            ]
            uow.results.upsert_ocr_result(
                frame_media_id,
                status=ocr_result.status.value,
                text=ocr_result.full_text,
                avg_confidence=ocr_result.confidence,
                model=getattr(self.pipeline.ocr_service, "backend", "paddleocr"),
                boxes=boxes,
            )

    def _persist_captions(self, uow: RepositoryUnitOfWork, result: PipelineResult, frame_media_ids: dict[str, uuid.UUID]) -> None:
        if result.caption_results is None:
            return
        assert uow.results is not None
        if not result.caption_results.results:
            for frame_media_id in frame_media_ids.values():
                uow.results.upsert_caption_result(
                    frame_media_id,
                    status=self._persistable_status(result.caption_results.status.value, result.caption_results.reason),
                    caption_text=result.caption_results.reason,
                    model=self.config.media_models.caption_model_name,
                )
            return
        for caption_result in result.caption_results.results:
            frame_media_id = frame_media_ids.get(caption_result.frame_id)
            if frame_media_id is None:
                continue
            uow.results.upsert_caption_result(
                frame_media_id,
                status=caption_result.status.value,
                caption_text=caption_result.caption_text,
                model=caption_result.generation_meta.get("model") or self.config.media_models.caption_model_name,
            )

    def _persist_detections(self, uow: RepositoryUnitOfWork, result: PipelineResult, frame_media_ids: dict[str, uuid.UUID]) -> None:
        if result.detection_results is None:
            return
        assert uow.results is not None
        if not result.detection_results.results:
            for frame_media_id in frame_media_ids.values():
                uow.results.upsert_object_result(
                    frame_media_id,
                    status=self._persistable_status(result.detection_results.status.value, result.detection_results.reason),
                    model=self.config.media_models.detection_model_name,
                    objects=[],
                )
            return
        for detection_result in result.detection_results.results:
            frame_media_id = frame_media_ids.get(detection_result.frame_id)
            if frame_media_id is None:
                continue
            objects = [
                DetectedObjectCreate(
                    label=detected.label,
                    confidence=detected.confidence,
                    **self._xyxy_kwargs(detected.bbox),
                )
                for detected in detection_result.detections
            ]
            uow.results.upsert_object_result(
                frame_media_id,
                status=detection_result.status.value,
                model=detection_result.inference_meta.get("model") or self.config.media_models.detection_model_name,
                objects=objects,
            )

    def _persist_transcript(self, uow: RepositoryUnitOfWork, media: MediaRecord, result: PipelineResult) -> None:
        if result.transcript_set is None:
            return
        assert uow.results is not None
        transcript = result.transcript_set
        segments = [
            TranscriptSegmentCreate(
                start_time=segment.start_sec,
                end_time=segment.end_sec,
                text=segment.text,
            )
            for segment in transcript.segments
        ]
        full_text = " ".join(segment.text for segment in transcript.segments if segment.text).strip()
        uow.results.upsert_transcript_for_video_media(
            media.media_id,
            status=self._persistable_status(transcript.status.value, transcript.reason),
            language=transcript.language,
            model=transcript.transcription_meta.get("model") or self.config.media_models.asr_model_name,
            full_text=full_text or None,
            segments=segments,
        )

    def _download_object(self, media: MediaRecord, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        object_key = MinioStorage.normalize_object_key(media.object_key)
        self.storage.client.fget_object(media.bucket_name, object_key, str(local_path))
        logger.info(f"Downloaded MinIO object '{media.bucket_name}/{object_key}' to '{local_path}'")

    def _upload_object(self, bucket_name: str, local_path: Path, object_key: str, content_type: str | None = None) -> None:
        object_key = MinioStorage.normalize_object_key(object_key)
        detected_type = content_type or mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
        self.storage.client.fput_object(bucket_name, object_key, str(local_path), content_type=detected_type)
        logger.info(f"Uploaded '{local_path}' to MinIO object '{bucket_name}/{object_key}'")

    def _write_keyframe_image(self, frame: Any, tmp_path: Path) -> Path:
        local_path = tmp_path / "frames" / f"frame_{frame.frame_index:06d}.jpg"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if frame.image_payload is None:
            if frame.image_path is None:
                raise ValueError(f"keyframe {frame.frame_id} has no image payload or image_path")
            source = Path(frame.image_path)
            if not source.is_file():
                raise FileNotFoundError(f"keyframe image does not exist: {source}")
            return source

        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required to write keyframe images") from exc

        success = cv2.imwrite(str(local_path), frame.image_payload, [int(cv2.IMWRITE_JPEG_QUALITY), self.config.media_pipeline.jpeg_quality])
        if not success:
            raise RuntimeError(f"failed to write keyframe image: {local_path}")
        return local_path

    @staticmethod
    def _persistable_status(status: str, reason: str | None = None) -> str:
        normalized_status = str(status).strip().lower()
        normalized_reason = str(reason or "").strip().lower()
        if normalized_status == "error":
            return "FAILED"
        if normalized_status == "skipped" and any(
            marker in normalized_reason
            for marker in ("missing dependency", "load failed", "failed:", "runtimeerror", "attributeerror")
        ):
            return "FAILED"
        return status

    @staticmethod
    def _frame_object_key(media: MediaRecord, frame: Any) -> str:
        stem = Path(media.object_key).stem
        return f"frames/{media.media_id}/{stem}_frame_{frame.frame_index:06d}.jpg"

    @staticmethod
    def _duration(result: PipelineResult) -> float | None:
        if result.validation_result and result.validation_result.technical_profile:
            duration = result.validation_result.technical_profile.duration
            if duration is not None:
                return duration
        raw_frames = VideoProcessingWorker._raw_frames(result)
        fps = VideoProcessingWorker._fps(result)
        if raw_frames is not None and fps:
            return raw_frames / fps
        return None

    @staticmethod
    def _fps(result: PipelineResult) -> float | None:
        if result.validation_result and result.validation_result.technical_profile:
            frame_rate = result.validation_result.technical_profile.frame_rate
            if frame_rate is not None:
                return frame_rate
        if result.keyframes:
            fps = result.keyframes.coverage_summary.get("fps")
            return float(fps) if fps is not None else None
        return None

    @staticmethod
    def _raw_frames(result: PipelineResult) -> int | None:
        if result.keyframes:
            total_frames = result.keyframes.selection_summary.get("total_frames")
            return int(total_frames) if total_frames is not None else None
        return None

    @staticmethod
    def _bbox_kwargs(bbox: list[list[float]] | None) -> dict[str, float | None]:
        if not bbox:
            return {"x1": None, "y1": None, "x2": None, "y2": None}
        points = [(float(point[0]), float(point[1])) for point in bbox if len(point) >= 2]
        if not points:
            return {"x1": None, "y1": None, "x2": None, "y2": None}
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)}

    @staticmethod
    def _xyxy_kwargs(bbox: list[float]) -> dict[str, float | None]:
        if len(bbox) < 4:
            return {"x1": None, "y1": None, "x2": None, "y2": None}
        return {"x1": float(bbox[0]), "y1": float(bbox[1]), "x2": float(bbox[2]), "y2": float(bbox[3])}

    @staticmethod
    def _pipeline_error_message(result: PipelineResult) -> str:
        if result.errors:
            return "; ".join(f"{error.stage}: {error.message}" for error in result.errors)
        return f"pipeline finished with status {result.status.value}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process video media from PostgreSQL + MinIO.")
    parser.add_argument("--media-id", help="Process one media_id directly instead of claiming a queued job.")
    parser.add_argument("--once", action="store_true", help="Claim and process one pending job.")
    parser.add_argument("--task-type", default=TaskType.KEYFRAME.value, help="Job task_type to claim. Default: keyframe.")
    parser.add_argument(
        "--keyframes-only",
        action="store_true",
        help="Disable OCR, captioning, object detection, and ASR for a lightweight first run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = AppConfig()
    if args.keyframes_only:
        config.media_pipeline.enable_ocr = False
        config.media_pipeline.enable_caption = False
        config.media_pipeline.enable_detection = False
        config.media_pipeline.enable_asr = False
    worker = VideoProcessingWorker(config=config)
    if args.media_id:
        result = worker.process_media(args.media_id)
        print(f"Processed media_id={args.media_id} status={result.status.value}")
        return

    if args.once:
        result = worker.process_next(args.task_type)
        if result is None:
            print("No pending job found.")
        else:
            print(f"Processed one job status={result.status.value}")
        return

    build_parser().print_help()


if __name__ == "__main__":
    main()
