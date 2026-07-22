"""Task-specific workers for the PostgreSQL-backed media processing queue."""

from __future__ import annotations

import argparse
import mimetypes
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Any

from src.configuration import AppConfig
from src.log.logger import logger
from src.rag_video_anh.pipeline.asr_service import AsrService
from src.rag_video_anh.pipeline.caption_service import CaptionService
from src.rag_video_anh.pipeline.detection_service import DetectionService
from src.rag_video_anh.pipeline.keyframe_extractor import KeyframeExtractorService
from src.rag_video_anh.pipeline.media_router import MediaRouterService
from src.rag_video_anh.pipeline.media_validator import MediaValidatorService
from src.rag_video_anh.pipeline.minio_storage import MinioStorage
from src.rag_video_anh.pipeline.ocr_service import OCRService
from src.rag_video_anh.repository import (
    DetectedObjectCreate,
    FrameCreate,
    FrameRecord,
    MediaRecord,
    OcrBoxCreate,
    ProcessingJobCreate,
    ProcessingJobRecord,
    ProcessingStatus,
    RepositoryUnitOfWork,
    TaskType,
    TranscriptSegmentCreate,
    VideoMetadataCreate,
)
from src.rag_video_anh.schemas import (
    ASRRequest,
    CaptionRequest,
    CaptionResultSet,
    DetectionRequest,
    DetectionResultSet,
    KeyFrame,
    KeyFrameSet,
    KeyframeExtractionRequest,
    MediaInput,
    OCRRequest,
    OCRResultSet,
    StageStatus,
    TranscriptSet,
)


@dataclass(frozen=True)
class TaskRunResult:
    """Outcome of one claimed processing job."""

    success: bool
    message: str = ""


class MediaTaskWorker:
    """Claim and execute one media task type from processing_jobs.

    This is the queue-style worker:
    - keyframe jobs use a video media_id, create frame rows, then enqueue child jobs.
    - ocr/caption/object_detection jobs use a frame media_id.
    - asr jobs use a video media_id.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        storage: MinioStorage | None = None,
        uow_factory: type[RepositoryUnitOfWork] = RepositoryUnitOfWork,
        validator: MediaValidatorService | None = None,
        router: MediaRouterService | None = None,
        keyframe_extractor: KeyframeExtractorService | None = None,
        ocr_service: OCRService | None = None,
        caption_service: CaptionService | None = None,
        detection_service: DetectionService | None = None,
        asr_service: AsrService | None = None,
    ) -> None:
        self.config = config or AppConfig()
        self.storage = storage or MinioStorage(config=self.config)
        self.uow_factory = uow_factory
        self.validator = validator or MediaValidatorService(config=self.config)
        self.router = router or MediaRouterService(config=self.config)
        self.keyframe_extractor = keyframe_extractor or KeyframeExtractorService(config=self.config)
        self.ocr_service = ocr_service or OCRService(config=self.config)
        self.caption_service = caption_service or CaptionService(config=self.config)
        self.detection_service = detection_service or DetectionService(config=self.config)
        self.asr_service = asr_service or AsrService(config=self.config)

    def process_next(self, task_type: str) -> ProcessingJobRecord | None:
        """Claim and process one pending job for task_type."""

        with self.uow_factory() as uow:
            assert uow.jobs is not None
            job = uow.jobs.claim_next(task_type)

        if job is None:
            logger.info(f"No pending '{task_type}' job found")
            return None

        try:
            result = self.process_job(job)
            with self.uow_factory() as uow:
                assert uow.jobs is not None
                if result.success:
                    job = uow.jobs.mark_done(job.job_id)
                else:
                    job = uow.jobs.mark_failed(job.job_id, result.message)
            return job
        except Exception as exc:
            with self.uow_factory() as uow:
                assert uow.jobs is not None
                job = uow.jobs.mark_failed(
                    job.job_id,
                    str(exc),
                    retry=True,
                    max_retries=self.config.media_pipeline.retry_count,
                )
            return job

    def run_forever(self, task_type: str, poll_interval: float = 5.0) -> None:
        """Continuously poll PostgreSQL for jobs of one task type."""

        logger.info(f"Starting '{task_type}' worker loop with poll_interval={poll_interval}s")
        while True:
            job = self.process_next(task_type)
            if job is None:
                sleep(poll_interval)

    def process_job(self, job: ProcessingJobRecord) -> TaskRunResult:
        """Dispatch a claimed job to the matching task handler."""

        task_type = str(job.task_type)
        if task_type == TaskType.KEYFRAME.value:
            return self.process_keyframe(job.media_id)
        if task_type == TaskType.OCR.value:
            return self.process_ocr(job.media_id)
        if task_type == TaskType.CAPTION.value:
            return self.process_caption(job.media_id)
        if task_type == TaskType.OBJECT_DETECTION.value:
            return self.process_object_detection(job.media_id)
        if task_type == TaskType.ASR.value:
            return self.process_asr(job.media_id)
        return TaskRunResult(False, f"unsupported task_type: {task_type}")

    def process_keyframe(self, video_media_id: str | uuid.UUID) -> TaskRunResult:
        """Extract keyframes for a video and enqueue downstream frame/video jobs."""

        media = self._get_media(video_media_id)
        if media.media_type != "video":
            return TaskRunResult(False, f"keyframe task requires video media, got {media.media_type}")

        suffix = Path(media.object_key).suffix or ".mp4"
        with tempfile.TemporaryDirectory(prefix="hit-mira-keyframe-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            local_video = tmp_path / f"{media.media_id}{suffix}"
            self._download_object(media, local_video)

            media_input = MediaInput(
                media_id=str(media.media_id),
                media_type=media.media_type,
                source_ref=media.object_key,
                media_path=str(local_video),
                metadata={"bucket_name": media.bucket_name, "object_key": media.object_key},
            )
            validation_result = self.validator.validate(media_input)
            if not validation_result.is_valid:
                return TaskRunResult(False, "; ".join(validation_result.errors))

            route = self.router.select_route(validation_result)
            keyframes = self.keyframe_extractor.extract(
                KeyframeExtractionRequest(media_input=media_input, validation_result=validation_result, route=route)
            )

            with self.uow_factory() as uow:
                assert uow.media is not None
                uow.media.upsert_video_metadata(
                    VideoMetadataCreate(
                        media_id=media.media_id,
                        duration=self._duration(keyframes),
                        fps=self._fps(keyframes),
                        raw_frames=self._raw_frames(keyframes),
                        selected_keyframes=len(keyframes.frames),
                    )
                )
                frames = self._persist_keyframes(uow, media, keyframes, tmp_path)
                self._enqueue_downstream_jobs(uow, media, frames)

            if keyframes.status != StageStatus.DONE or not keyframes.frames:
                return TaskRunResult(False, keyframes.reason or "keyframe extraction did not produce frames")
            return TaskRunResult(True, f"extracted {len(keyframes.frames)} keyframe(s)")

    def process_ocr(self, frame_media_id: str | uuid.UUID) -> TaskRunResult:
        """Run OCR for one frame media row."""

        frame, media = self._get_frame_and_media(frame_media_id)
        with tempfile.TemporaryDirectory(prefix="hit-mira-ocr-") as tmp_dir:
            local_frame = self._download_frame(media, Path(tmp_dir))
            keyframe = self._keyframe_from_record(frame, local_frame)
            result = self.ocr_service.recognize(
                OCRRequest(
                    media_id=str(media.media_id),
                    keyframes=KeyFrameSet(media_id=str(media.media_id), frames=[keyframe], status=StageStatus.DONE),
                )
            )
        self._persist_ocr(media.media_id, result)
        return self._task_result_from_stage(result.status.value, result.reason)

    def process_caption(self, frame_media_id: str | uuid.UUID) -> TaskRunResult:
        """Generate a caption for one frame media row."""

        frame, media = self._get_frame_and_media(frame_media_id)
        with tempfile.TemporaryDirectory(prefix="hit-mira-caption-") as tmp_dir:
            local_frame = self._download_frame(media, Path(tmp_dir))
            keyframe = self._keyframe_from_record(frame, local_frame)
            result = self.caption_service.caption(
                CaptionRequest(
                    media_id=str(media.media_id),
                    keyframes=KeyFrameSet(media_id=str(media.media_id), frames=[keyframe], status=StageStatus.DONE),
                )
            )
        self._persist_caption(media.media_id, result)
        return self._task_result_from_stage(result.status.value, result.reason)

    def process_object_detection(self, frame_media_id: str | uuid.UUID) -> TaskRunResult:
        """Run object detection for one frame media row."""

        frame, media = self._get_frame_and_media(frame_media_id)
        with tempfile.TemporaryDirectory(prefix="hit-mira-object-") as tmp_dir:
            local_frame = self._download_frame(media, Path(tmp_dir))
            keyframe = self._keyframe_from_record(frame, local_frame)
            result = self.detection_service.detect(
                DetectionRequest(
                    media_id=str(media.media_id),
                    keyframes=KeyFrameSet(media_id=str(media.media_id), frames=[keyframe], status=StageStatus.DONE),
                )
            )
        self._persist_detection(media.media_id, result)
        return self._task_result_from_stage(result.status.value, result.reason)

    def process_asr(self, video_media_id: str | uuid.UUID) -> TaskRunResult:
        """Run ASR for one video media row."""

        media = self._get_media(video_media_id)
        if media.media_type != "video":
            return TaskRunResult(False, f"asr task requires video media, got {media.media_type}")

        suffix = Path(media.object_key).suffix or ".mp4"
        with tempfile.TemporaryDirectory(prefix="hit-mira-asr-") as tmp_dir:
            local_video = Path(tmp_dir) / f"{media.media_id}{suffix}"
            self._download_object(media, local_video)
            media_input = MediaInput(
                media_id=str(media.media_id),
                media_type=media.media_type,
                source_ref=media.object_key,
                media_path=str(local_video),
                metadata={"bucket_name": media.bucket_name, "object_key": media.object_key},
            )
            result = self.asr_service.transcribe(ASRRequest(media_input=media_input))

        self._persist_transcript(media.media_id, result)
        return self._task_result_from_stage(result.status.value, result.reason)

    def _persist_keyframes(
        self,
        uow: RepositoryUnitOfWork,
        media: MediaRecord,
        keyframes: KeyFrameSet,
        tmp_path: Path,
    ) -> list[FrameRecord]:
        assert uow.media is not None
        persisted_frames: list[FrameRecord] = []
        existing_frames = uow.media.list_frames_for_video(media.media_id)
        existing_by_index = {frame.frame_index: frame for frame in existing_frames if frame.frame_index is not None}
        existing_by_key = {frame.object_key: frame for frame in existing_frames}

        for frame in keyframes.frames:
            object_key = self._frame_object_key(media, frame)
            existing = existing_by_index.get(frame.frame_index) or existing_by_key.get(object_key)
            if existing is not None:
                persisted_frames.append(existing)
                logger.info(f"Reusing existing keyframe media_id {existing.media_id} for frame_index {frame.frame_index}")
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
            persisted_frames.append(frame_record)
            existing_by_index[frame.frame_index] = frame_record
            existing_by_key[object_key] = frame_record
        return persisted_frames

    def _enqueue_downstream_jobs(self, uow: RepositoryUnitOfWork, media: MediaRecord, frames: list[FrameRecord]) -> None:
        if self.config.media_pipeline.enable_ocr:
            for frame in frames:
                self._ensure_job(uow, frame.media_id, TaskType.OCR.value)
        if self.config.media_pipeline.enable_caption:
            for frame in frames:
                self._ensure_job(uow, frame.media_id, TaskType.CAPTION.value)
        if self.config.media_pipeline.enable_detection:
            for frame in frames:
                self._ensure_job(uow, frame.media_id, TaskType.OBJECT_DETECTION.value)
        if self.config.media_pipeline.enable_asr:
            self._ensure_job(uow, media.media_id, TaskType.ASR.value)

    def _ensure_job(self, uow: RepositoryUnitOfWork, media_id: uuid.UUID, task_type: str) -> None:
        assert uow.jobs is not None
        existing = uow.jobs.list_by_media(media_id)
        active_statuses = {
            ProcessingStatus.PENDING.value,
            ProcessingStatus.PROCESSING.value,
            ProcessingStatus.DONE.value,
        }
        if any(job.task_type == task_type and job.status in active_statuses for job in existing):
            return
        uow.jobs.create(ProcessingJobCreate(media_id=media_id, task_type=task_type))

    def _persist_ocr(self, frame_media_id: uuid.UUID, result: OCRResultSet) -> None:
        with self.uow_factory() as uow:
            assert uow.results is not None
            if not result.results:
                uow.results.upsert_ocr_result(
                    frame_media_id,
                    status=self._persistable_status(result.status.value, result.reason),
                    text=result.reason,
                    model="paddleocr",
                    boxes=[],
                )
                return
            ocr_result = result.results[0]
            boxes = [
                OcrBoxCreate(text=span.text, confidence=span.confidence, **self._bbox_kwargs(span.bbox))
                for span in ocr_result.text_spans
            ]
            uow.results.upsert_ocr_result(
                frame_media_id,
                status=self._persistable_status(ocr_result.status.value, ocr_result.reason),
                text=ocr_result.full_text or ocr_result.reason,
                avg_confidence=ocr_result.confidence,
                model="paddleocr",
                boxes=boxes,
            )

    def _persist_caption(self, frame_media_id: uuid.UUID, result: CaptionResultSet) -> None:
        with self.uow_factory() as uow:
            assert uow.results is not None
            if not result.results:
                uow.results.upsert_caption_result(
                    frame_media_id,
                    status=self._persistable_status(result.status.value, result.reason),
                    caption_text=result.reason,
                    model=self.config.media_models.caption_model_name,
                )
                return
            caption_result = result.results[0]
            uow.results.upsert_caption_result(
                frame_media_id,
                status=self._persistable_status(caption_result.status.value, caption_result.reason),
                caption_text=caption_result.caption_text or caption_result.reason,
                model=caption_result.generation_meta.get("model") or self.config.media_models.caption_model_name,
            )

    def _persist_detection(self, frame_media_id: uuid.UUID, result: DetectionResultSet) -> None:
        with self.uow_factory() as uow:
            assert uow.results is not None
            if not result.results:
                uow.results.upsert_object_result(
                    frame_media_id,
                    status=self._persistable_status(result.status.value, result.reason),
                    model=self.config.media_models.detection_model_name,
                    objects=[],
                )
                return
            detection_result = result.results[0]
            objects = [
                DetectedObjectCreate(label=detected.label, confidence=detected.confidence, **self._xyxy_kwargs(detected.bbox))
                for detected in detection_result.detections
            ]
            uow.results.upsert_object_result(
                frame_media_id,
                status=self._persistable_status(detection_result.status.value, detection_result.reason),
                model=detection_result.inference_meta.get("model") or self.config.media_models.detection_model_name,
                objects=objects,
            )

    def _persist_transcript(self, video_media_id: uuid.UUID, result: TranscriptSet) -> None:
        with self.uow_factory() as uow:
            assert uow.media is not None
            assert uow.results is not None
            uow.media.upsert_video_metadata(VideoMetadataCreate(media_id=video_media_id))
            segments = [
                TranscriptSegmentCreate(start_time=segment.start_sec, end_time=segment.end_sec, text=segment.text)
                for segment in result.segments
            ]
            full_text = " ".join(segment.text for segment in result.segments if segment.text).strip()
            uow.results.upsert_transcript_for_video_media(
                video_media_id,
                status=self._persistable_status(result.status.value, result.reason),
                language=result.language,
                model=result.transcription_meta.get("model") or self.config.media_models.whisper_model_size,
                full_text=full_text or result.reason,
                segments=segments,
            )

    def _get_media(self, media_id: str | uuid.UUID) -> MediaRecord:
        with self.uow_factory() as uow:
            assert uow.media is not None
            media = uow.media.get_media(media_id)
        if media is None:
            raise ValueError(f"media does not exist: {media_id}")
        return media

    def _get_frame_and_media(self, frame_media_id: str | uuid.UUID) -> tuple[FrameRecord, MediaRecord]:
        with self.uow_factory() as uow:
            assert uow.media is not None
            media = uow.media.get_media(frame_media_id)
            frame = uow.media.get_frame_by_media_id(frame_media_id)
        if media is None:
            raise ValueError(f"media does not exist: {frame_media_id}")
        if media.media_type != "frame":
            raise ValueError(f"frame task requires frame media, got {media.media_type}")
        if frame is None:
            raise ValueError(f"frame metadata does not exist for media_id: {frame_media_id}")
        return frame, media

    def _download_frame(self, media: MediaRecord, tmp_path: Path) -> Path:
        suffix = Path(media.object_key).suffix or ".jpg"
        local_frame = tmp_path / f"{media.media_id}{suffix}"
        self._download_object(media, local_frame)
        return local_frame

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

    def _write_keyframe_image(self, frame: KeyFrame, tmp_path: Path) -> Path:
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
    def _keyframe_from_record(frame: FrameRecord, image_path: Path) -> KeyFrame:
        timestamp_sec = float(frame.timestamp or 0.0)
        frame_index = int(frame.frame_index or 0)
        return KeyFrame(
            frame_id=str(frame.media_id),
            media_id=str(frame.media_id),
            frame_index=frame_index,
            timestamp_ms=int(timestamp_sec * 1000),
            timestamp_sec=timestamp_sec,
            image_path=str(image_path),
        )

    @staticmethod
    def _task_result_from_stage(status: str, reason: str | None = None) -> TaskRunResult:
        persisted_status = MediaTaskWorker._persistable_status(status, reason)
        if persisted_status == ProcessingStatus.FAILED.value:
            return TaskRunResult(False, reason or f"stage status: {status}")
        return TaskRunResult(True, reason or f"stage status: {status}")

    @staticmethod
    def _persistable_status(status: str, reason: str | None = None) -> str:
        normalized_status = str(status).strip().lower()
        normalized_reason = str(reason or "").strip().lower()
        if normalized_status == "error":
            return ProcessingStatus.FAILED.value
        if normalized_status == "skipped" and any(
            marker in normalized_reason
            for marker in ("missing dependency", "load failed", "failed:", "runtimeerror", "attributeerror")
        ):
            return ProcessingStatus.FAILED.value
        return status

    @staticmethod
    def _frame_object_key(media: MediaRecord, frame: KeyFrame) -> str:
        stem = Path(media.object_key).stem
        return f"frames/{media.media_id}/{stem}_frame_{frame.frame_index:06d}.jpg"

    @staticmethod
    def _duration(keyframes: KeyFrameSet) -> float | None:
        raw_frames = MediaTaskWorker._raw_frames(keyframes)
        fps = MediaTaskWorker._fps(keyframes)
        if raw_frames is not None and fps:
            return raw_frames / fps
        return None

    @staticmethod
    def _fps(keyframes: KeyFrameSet) -> float | None:
        fps = keyframes.coverage_summary.get("fps")
        return float(fps) if fps is not None else None

    @staticmethod
    def _raw_frames(keyframes: KeyFrameSet) -> int | None:
        total_frames = keyframes.selection_summary.get("total_frames")
        return int(total_frames) if total_frames is not None else None

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one task-specific media worker.")
    parser.add_argument(
        "--task-type",
        required=True,
        choices=[task.value for task in TaskType if task.value != TaskType.EMBEDDING.value],
        help="Type of job this worker should claim.",
    )
    parser.add_argument("--once", action="store_true", help="Claim at most one pending job and exit.")
    parser.add_argument("--loop", action="store_true", help="Poll PostgreSQL forever for this task type.")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds to sleep when no job is found.")
    parser.add_argument("--media-id", help="Debug mode: run this task directly for one media_id without claiming a job.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    worker = MediaTaskWorker()

    if args.media_id:
        result = worker.process_job(
            ProcessingJobRecord(
                media_id=uuid.UUID(str(args.media_id)),
                task_type=args.task_type,
            )
        )
        print(f"Processed {args.task_type} media_id={args.media_id} success={result.success} message={result.message}")
        return

    if args.loop:
        worker.run_forever(args.task_type, poll_interval=args.poll_interval)
        return

    if args.once:
        job = worker.process_next(args.task_type)
        if job is None:
            print(f"No pending {args.task_type} job found.")
        else:
            print(f"Processed job_id={job.job_id} task_type={job.task_type} status={job.status}")
        return

    build_parser().print_help()


if __name__ == "__main__":
    main()
