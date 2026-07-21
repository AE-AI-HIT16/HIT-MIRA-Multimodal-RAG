"""Repository for AI processing results."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from src.rag_video_anh.repository.models import (
    CaptionResultModel,
    DetectedObjectModel,
    EmbeddingModel,
    ObjectResultModel,
    OcrBoxModel,
    OcrResultModel,
    TranscriptModel,
    TranscriptSegmentModel,
    VideoModel,
)
from src.rag_video_anh.repository.schemas import (
    CaptionResultRecord,
    DetectedObjectCreate,
    DetectedObjectRecord,
    EmbeddingRecord,
    ObjectResultRecord,
    OcrBoxCreate,
    OcrBoxRecord,
    OcrResultRecord,
    ProcessingStatus,
    TranscriptRecord,
    TranscriptSegmentCreate,
    TranscriptSegmentRecord,
)


class AIResultRepository:
    """Read/write access for all AI result tables.

    Methods are intentionally organized by service output. Services provide
    domain DTOs; this layer owns the table layout and child-row replacement.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_ocr_result(
        self,
        media_id,
        *,
        status: str = ProcessingStatus.DONE.value,
        text: str | None = None,
        avg_confidence: float | None = None,
        model: str | None = None,
        boxes: list[OcrBoxCreate] | None = None,
    ) -> OcrResultRecord:
        row = self.session.scalar(
            select(OcrResultModel)
            .options(selectinload(OcrResultModel.boxes))
            .where(OcrResultModel.media_id == media_id)
        )
        if row is None:
            row = OcrResultModel(media_id=media_id)
            self.session.add(row)
            self.session.flush()

        row.status = self._status(status)
        row.text = text
        row.avg_confidence = avg_confidence
        row.model = model
        if boxes is not None:
            row.boxes.clear()
            for box in boxes:
                row.boxes.append(
                    OcrBoxModel(
                        text=box.text,
                        confidence=box.confidence,
                        x1=box.x1,
                        y1=box.y1,
                        x2=box.x2,
                        y2=box.y2,
                    )
                )
        self.session.flush()
        return self._ocr_record(row)

    def get_ocr_result(self, media_id) -> OcrResultRecord | None:
        row = self.session.scalar(
            select(OcrResultModel)
            .options(selectinload(OcrResultModel.boxes))
            .where(OcrResultModel.media_id == media_id)
        )
        return self._ocr_record(row) if row else None

    def upsert_caption_result(
        self,
        media_id,
        *,
        status: str = ProcessingStatus.DONE.value,
        caption_text: str | None = None,
        model: str | None = None,
    ) -> CaptionResultRecord:
        row = self.session.scalar(select(CaptionResultModel).where(CaptionResultModel.media_id == media_id))
        if row is None:
            row = CaptionResultModel(media_id=media_id)
            self.session.add(row)

        row.status = self._status(status)
        row.caption_text = caption_text
        row.model = model
        self.session.flush()
        return self._caption_record(row)

    def get_caption_result(self, media_id) -> CaptionResultRecord | None:
        row = self.session.scalar(select(CaptionResultModel).where(CaptionResultModel.media_id == media_id))
        return self._caption_record(row) if row else None

    def upsert_object_result(
        self,
        media_id,
        *,
        status: str = ProcessingStatus.DONE.value,
        model: str | None = None,
        objects: list[DetectedObjectCreate] | None = None,
    ) -> ObjectResultRecord:
        row = self.session.scalar(
            select(ObjectResultModel)
            .options(selectinload(ObjectResultModel.objects))
            .where(ObjectResultModel.media_id == media_id)
        )
        if row is None:
            row = ObjectResultModel(media_id=media_id)
            self.session.add(row)
            self.session.flush()

        row.status = self._status(status)
        row.model = model
        if objects is not None:
            row.objects.clear()
            for detected in objects:
                row.objects.append(
                    DetectedObjectModel(
                        label=detected.label,
                        confidence=detected.confidence,
                        x1=detected.x1,
                        y1=detected.y1,
                        x2=detected.x2,
                        y2=detected.y2,
                    )
                )
        self.session.flush()
        return self._object_record(row)

    def get_object_result(self, media_id) -> ObjectResultRecord | None:
        row = self.session.scalar(
            select(ObjectResultModel)
            .options(selectinload(ObjectResultModel.objects))
            .where(ObjectResultModel.media_id == media_id)
        )
        return self._object_record(row) if row else None

    def upsert_transcript_for_video_media(
        self,
        video_media_id,
        *,
        status: str = ProcessingStatus.DONE.value,
        language: str | None = None,
        model: str | None = None,
        full_text: str | None = None,
        segments: list[TranscriptSegmentCreate] | None = None,
    ) -> TranscriptRecord:
        video = self.session.scalar(select(VideoModel).where(VideoModel.media_id == video_media_id))
        if video is None:
            raise ValueError(f"video metadata does not exist for media_id: {video_media_id}")
        return self.upsert_transcript(
            video.video_id,
            status=status,
            language=language,
            model=model,
            full_text=full_text,
            segments=segments,
        )

    def upsert_transcript(
        self,
        video_id,
        *,
        status: str = ProcessingStatus.DONE.value,
        language: str | None = None,
        model: str | None = None,
        full_text: str | None = None,
        segments: list[TranscriptSegmentCreate] | None = None,
    ) -> TranscriptRecord:
        row = self.session.scalar(
            select(TranscriptModel)
            .options(selectinload(TranscriptModel.segments))
            .where(TranscriptModel.video_id == video_id)
        )
        if row is None:
            row = TranscriptModel(video_id=video_id)
            self.session.add(row)
            self.session.flush()

        row.status = self._status(status)
        row.language = language
        row.model = model
        row.full_text = full_text
        if segments is not None:
            row.segments.clear()
            for segment in segments:
                row.segments.append(
                    TranscriptSegmentModel(
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        text=segment.text,
                    )
                )
        self.session.flush()
        return self._transcript_record(row)

    def get_transcript_by_video_id(self, video_id) -> TranscriptRecord | None:
        row = self.session.scalar(
            select(TranscriptModel)
            .options(selectinload(TranscriptModel.segments))
            .where(TranscriptModel.video_id == video_id)
        )
        return self._transcript_record(row) if row else None

    def get_transcript_by_video_media_id(self, video_media_id) -> TranscriptRecord | None:
        video = self.session.scalar(select(VideoModel).where(VideoModel.media_id == video_media_id))
        if video is None:
            return None
        return self.get_transcript_by_video_id(video.video_id)

    def upsert_embedding(self, media_id, *, vector_db_id: str | None = None, model: str | None = None) -> EmbeddingRecord:
        row = self.session.scalar(select(EmbeddingModel).where(EmbeddingModel.media_id == media_id))
        if row is None:
            row = EmbeddingModel(media_id=media_id)
            self.session.add(row)
        row.vector_db_id = vector_db_id
        row.model = model
        self.session.flush()
        return self._embedding_record(row)

    def get_embedding(self, media_id) -> EmbeddingRecord | None:
        row = self.session.scalar(select(EmbeddingModel).where(EmbeddingModel.media_id == media_id))
        return self._embedding_record(row) if row else None

    @staticmethod
    def _status(status: str | ProcessingStatus) -> str:
        value = status.value if isinstance(status, ProcessingStatus) else str(status)
        normalized = value.strip().upper()
        aliases = {
            "DONE": ProcessingStatus.DONE.value,
            "SUCCESS": ProcessingStatus.DONE.value,
            "PENDING": ProcessingStatus.PENDING.value,
            "PROCESSING": ProcessingStatus.PROCESSING.value,
            "ERROR": ProcessingStatus.FAILED.value,
            "FAILED": ProcessingStatus.FAILED.value,
            "FAILURE": ProcessingStatus.FAILED.value,
            "SKIPPED": ProcessingStatus.DONE.value,
            "NOT_FOUND": ProcessingStatus.DONE.value,
        }
        return aliases.get(normalized, normalized)

    @staticmethod
    def _ocr_record(row: OcrResultModel) -> OcrResultRecord:
        boxes = [
            OcrBoxRecord(
                box_id=box.box_id,
                ocr_id=box.ocr_id,
                text=box.text,
                confidence=box.confidence,
                x1=box.x1,
                y1=box.y1,
                x2=box.x2,
                y2=box.y2,
            )
            for box in sorted(row.boxes, key=lambda item: str(item.box_id))
        ]
        return OcrResultRecord(
            ocr_id=row.ocr_id,
            media_id=row.media_id,
            status=row.status or ProcessingStatus.PENDING.value,
            text=row.text,
            avg_confidence=row.avg_confidence,
            model=row.model,
            created_at=row.created_at,
            boxes=boxes,
        )

    @staticmethod
    def _caption_record(row: CaptionResultModel) -> CaptionResultRecord:
        return CaptionResultRecord(
            caption_id=row.caption_id,
            media_id=row.media_id,
            status=row.status or ProcessingStatus.PENDING.value,
            caption_text=row.caption_text,
            model=row.model,
            created_at=row.created_at,
        )

    @staticmethod
    def _object_record(row: ObjectResultModel) -> ObjectResultRecord:
        objects = [
            DetectedObjectRecord(
                object_id=detected.object_id,
                object_result_id=detected.object_result_id,
                label=detected.label,
                confidence=detected.confidence,
                x1=detected.x1,
                y1=detected.y1,
                x2=detected.x2,
                y2=detected.y2,
            )
            for detected in sorted(row.objects, key=lambda item: str(item.object_id))
        ]
        return ObjectResultRecord(
            object_result_id=row.object_result_id,
            media_id=row.media_id,
            status=row.status or ProcessingStatus.PENDING.value,
            model=row.model,
            created_at=row.created_at,
            objects=objects,
        )

    @staticmethod
    def _transcript_record(row: TranscriptModel) -> TranscriptRecord:
        segments = [
            TranscriptSegmentRecord(
                segment_id=segment.segment_id,
                transcript_id=segment.transcript_id,
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=segment.text,
            )
            for segment in sorted(row.segments, key=lambda item: ((item.start_time or 0.0), str(item.segment_id)))
        ]
        return TranscriptRecord(
            transcript_id=row.transcript_id,
            video_id=row.video_id,
            status=row.status or ProcessingStatus.PENDING.value,
            language=row.language,
            model=row.model,
            full_text=row.full_text,
            created_at=row.created_at,
            segments=segments,
        )

    @staticmethod
    def _embedding_record(row: EmbeddingModel) -> EmbeddingRecord:
        return EmbeddingRecord(
            embedding_id=row.embedding_id,
            media_id=row.media_id,
            vector_db_id=row.vector_db_id,
            model=row.model,
            created_at=row.created_at,
        )
