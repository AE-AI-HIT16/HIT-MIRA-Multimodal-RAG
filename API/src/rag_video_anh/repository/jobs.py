"""Repository for worker job queue state."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.rag_video_anh.repository.models import ProcessingJobModel
from src.rag_video_anh.repository.schemas import ProcessingJobCreate, ProcessingJobRecord, ProcessingStatus


class ProcessingJobRepository:
    """Read/write access for processing_jobs."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, job: ProcessingJobCreate) -> ProcessingJobRecord:
        row = ProcessingJobModel(media_id=job.media_id, task_type=job.task_type)
        self.session.add(row)
        self.session.flush()
        return self._to_record(row)

    def get(self, job_id) -> ProcessingJobRecord | None:
        row = self.session.get(ProcessingJobModel, job_id)
        return self._to_record(row) if row else None

    def claim_next(self, task_type: str) -> ProcessingJobRecord | None:
        stmt = (
            select(ProcessingJobModel)
            .where(
                ProcessingJobModel.task_type == task_type,
                ProcessingJobModel.status == ProcessingStatus.PENDING.value,
            )
            .order_by(ProcessingJobModel.created_at, ProcessingJobModel.job_id)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        row = self.session.scalar(stmt)
        if row is None:
            return None
        row.status = ProcessingStatus.PROCESSING.value
        row.started_at = datetime.utcnow()
        row.error_message = None
        self.session.flush()
        return self._to_record(row)

    def mark_done(self, job_id) -> ProcessingJobRecord:
        row = self._require_job(job_id)
        row.status = ProcessingStatus.DONE.value
        row.finished_at = datetime.utcnow()
        row.error_message = None
        self.session.flush()
        return self._to_record(row)

    def mark_failed(self, job_id, error_message: str, *, retry: bool = False, max_retries: int | None = None) -> ProcessingJobRecord:
        row = self._require_job(job_id)
        row.retry_count = int(row.retry_count or 0) + 1
        row.error_message = error_message
        row.finished_at = datetime.utcnow()
        if retry and (max_retries is None or row.retry_count <= max_retries):
            row.status = ProcessingStatus.PENDING.value
            row.started_at = None
            row.finished_at = None
        else:
            row.status = ProcessingStatus.FAILED.value
        self.session.flush()
        return self._to_record(row)

    def list_by_media(self, media_id) -> list[ProcessingJobRecord]:
        stmt = select(ProcessingJobModel).where(ProcessingJobModel.media_id == media_id).order_by(ProcessingJobModel.created_at)
        return [self._to_record(row) for row in self.session.scalars(stmt).all()]

    def _require_job(self, job_id) -> ProcessingJobModel:
        row = self.session.get(ProcessingJobModel, job_id)
        if row is None:
            raise ValueError(f"processing job does not exist: {job_id}")
        return row

    @staticmethod
    def _to_record(row: ProcessingJobModel) -> ProcessingJobRecord:
        return ProcessingJobRecord(
            job_id=row.job_id,
            media_id=row.media_id,
            task_type=row.task_type,
            status=row.status or ProcessingStatus.PENDING.value,
            retry_count=int(row.retry_count or 0),
            error_message=row.error_message,
            created_at=row.created_at,
            started_at=row.started_at,
            finished_at=row.finished_at,
        )
