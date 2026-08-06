"""Repository cho nguồn/bộ dữ liệu nằm trên object storage."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.rag_video_anh.repository.models import DatasetModel
from src.rag_video_anh.repository.schemas import DatasetCreate, DatasetRecord


class DatasetRepository:
    """Upsert dataset theo vị trí ổn định `(bucket_name, object_prefix)`."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_by_location(self, dataset: DatasetCreate) -> DatasetRecord:
        row = self.session.scalar(
            select(DatasetModel).where(
                DatasetModel.bucket_name == dataset.bucket_name,
                DatasetModel.object_prefix == dataset.object_prefix,
            )
        )
        values = {
            "name": dataset.name,
            "source_type": dataset.source_type,
            "bucket_name": dataset.bucket_name,
            "object_prefix": dataset.object_prefix,
            "source_url": dataset.source_url,
            "dataset_metadata": dict(dataset.metadata or {}),
        }
        if row is None:
            row = DatasetModel(**values)
            self.session.add(row)
        else:
            for field_name, value in values.items():
                setattr(row, field_name, value)
        self.session.flush()
        return self._to_record(row)

    def get(self, dataset_id) -> DatasetRecord | None:
        row = self.session.get(DatasetModel, dataset_id)
        return self._to_record(row) if row else None

    @staticmethod
    def _to_record(row: DatasetModel) -> DatasetRecord:
        return DatasetRecord(
            dataset_id=row.dataset_id,
            name=row.name,
            source_type=row.source_type,
            bucket_name=row.bucket_name,
            object_prefix=row.object_prefix,
            source_url=row.source_url,
            metadata=dict(row.dataset_metadata or {}),
            imported_at=row.imported_at,
        )
