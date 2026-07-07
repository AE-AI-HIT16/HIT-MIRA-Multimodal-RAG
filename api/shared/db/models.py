"""ORM models (SQLAlchemy).

VÍ DỤ MẪU cho sẵn: `Consent` (BR-101/701) — dùng làm khuôn thiết kế bảng.

TODO (vai Data Engineer): dựng nốt 18 bảng còn lại theo PRD §5 Data Model:
  users · posts · media_assets · video_frames · transcripts · captions ·
  ocr_texts · events · regulations · rule_chunks · embeddings · persons ·
  face_embeddings · person_appearances · conversations · messages ·
  eval_queries · query_logs · removal_requests
Mỗi bảng: khai báo cột đúng kiểu, quan hệ (FK/relationship), index cần thiết.
Xem docs/prd.md §5 để biết trường bắt buộc của từng bảng.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from shared.db.session import Base


class Consent(Base):
    """Hồ sơ cấp quyền dùng dữ liệu — cổng chặn ingestion (BR-101, BR-701)."""

    __tablename__ = "consents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scope: Mapped[str] = mapped_column(String(255))                 # phạm vi sử dụng
    granted_at: Mapped[datetime] = mapped_column(DateTime)          # ngày hiệu lực
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    signed_by: Mapped[str] = mapped_column(String(255))            # người ký (BCN)
    evidence_file: Mapped[str | None] = mapped_column(String(512), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)


# TODO(Data Engineer): thêm các model còn lại ở đây (xem docstring trên).
