"""Logic nghiệp vụ consent (VÍ DỤ MẪU — viết đủ để làm khuôn).

Khuôn mẫu cho các domain khác: hàm nhận `Session` + schema, trả ORM/DTO,
không chứa chi tiết HTTP (đó là việc của router.py).
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domains.consent.schemas import ConsentCreate
from shared.db.models import Consent


def create_consent(session: Session, data: ConsentCreate) -> Consent:
    """Lưu hồ sơ cấp quyền (BR-101 · US-101.1 AC-1)."""
    consent = Consent(**data.model_dump(), enabled=True)
    session.add(consent)
    session.commit()
    session.refresh(consent)
    return consent


def get_active_consent(session: Session, now: datetime | None = None) -> Consent | None:
    """Trả consent còn hiệu lực (enabled + chưa hết hạn), hoặc None.

    Dùng cho cổng chặn ingestion (BR-701 · US-701.1).
    """
    now = now or datetime.now()
    stmt = select(Consent).where(Consent.enabled.is_(True)).order_by(Consent.granted_at.desc())
    for consent in session.scalars(stmt):
        if consent.expires_at is None or consent.expires_at >= now:
            return consent
    return None
