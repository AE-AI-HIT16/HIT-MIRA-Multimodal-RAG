"""HTTP routes cho consent (VÍ DỤ MẪU — khuôn router.py cho mọi domain).

Router chỉ lo: nhận request → gọi service → trả response. Không chứa logic.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.deps import require_admin
from app.domains.consent import service
from app.domains.consent.schemas import ConsentCreate, ConsentOut
from shared.db.session import get_session

router = APIRouter()


@router.post(
    "",
    response_model=ConsentOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_admin)],
)
def create_consent(payload: ConsentCreate, session: Session = Depends(get_session)) -> ConsentOut:
    """Lưu hồ sơ cấp quyền dữ liệu (chỉ admin)."""
    return service.create_consent(session, payload)


@router.get("/active", response_model=ConsentOut | None)
def active_consent(session: Session = Depends(get_session)) -> ConsentOut | None:
    """Xem consent còn hiệu lực (None nếu chưa có)."""
    return service.get_active_consent(session)
