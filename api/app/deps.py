"""Dependencies dùng chung cho các router (CHO SẴN)."""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.domains.consent import service as consent_service
from shared.db.session import get_session


def require_consent(session: Session = Depends(get_session)) -> None:
    """Cổng chặn ingestion: phải có consent hợp lệ mới cho nạp dữ liệu.

    [BR-101, BR-701 · US-101.1]  Không có consent còn hiệu lực → 403.
    """
    if consent_service.get_active_consent(session) is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Chưa có quyền sử dụng dữ liệu",
        )


def require_admin() -> None:
    """Chỉ cho phép role admin gọi endpoint quản trị.

    TODO(US-505 · NFR bảo mật): giải mã JWT, kiểm tra role == admin.
    Tạm no-op cho POC — sinh viên implement khi làm auth.
    """
    return None
