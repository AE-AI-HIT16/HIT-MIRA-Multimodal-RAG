"""Pydantic schemas cho consent (VÍ DỤ MẪU)."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ConsentCreate(BaseModel):
    scope: str
    granted_at: datetime
    expires_at: datetime | None = None
    signed_by: str
    evidence_file: str | None = None


class ConsentOut(ConsentCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    enabled: bool
