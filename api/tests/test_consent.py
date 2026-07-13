"""Test domain mẫu consent (CHO SẴN, chạy xanh) — TC-101, TC-701.

Đây là KHUÔN test cho các domain khác: dùng fixture `client`/`db`, kiểm tra
Acceptance Criteria trong PRD.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi import HTTPException

from app.deps import require_consent
from app.domains.consent import service
from app.domains.consent.schemas import ConsentCreate


def _make(db, **over):
    data = ConsentCreate(
        scope="capstone", granted_at=datetime.now(), signed_by="BCN", **over
    )
    return service.create_consent(db, data)


# ---- Service / gate (TC-701) ----
def test_gate_blocks_without_consent(db):
    with pytest.raises(HTTPException) as exc:
        require_consent(session=db)
    assert exc.value.status_code == 403


def test_gate_allows_with_active_consent(db):
    _make(db)
    assert require_consent(session=db) is None


def test_expired_consent_not_active(db):
    _make(db, expires_at=datetime.now() - timedelta(days=1))
    assert service.get_active_consent(db) is None


# ---- Endpoints (TC-101) ----
def test_create_consent_endpoint(client):
    r = client.post(
        "/consents",
        json={"scope": "capstone", "granted_at": "2026-01-01T00:00:00", "signed_by": "BCN"},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["scope"] == "capstone" and body["enabled"] is True


def test_active_consent_endpoint(client):
    assert client.get("/consents/active").json() is None
    client.post(
        "/consents",
        json={"scope": "capstone", "granted_at": "2026-01-01T00:00:00", "signed_by": "BCN"},
    )
    assert client.get("/consents/active").json()["scope"] == "capstone"
