"""Quyền admin và contract nạp tài liệu nội quy."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.routers.auth import yeu_cau_admin
from src.routers.documents import get_pipeline_service, get_vector_store
from src.server import app


def test_cac_endpoint_tai_lieu_deu_can_quyen_admin() -> None:
    """Đặc biệt `/ingest` nhận đường dẫn server, nên không endpoint nào được mở."""
    app.dependency_overrides.clear()
    with TestClient(app) as khach:
        requests = (
            khach.post(
                "/api/documents/upload",
                files={"file": ("noi-quy.pdf", b"%PDF-1.4", "application/pdf")},
            ),
            khach.post(
                "/api/documents/ingest",
                json={"file_path": "/tmp/noi-quy.pdf"},
            ),
            khach.delete("/api/documents/noi-quy"),
        )

    assert all(response.status_code in (401, 403) for response in requests)


def test_admin_nap_pdf_nhan_dung_contract_pipeline() -> None:
    class FakePipeline:
        def ingest_document(self, _path: str, filename: str, document_id: str | None):
            return {
                "document_id": document_id or "noi-quy-2026",
                "filename": filename,
                "total_pages": 3,
                "total_documents": 3,
                "total_chunks": 12,
                "status": "completed",
                "errors": [],
                "ingest": {"upserted": 12},
            }

    app.dependency_overrides[yeu_cau_admin] = lambda: {
        "email": "admin@hit",
        "role": "admin",
    }
    app.dependency_overrides[get_pipeline_service] = FakePipeline
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/documents/upload",
                files={"file": ("noi-quy.pdf", b"%PDF-1.4", "application/pdf")},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 201
    assert response.json()["total_chunks"] == 12
    assert response.json()["filename"] == "noi-quy.pdf"


def test_guard_admin_chay_truoc_cac_dependency_ket_noi_ngoai() -> None:
    """Request lạ phải bị chặn trước khi dựng pipeline Qdrant/embedding."""
    app.dependency_overrides.clear()
    da_dung_pipeline = False
    da_dung_vector_store = False

    def pipeline_khong_duoc_goi():
        nonlocal da_dung_pipeline
        da_dung_pipeline = True
        raise AssertionError("pipeline đã được dựng trước khi kiểm tra quyền")

    def vector_store_khong_duoc_goi():
        nonlocal da_dung_vector_store
        da_dung_vector_store = True
        raise AssertionError("vector store đã được dựng trước khi kiểm tra quyền")

    app.dependency_overrides[get_pipeline_service] = pipeline_khong_duoc_goi
    app.dependency_overrides[get_vector_store] = vector_store_khong_duoc_goi
    try:
        with TestClient(app) as client:
            upload = client.post(
                "/api/documents/upload",
                files={"file": ("noi-quy.pdf", b"%PDF-1.4", "application/pdf")},
            )
            delete = client.delete("/api/documents/noi-quy")
    finally:
        app.dependency_overrides.clear()

    assert upload.status_code in (401, 403)
    assert delete.status_code in (401, 403)
    assert da_dung_pipeline is False
    assert da_dung_vector_store is False
