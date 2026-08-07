from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.rag_video_anh.embedding.embedding_service import ImageEmbeddingConfigurationError
from src.rag_video_anh.vector_store.vector_store import VideoVectorStoreConfigurationError
from src.routers.media_retrieval import get_media_retrieval_service
from src.server import app


class FakeMediaRetrievalService:
    def __init__(self, result: dict | None = None, error: Exception | None = None) -> None:
        self.result = result if result is not None else {"total": 0, "found": False}
        self.error = error
        self.calls: list[tuple] = []

    def retrieve(self, query, top_k=None, video_ids=None, source="both", image=None, years=None):
        self.calls.append((query, top_k, video_ids, source, image, years))
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def override(service: FakeMediaRetrievalService) -> None:
    app.dependency_overrides[get_media_retrieval_service] = lambda: service


def test_media_search_route_is_registered():
    assert "/api/media/search" in {route.path for route in app.routes}


def test_post_media_search_returns_service_payload(client):
    service = FakeMediaRetrievalService(
        {
            "query": "sinh hoạt CLB",
            "source": "both",
            "clips": [{"video_id": "video-1", "timestamp_sec": 12.0}],
            "videos": [],
            "context": "[1] video video-1 tại 00:12",
            "total": 1,
            "found": True,
            "errors": [],
        }
    )
    override(service)

    response = client.post(
        "/api/media/search",
        json={"query": "sinh hoạt CLB", "top_k": 3, "video_ids": ["video-1"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["clips"][0]["video_id"] == "video-1"
    assert service.calls == [("sinh hoạt CLB", 3, ["video-1"], "both", None, None)]


def test_get_media_search_passes_query_params(client):
    service = FakeMediaRetrievalService()
    override(service)

    response = client.get("/api/media/search", params={"query": "cắm hoa", "source": "clip"})

    assert response.status_code == 200
    assert service.calls == [("cắm hoa", None, None, "clip", None, None)]


def test_post_media_search_forwards_years(client):
    """Bộ lọc năm phải đi tới service — sai chỗ này thì API vẫn 200 mà không lọc."""
    override(service := FakeMediaRetrievalService())

    response = client.post("/api/media/search", json={"query": "open day", "years": [2024, 2025]})

    assert response.status_code == 200
    assert service.calls[0][5] == [2024, 2025]


def test_get_media_search_forwards_years(client):
    override(service := FakeMediaRetrievalService())

    client.get("/api/media/search", params={"query": "open day", "years": [2024, 2025]})

    assert service.calls[0][5] == [2024, 2025]


def test_media_search_maps_invalid_year_to_422(client):
    """Năm rác là lỗi của client, không được thành 500."""
    override(FakeMediaRetrievalService())

    assert client.post("/api/media/search", json={"query": "a", "years": ["hai nghìn"]}).status_code == 422


def test_media_search_returns_not_found_payload_without_fabricating(client):
    service = FakeMediaRetrievalService(
        {"query": "abc", "source": "both", "clips": [], "videos": [], "context": "", "total": 0, "found": False, "errors": []}
    )
    override(service)

    body = client.post("/api/media/search", json={"query": "abc"}).json()

    assert body["found"] is False
    assert body["clips"] == [] and body["videos"] == []
    assert body["context"] == ""


def test_media_search_rejects_empty_query(client):
    override(FakeMediaRetrievalService())

    assert client.post("/api/media/search", json={"query": ""}).status_code == 422


def test_media_search_rejects_out_of_range_top_k(client):
    override(FakeMediaRetrievalService())

    assert client.post("/api/media/search", json={"query": "a", "top_k": 99}).status_code == 422


@pytest.mark.parametrize(
    "error",
    [
        ImageEmbeddingConfigurationError("Missing image embedding configuration. Set JINA_API_KEY."),
        VideoVectorStoreConfigurationError("Missing Qdrant URL. Set QDRANT_URL."),
    ],
)
def test_config_errors_map_to_503_not_422(client, error):
    """Hai lỗi cấu hình này kế thừa ValueError -> không được báo thành lỗi client."""
    override(FakeMediaRetrievalService(error=error))

    response = client.post("/api/media/search", json={"query": "a"})

    assert response.status_code == 503
    assert "chưa được cấu hình" in response.json()["detail"]


# ── Truy vấn bằng ảnh (US-302.1, US-303.1, US-502.1) ─────────────────────────

JPEG_BYTES = b"\xff\xd8\xff\xe0-anh-gia-cho-test"


def post_image(client, *, content_type="image/jpeg", data=None, blob=JPEG_BYTES):
    return client.post(
        "/api/media/search-image",
        files={"image": ("anh.jpg", blob, content_type)},
        data=data or {},
    )


def test_search_image_route_is_registered():
    assert "/api/media/search-image" in {route.path for route in app.routes}


def test_post_search_image_forwards_raw_bytes_and_optional_text(client):
    service = FakeMediaRetrievalService(
        {"query": "", "source": "clip", "clips": [], "videos": [], "context": "", "total": 0, "found": False, "errors": [], "notes": []}
    )
    override(service)

    response = post_image(
        client, data={"query": "ảnh này ở sự kiện nào", "source": "clip", "top_k": "3"}
    )

    assert response.status_code == 200
    query, top_k, video_ids, source, image, _years = service.calls[0]
    # Ảnh phải tới service nguyên vẹn: sai một byte là sai cả vector.
    assert image == JPEG_BYTES
    assert (query, top_k, video_ids, source) == ("ảnh này ở sự kiện nào", 3, None, "clip")


def test_post_search_image_allows_image_without_any_text(client):
    """US-502.1: gửi ảnh không kèm chữ vẫn phải chạy được."""
    override(service := FakeMediaRetrievalService())

    assert post_image(client).status_code == 200
    assert service.calls[0][0] is None


def test_search_image_rejects_non_image_content_type(client):
    override(FakeMediaRetrievalService())

    response = post_image(client, content_type="application/pdf")

    assert response.status_code == 422
    assert "JPG, PNG hoặc WEBP" in response.json()["detail"]


def test_search_image_rejects_oversized_image(client):
    """US-502.1 AC-2: quá hạn mức thì chặn ở server, không chỉ ở trình duyệt."""
    override(service := FakeMediaRetrievalService())

    response = post_image(client, blob=b"\xff\xd8\xff" + b"0" * (9 * 1024 * 1024))

    assert response.status_code == 413
    assert service.calls == []


def test_search_image_rejects_empty_file(client):
    override(FakeMediaRetrievalService())

    assert post_image(client, blob=b"").status_code == 422


def test_search_image_maps_unreadable_image_to_422_not_500(client):
    override(FakeMediaRetrievalService(error=ValueError("Không đọc được ảnh truy vấn (query-image-0).")))

    response = post_image(client)

    assert response.status_code == 422
    assert "Không đọc được ảnh truy vấn" in response.json()["detail"]


def test_search_image_maps_config_errors_to_503(client):
    override(FakeMediaRetrievalService(error=ImageEmbeddingConfigurationError("Set JINA_API_KEY.")))

    assert post_image(client).status_code == 503


def test_media_search_maps_invalid_source_to_422(client):
    service = FakeMediaRetrievalService(error=ValueError("source phải thuộc ('clip', 'transcript', 'both')"))
    override(service)

    response = client.post("/api/media/search", json={"query": "a", "source": "video"})

    assert response.status_code == 422
    assert "source" in response.json()["detail"]
