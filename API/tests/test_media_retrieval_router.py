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

    def retrieve(self, query, top_k=None, video_ids=None, source="both"):
        self.calls.append((query, top_k, video_ids, source))
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
    assert service.calls == [("sinh hoạt CLB", 3, ["video-1"], "both")]


def test_get_media_search_passes_query_params(client):
    service = FakeMediaRetrievalService()
    override(service)

    response = client.get("/api/media/search", params={"query": "cắm hoa", "source": "clip"})

    assert response.status_code == 200
    assert service.calls == [("cắm hoa", None, None, "clip")]


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


def test_media_search_maps_invalid_source_to_422(client):
    service = FakeMediaRetrievalService(error=ValueError("source phải thuộc ('clip', 'transcript', 'both')"))
    override(service)

    response = client.post("/api/media/search", json={"query": "a", "source": "video"})

    assert response.status_code == 422
    assert "source" in response.json()["detail"]
