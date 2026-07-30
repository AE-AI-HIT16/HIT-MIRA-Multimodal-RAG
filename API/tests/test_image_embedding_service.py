from __future__ import annotations

import base64
from pathlib import Path

import pytest

from src.rag_video_anh.embedding.embedding_service import (
    ImageEmbeddingConfigurationError,
    ImageEmbeddingService,
)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.status_checked = False

    def raise_for_status(self) -> None:
        self.status_checked = True

    def json(self) -> dict:
        return self.payload


def test_image_embedding_service_posts_keyframes_to_jina_api(tmp_path: Path) -> None:
    first_image = tmp_path / "first.jpg"
    second_image = tmp_path / "second.jpg"
    first_image.write_bytes(b"first-image")
    second_image.write_bytes(b"second-image")
    captured: dict = {}
    response = FakeResponse(
        {
            "data": [
                {"index": 1, "embedding": [0.4, 0.5, 0.6]},
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
            ]
        }
    )

    def fake_post(url, *, headers, json, timeout):
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        return response

    service = ImageEmbeddingService(
        api_key="jina-test-key",
        model_name="jina-clip-v2",
        dimensions=3,
        timeout=12.0,
        http_post=fake_post,
    )

    vectors = service.embed_images([first_image, second_image])

    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert response.status_checked is True
    assert captured["url"] == "https://api.jina.ai/v1/embeddings"
    assert captured["headers"]["Authorization"] == "Bearer jina-test-key"
    assert captured["timeout"] == 12.0
    assert captured["json"]["model"] == "jina-clip-v2"
    assert captured["json"]["embedding_type"] == "float"
    assert captured["json"]["dimensions"] == 3
    assert captured["json"]["normalized"] is True
    assert base64.b64decode(captured["json"]["input"][0]["image"]) == b"first-image"
    assert base64.b64decode(captured["json"]["input"][1]["image"]) == b"second-image"


def test_image_embedding_service_posts_transcripts_to_jina_api() -> None:
    captured: dict = {}
    response = FakeResponse(
        {
            "data": [
                {"index": 1, "embedding": [0.4, 0.5, 0.6]},
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
            ]
        }
    )

    def fake_post(url, *, headers, json, timeout):
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        return response

    service = ImageEmbeddingService(
        api_key="jina-test-key",
        model_name="jina-clip-v2",
        dimensions=3,
        http_post=fake_post,
    )

    vectors = service.embed_texts(["mở đầu", "hoạt động"])

    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert captured["url"] == "https://api.jina.ai/v1/embeddings"
    assert captured["headers"]["Authorization"] == "Bearer jina-test-key"
    assert captured["json"]["model"] == "jina-clip-v2"
    assert captured["json"]["task"] == "retrieval.query"
    assert captured["json"]["embedding_type"] == "float"
    assert captured["json"]["dimensions"] == 3
    assert captured["json"]["normalized"] is True
    assert captured["json"]["input"] == [{"text": "mở đầu"}, {"text": "hoạt động"}]


def test_image_embedding_service_requires_jina_api_key(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    service = ImageEmbeddingService(api_key="", dimensions=3, http_post=lambda *args, **kwargs: None)

    with pytest.raises(ImageEmbeddingConfigurationError, match="JINA_API_KEY"):
        service.embed_images([image_path])
