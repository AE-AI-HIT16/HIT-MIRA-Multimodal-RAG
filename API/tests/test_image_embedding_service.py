from __future__ import annotations

import base64
from pathlib import Path

import pytest

from src.rag_video_anh.embedding import embedding_service
from src.rag_video_anh.embedding.embedding_service import (
    RATE_LIMIT_COOLDOWN_SECONDS,
    TOKENS_PER_IMAGE,
    ImageEmbeddingConfigurationError,
    ImageEmbeddingService,
)


class FakeClock:
    """Đồng hồ giả để test việc giữ nhịp mà không phải chờ thật."""

    def __init__(self) -> None:
        self.now = 1_000.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def install(self, monkeypatch: pytest.MonkeyPatch) -> FakeClock:
        monkeypatch.setattr(embedding_service.time, "monotonic", self.monotonic)
        monkeypatch.setattr(embedding_service.time, "sleep", self.sleep)
        return self


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

    # Khai endpoint tường minh: bài test này nói về GIAO THỨC gửi đi, không phải
    # về việc endpoint mặc định trỏ đâu. Endpoint mặc định là chuyện vận hành —
    # nó đã đổi sang embedding server tự host, và đổi nữa cũng không được làm
    # đổ một bài test về định dạng request.
    service = ImageEmbeddingService(
        api_key="jina-test-key",
        base_url="https://api.jina.ai/v1/embeddings",
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
        base_url="https://api.jina.ai/v1/embeddings",
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


class RecordingPoster:
    """Trả vector giả theo đúng số phần tử của từng lô và ghi lại mọi request."""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    def __call__(self, url, *, headers, json, timeout):
        self.requests.append(json)
        offset = sum(len(request["input"]) for request in self.requests[:-1])
        return FakeResponse(
            {"data": [{"index": i, "embedding": [float(offset + i)]} for i in range(len(json["input"]))]}
        )


def _service(poster: RecordingPoster, **kwargs) -> ImageEmbeddingService:
    return ImageEmbeddingService(
        api_key="jina-test-key",
        base_url="https://embedding.invalid/v1/embeddings",
        dimensions=1,
        http_post=poster,
        **kwargs,
    )


def test_embed_images_splits_into_batches_and_keeps_order(tmp_path: Path) -> None:
    """Gộp cả trăm keyframe vào một request thì Jina trả 413."""
    paths = []
    for i in range(5):
        path = tmp_path / f"frame_{i}.jpg"
        path.write_bytes(f"frame-{i}".encode())
        paths.append(path)
    poster = RecordingPoster()

    vectors = _service(poster, image_batch_size=2).embed_images(paths)

    assert [len(request["input"]) for request in poster.requests] == [2, 2, 1]
    assert vectors == [[0.0], [1.0], [2.0], [3.0], [4.0]]


def test_embed_images_splits_when_byte_budget_is_exceeded(tmp_path: Path) -> None:
    """Keyframe nặng nhẹ khác nhau nên chỉ giới hạn số lượng là chưa đủ."""
    paths = []
    for i in range(3):
        path = tmp_path / f"big_{i}.jpg"
        path.write_bytes(b"x" * 3000)
        paths.append(path)
    poster = RecordingPoster()

    _service(poster, image_batch_size=10, max_request_bytes=5000).embed_images(paths)

    assert [len(request["input"]) for request in poster.requests] == [1, 1, 1]


def test_embed_images_downscales_oversized_keyframes(tmp_path: Path) -> None:
    """Ảnh 1280x720 tốn 24.000 token/ảnh, bản 512px chỉ tốn 4.000."""
    from io import BytesIO

    from PIL import Image

    path = tmp_path / "keyframe.jpg"
    Image.new("RGB", (1280, 720), color=(120, 40, 200)).save(path, format="JPEG", quality=95)
    poster = RecordingPoster()

    _service(poster, max_image_side=512).embed_images([path])

    sent = base64.b64decode(poster.requests[0]["input"][0]["image"])
    with Image.open(BytesIO(sent)) as image:
        assert max(image.size) == 512
    assert len(sent) < path.stat().st_size


def test_embed_images_keeps_full_resolution_when_downscaling_is_disabled(tmp_path: Path) -> None:
    from PIL import Image

    path = tmp_path / "keyframe.jpg"
    Image.new("RGB", (1280, 720), color=(10, 10, 10)).save(path, format="JPEG")
    poster = RecordingPoster()

    _service(poster, max_image_side=0).embed_images([path])

    assert base64.b64decode(poster.requests[0]["input"][0]["image"]) == path.read_bytes()


def test_embed_texts_splits_into_batches() -> None:
    poster = RecordingPoster()

    vectors = _service(poster, text_batch_size=2).embed_texts(["a", "b", "c"])

    assert [len(request["input"]) for request in poster.requests] == [2, 1]
    assert vectors == [[0.0], [1.0], [2.0]]


def test_rate_limited_retry_waits_for_the_whole_token_window() -> None:
    """Hạn mức Jina tính theo phút, backoff 1s/2s không giúp được gì."""

    class RateLimited(Exception):
        class response:  # noqa: N801 - chỉ để giả thuộc tính của httpx
            status_code = 429
            headers: dict = {}

    assert ImageEmbeddingService._retry_delay_seconds(RateLimited(), attempt=0) == RATE_LIMIT_COOLDOWN_SECONDS


def _throttle_service(**kwargs) -> ImageEmbeddingService:
    return ImageEmbeddingService(
        api_key="jina-test-key",
        base_url="https://embedding.invalid/v1/embeddings",
        dimensions=1,
        http_post=lambda *a, **k: None,
        **kwargs,
    )


def test_reserve_tokens_waits_only_until_the_oldest_batch_leaves_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Đo thực tế: chờ 429 rồi ngủ 60s chỉ đạt ~10 ảnh/phút, chủ động thì ~25."""
    clock = FakeClock().install(monkeypatch)
    service = _throttle_service(tokens_per_minute=10_000)

    service._reserve_tokens(6_000)
    assert clock.slept == []

    service._reserve_tokens(6_000)

    assert clock.slept == [pytest.approx(60.0)]


def test_reserve_tokens_can_be_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock().install(monkeypatch)
    service = _throttle_service(tokens_per_minute=0)

    for _ in range(5):
        service._reserve_tokens(TOKENS_PER_IMAGE * 100)

    assert clock.slept == []


def test_reserve_tokens_does_not_hang_when_one_request_exceeds_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chờ mãi cũng không lọt thì phải gửi đi, để nhánh thử lại 429 xử lý."""
    clock = FakeClock().install(monkeypatch)
    service = _throttle_service(tokens_per_minute=1_000)

    service._reserve_tokens(TOKENS_PER_IMAGE)

    assert clock.slept == []


def test_embed_images_reserves_the_token_budget_of_every_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ảnh thứ 25 trở đi trong một phút phải bị giữ lại, không được bắn thẳng."""
    clock = FakeClock().install(monkeypatch)
    paths = []
    for i in range(30):
        path = tmp_path / f"frame_{i}.jpg"
        path.write_bytes(f"frame-{i}".encode())
        paths.append(path)
    poster = RecordingPoster()

    ImageEmbeddingService(
        api_key="jina-test-key",
        base_url="https://embedding.invalid/v1/embeddings",
        dimensions=1,
        http_post=poster,
        image_batch_size=5,
        max_image_side=0,
    ).embed_images(paths)

    assert len(poster.requests) == 6
    # 30 ảnh = 120.000 token > 100.000/phút, nên phải có ít nhất một lần chờ.
    assert clock.slept


def test_image_embedding_service_requires_jina_api_key(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    service = ImageEmbeddingService(api_key="", dimensions=3, http_post=lambda *args, **kwargs: None)

    with pytest.raises(ImageEmbeddingConfigurationError, match="JINA_API_KEY"):
        service.embed_images([image_path])
