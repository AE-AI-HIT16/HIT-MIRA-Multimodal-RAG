"""Truy vấn bằng ẢNH — US-302.1, US-303.1, US-502.1.

Điều đang được chứng minh: ảnh truy vấn được nhúng thành vector rồi so trực
tiếp với vector ảnh trong `media_clip`, chứ không đi vòng qua việc tả ảnh bằng
lời rồi tìm bằng chữ. Toàn bộ chạy offline bằng fake — không gọi Jina, không
gọi Qdrant.
"""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import pytest

from src.rag_video_anh.embedding.embedding_service import ImageEmbeddingService
from src.rag_video_anh.retrieval.retrieval_service import VideoRetrievalService
from src.rag_video_anh.retrieval.retriever import VideoRetriever

MEDIA_CLIP = "media_clip"
VIDEO_TRANSCRIPT = "video_transcript"

TEXT_VECTOR = [0.1, 0.2, 0.3]
IMAGE_VECTOR = [0.7, 0.8, 0.9]


def png_bytes(size: tuple[int, int] = (64, 64)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def stub_config(top_k: int | None = 5) -> SimpleNamespace:
    return SimpleNamespace(
        retrieval=SimpleNamespace(top_k=top_k),
        qdrant=SimpleNamespace(url="http://localhost:6333", api_key=None, distance="COSINE"),
    )


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class FakeEmbedder:
    """Phân biệt rõ lượt nhúng chữ và lượt nhúng ảnh để test soi được đường đi."""

    def __init__(self) -> None:
        self.text_calls: list[list[str]] = []
        self.image_calls: list[list[bytes]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.text_calls.append(list(texts))
        return [list(TEXT_VECTOR) for _ in texts]

    def embed_image_blobs(self, images: list[bytes]) -> list[list[float]]:
        self.image_calls.append(list(images))
        return [list(IMAGE_VECTOR) for _ in images]


class FakeVideoVectorStore:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search_points(self, *, collection_name, vector, limit, query_filter=None):
        self.calls.append({"collection_name": collection_name, "vector": list(vector)})
        return []

    def search_filter(self, video_ids=None, years=None):
        loc = {}
        if video_ids:
            loc["video_ids"] = list(video_ids)
        if years:
            loc["years"] = list(years)
        return loc or None


def build_service() -> tuple[VideoRetrievalService, FakeEmbedder, FakeVideoVectorStore]:
    embedder = FakeEmbedder()
    store = FakeVideoVectorStore()
    retriever = VideoRetriever(
        embedding_service=embedder, vector_store=store, config=stub_config()
    )
    return VideoRetrievalService(retriever=retriever), embedder, store


def vectors_by_collection(store: FakeVideoVectorStore) -> dict[str, list[float]]:
    return {call["collection_name"]: call["vector"] for call in store.calls}


# ── Tầng nhúng ────────────────────────────────────────────────────────────────


def build_embedding_service(captured: dict, **kwargs) -> ImageEmbeddingService:
    def fake_post(url, *, headers, json, timeout):
        captured.update(url=url, json=json)
        return FakeResponse({"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]})

    options = {
        "api_key": "jina-test-key",
        "base_url": "https://api.jina.ai/v1/embeddings",
        "model_name": "jina-clip-v2",
        "dimensions": 3,
        "http_post": fake_post,
        "tokens_per_minute": 0,
    }
    options.update(kwargs)
    return ImageEmbeddingService(**options)


def test_embed_image_blobs_sends_image_input_without_a_task_field() -> None:
    """Ảnh gửi ở khoá `image` và KHÔNG kèm `task`.

    `task=retrieval.query` chỉ dành cho đầu vào text; gắn nó vào một request
    ảnh là gửi thứ jina-clip-v2 không nhận.
    """
    captured: dict = {}
    service = build_embedding_service(captured)

    vectors = service.embed_image_blobs([png_bytes()])

    assert vectors == [[0.1, 0.2, 0.3]]
    assert list(captured["json"]["input"][0]) == ["image"]
    assert "task" not in captured["json"]


def test_embed_image_blobs_downscales_query_image_like_keyframes() -> None:
    """Ảnh người dùng gửi thường 3000px; hạn mức token là thật, không miễn trừ."""
    from PIL import Image

    captured: dict = {}
    service = build_embedding_service(captured, max_image_side=512)

    service.embed_image_blobs([png_bytes((1600, 1200))])

    sent = base64.b64decode(captured["json"]["input"][0]["image"])
    with Image.open(io.BytesIO(sent)) as image:
        assert max(image.size) == 512


def test_embed_image_blobs_rejects_unreadable_image_before_calling_provider() -> None:
    captured: dict = {}
    service = build_embedding_service(captured)

    with pytest.raises(ValueError, match="Không đọc được ảnh truy vấn"):
        service.embed_image_blobs([b"day-khong-phai-anh"])

    # Không được tốn một lượt gọi (và một khoản quota) cho ảnh chắc chắn hỏng.
    assert captured == {}


def test_embed_image_blobs_rejects_empty_input() -> None:
    service = build_embedding_service({})

    with pytest.raises(ValueError, match="Ảnh truy vấn rỗng"):
        service.embed_image_blobs([b""])


# ── Tầng truy hồi ─────────────────────────────────────────────────────────────


def test_image_query_searches_media_clip_with_the_image_vector() -> None:
    """US-303.1 AC-1: so ảnh với ảnh, không tả ảnh thành chữ rồi mới tìm."""
    service, embedder, store = build_service()

    result = service.retrieve(image=png_bytes(), source="clip")

    assert vectors_by_collection(store)[MEDIA_CLIP] == IMAGE_VECTOR
    assert embedder.text_calls == []
    assert result["query_kind"] == "image"
    assert result["query"] == ""


def test_image_only_query_skips_transcript_branch_and_says_why() -> None:
    """Bỏ qua vì định tuyến thì phải nói ra, không được im lặng trả 0 video."""
    service, _, store = build_service()

    result = service.retrieve(image=png_bytes(), source="both")

    assert list(vectors_by_collection(store)) == [MEDIA_CLIP]
    assert result["videos"] == []
    assert any("video_transcript" in note for note in result["notes"])
    assert result["errors"] == []


def test_image_plus_text_gives_each_branch_the_vector_it_can_use() -> None:
    """Ảnh lo nhánh hình, chữ lo nhánh lời thoại — không nhánh nào bị bỏ."""
    service, embedder, store = build_service()

    result = service.retrieve(query="buổi training Python", image=png_bytes(), source="both")

    vectors = vectors_by_collection(store)
    assert vectors[MEDIA_CLIP] == IMAGE_VECTOR
    assert vectors[VIDEO_TRANSCRIPT] == TEXT_VECTOR
    assert embedder.text_calls == [["buổi training Python"]]
    assert result["query_kind"] == "image+text"
    assert result["notes"] == []


def test_text_only_query_still_embeds_exactly_once() -> None:
    """Chống thoái lui: thêm đường ảnh không được phá tính chất một-lần-nhúng."""
    service, embedder, store = build_service()

    result = service.retrieve(query="ảnh buổi training", source="both")

    assert len(embedder.text_calls) == 1
    assert embedder.image_calls == []
    assert set(vectors_by_collection(store)) == {MEDIA_CLIP, VIDEO_TRANSCRIPT}
    assert result["query_kind"] == "text"


def test_query_without_text_and_without_image_is_rejected() -> None:
    service, _, _ = build_service()

    with pytest.raises(ValueError, match="query"):
        service.retrieve(query="   ")


def test_unreadable_image_surfaces_as_value_error_not_empty_results() -> None:
    """US-303.1 edge: ảnh decode fail → lỗi rõ, không trả 'không tìm thấy'."""
    embedder = FakeEmbedder()
    store = FakeVideoVectorStore()

    def reject(images):
        raise ValueError("Không đọc được ảnh truy vấn (query-image-0).")

    embedder.embed_image_blobs = reject  # type: ignore[method-assign]
    service = VideoRetrievalService(
        retriever=VideoRetriever(
            embedding_service=embedder, vector_store=store, config=stub_config()
        )
    )

    with pytest.raises(ValueError, match="Không đọc được ảnh truy vấn"):
        service.retrieve(image=b"hong", source="clip")

    assert store.calls == []
