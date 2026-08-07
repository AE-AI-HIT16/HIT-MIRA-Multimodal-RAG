from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.rag_video_anh.retrieval.retrieval_service import (
    VideoRetrievalService,
    build_video_retrieval_service,
)
from src.rag_video_anh.retrieval.retriever import VideoRetriever
from src.rag_video_anh.vector_store.vector_store import QdrantVideoVectorStore

MEDIA_CLIP = "media_clip"
VIDEO_TRANSCRIPT = "video_transcript"


def stub_config(top_k: int | None = 5) -> SimpleNamespace:
    return SimpleNamespace(
        retrieval=SimpleNamespace(top_k=top_k),
        qdrant=SimpleNamespace(url="http://localhost:6333", api_key=None, distance="COSINE"),
    )


class FakeEmbedder:
    """Đếm số lượt gọi để chứng minh chỉ nhúng câu hỏi một lần mỗi request."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [0.1, 0.2, 0.3]
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [list(self.vector) for _ in texts]


class FakeVideoVectorStore:
    def __init__(self, results: dict[str, list[dict]] | None = None) -> None:
        self.results = results or {}
        self.calls: list[dict] = []
        self.failing_collections: set[str] = set()

    def search_points(self, *, collection_name, vector, limit, query_filter=None):
        self.calls.append(
            {
                "collection_name": collection_name,
                "vector": list(vector),
                "limit": limit,
                "query_filter": query_filter,
            }
        )
        if collection_name in self.failing_collections:
            raise RuntimeError(f"qdrant down for {collection_name}")
        return list(self.results.get(collection_name, []))

    def search_filter(self, video_ids=None, years=None):
        loc = {}
        if video_ids:
            loc["video_ids"] = list(video_ids)
        if years:
            loc["years"] = list(years)
        return loc or None


def point(score: float, payload: dict) -> dict:
    return {"id": payload.get("unit_id", "id"), "score": score, "payload": payload}


def clip_payload(**overrides) -> dict:
    payload = {
        "unit_id": "clip-1",
        "video_id": "video-1",
        "post_id": "post-1",
        "frame_index": 12,
        "timestamp_sec": 75.5,
        "bucket_name": "hit-mira-media",
        "frame_object_key": "frames/video-1/000012.jpg",
        "caption": "Thành viên CLB thuyết trình trên sân khấu",
        "ocr_text": "HIT-MIRA 2026",
        "detected_objects": ["person", "microphone"],
    }
    payload.update(overrides)
    return payload


def transcript_payload(**overrides) -> dict:
    payload = {
        "unit_id": "tr-1",
        "video_id": "video-1",
        "post_id": "post-1",
        "start_sec": 10.0,
        "end_sec": 20.0,
        "text": "Chào mừng các bạn đến với buổi sinh hoạt CLB",
        "language": "vi",
    }
    payload.update(overrides)
    return payload


def build_retriever(store: FakeVideoVectorStore, embedder: FakeEmbedder) -> VideoRetriever:
    return VideoRetriever(
        embedding_service=embedder,
        vector_store=store,
        config=stub_config(),
    )


# ---------- VideoRetriever ----------


def test_embed_query_normalizes_whitespace_and_embeds_once():
    embedder = FakeEmbedder()
    retriever = build_retriever(FakeVideoVectorStore(), embedder)

    vector = retriever.embed_query("  buổi   sinh hoạt \n CLB ")

    assert vector == [0.1, 0.2, 0.3]
    assert embedder.calls == [["buổi sinh hoạt CLB"]]


def test_embed_query_rejects_empty_query():
    retriever = build_retriever(FakeVideoVectorStore(), FakeEmbedder())

    with pytest.raises(ValueError):
        retriever.embed_query("   ")


def test_retrieve_clips_maps_payload_to_citation_fields():
    store = FakeVideoVectorStore({MEDIA_CLIP: [point(0.82, clip_payload())]})
    retriever = build_retriever(store, FakeEmbedder())

    hits = retriever.retrieve_clips("thuyết trình")

    assert len(hits) == 1
    hit = hits[0].as_dict()
    assert hit["video_id"] == "video-1"
    assert hit["timestamp_sec"] == 75.5
    assert hit["caption"] == "Thành viên CLB thuyết trình trên sân khấu"
    assert hit["ocr_text"] == "HIT-MIRA 2026"
    assert hit["frame_object_key"] == "frames/video-1/000012.jpg"
    assert hit["detected_objects"] == ["person", "microphone"]
    assert hit["score"] == pytest.approx(0.82)


def test_retrieve_clips_returns_empty_when_nothing_indexed():
    retriever = build_retriever(FakeVideoVectorStore(), FakeEmbedder())

    assert retriever.retrieve_clips("không có gì") == []


def test_retrieve_by_transcript_merges_segments_of_same_video():
    """Nhiều đoạn của cùng một video phải gộp thành 1 kết quả, lấy điểm cao nhất."""
    store = FakeVideoVectorStore(
        {
            VIDEO_TRANSCRIPT: [
                point(0.90, transcript_payload(unit_id="tr-1", start_sec=10.0, end_sec=20.0)),
                point(0.70, transcript_payload(unit_id="tr-2", start_sec=40.0, end_sec=50.0)),
                point(0.60, transcript_payload(unit_id="tr-3", video_id="video-2")),
            ]
        }
    )
    retriever = build_retriever(store, FakeEmbedder())

    hits = retriever.retrieve_by_transcript("sinh hoạt")

    assert [hit.video_id for hit in hits] == ["video-1", "video-2"]
    first = hits[0]
    assert first.score == pytest.approx(0.90), "điểm video = điểm đoạn cao nhất"
    assert [moment.unit_id for moment in first.moments] == ["tr-1", "tr-2"]
    assert [moment.score for moment in first.moments] == [0.90, 0.70]
    assert first.language == "vi"


def test_retrieve_by_transcript_does_not_double_count_video_in_top_k():
    store = FakeVideoVectorStore(
        {
            VIDEO_TRANSCRIPT: [
                point(0.9, transcript_payload(unit_id="a")),
                point(0.8, transcript_payload(unit_id="b")),
                point(0.7, transcript_payload(unit_id="c")),
            ]
        }
    )
    retriever = build_retriever(store, FakeEmbedder())

    hits = retriever.retrieve_by_transcript("sinh hoạt", top_k=2)

    assert len(hits) == 1
    assert len(hits[0].moments) == 3


def test_retrieve_by_transcript_keeps_segments_without_video_id_separate():
    store = FakeVideoVectorStore(
        {
            VIDEO_TRANSCRIPT: [
                point(0.9, transcript_payload(unit_id="x", video_id=None)),
                point(0.8, transcript_payload(unit_id="y", video_id=None)),
            ]
        }
    )
    retriever = build_retriever(store, FakeEmbedder())

    hits = retriever.retrieve_by_transcript("sinh hoạt")

    assert len(hits) == 2, "thiếu video_id thì không được gộp lẫn vào nhau"
    assert all(hit.video_id is None for hit in hits)


def test_retrieve_by_transcript_overfetches_candidates_before_merging():
    store = FakeVideoVectorStore()
    retriever = build_retriever(store, FakeEmbedder())

    retriever.retrieve_by_transcript("sinh hoạt", top_k=5)

    assert store.calls[0]["limit"] == 15, "phải lấy dư ứng viên vì gộp sẽ làm giảm số kết quả"


def test_retriever_passes_video_id_filter_to_store():
    store = FakeVideoVectorStore()
    retriever = build_retriever(store, FakeEmbedder())

    retriever.retrieve_clips("query", video_ids=["video-9"])

    assert store.calls[0]["query_filter"] == {"video_ids": ["video-9"]}


def test_retriever_clamps_top_k_to_max():
    store = FakeVideoVectorStore()
    retriever = build_retriever(store, FakeEmbedder())

    retriever.retrieve_clips("query", top_k=999)

    assert store.calls[0]["limit"] == 50


# ---------- QdrantVideoVectorStore.search_points ----------


class FakeQdrantClient:
    """Client kiểu qdrant-client >= 1.10 (có query_points)."""

    def __init__(self, points: list, exists: bool = True) -> None:
        self.points = points
        self.exists = exists
        self.query_calls: list[dict] = []

    def collection_exists(self, collection_name: str) -> bool:
        return self.exists

    def query_points(self, **kwargs):
        self.query_calls.append(kwargs)
        return SimpleNamespace(points=self.points)


class LegacyQdrantClient:
    """Client bản cũ: chỉ có search(), không có query_points."""

    def __init__(self, points: list) -> None:
        self.points = points
        self.search_calls: list[dict] = []

    def collection_exists(self, collection_name: str) -> bool:
        return True

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return self.points


def build_store(client) -> QdrantVideoVectorStore:
    return QdrantVideoVectorStore(client=client, config=stub_config())


def test_search_points_normalizes_scored_points():
    client = FakeQdrantClient(
        [SimpleNamespace(id="p1", score=0.75, payload={"video_id": "video-1"})]
    )
    store = build_store(client)

    results = store.search_points(collection_name=MEDIA_CLIP, vector=[0.1, 0.2], limit=3)

    assert results == [{"id": "p1", "score": 0.75, "payload": {"video_id": "video-1"}}]
    assert client.query_calls[0]["limit"] == 3
    assert client.query_calls[0]["with_payload"] is True


def test_search_points_returns_empty_when_collection_missing():
    """Video chưa index là trạng thái hợp lệ -> [] chứ không nổ lỗi, không bịa."""
    store = build_store(FakeQdrantClient([], exists=False))

    assert store.search_points(collection_name=MEDIA_CLIP, vector=[0.1], limit=5) == []


def test_search_points_falls_back_to_legacy_search_api():
    client = LegacyQdrantClient([SimpleNamespace(id="p9", score=0.5, payload=None)])
    store = build_store(client)

    results = store.search_points(collection_name=VIDEO_TRANSCRIPT, vector=[0.3], limit=2)

    assert results == [{"id": "p9", "score": 0.5, "payload": {}}]
    assert client.search_calls[0]["query_vector"] == [0.3]


def test_search_points_rejects_invalid_inputs():
    store = build_store(FakeQdrantClient([]))

    with pytest.raises(ValueError):
        store.search_points(collection_name=MEDIA_CLIP, vector=[], limit=5)
    with pytest.raises(ValueError):
        store.search_points(collection_name=MEDIA_CLIP, vector=[0.1], limit=0)


def test_search_filter_builds_qdrant_filter():
    store = build_store(FakeQdrantClient([]))

    assert store.search_filter(None) is None
    assert store.search_filter(["  "]) is None
    built = store.search_filter(["video-1", "video-2"])
    assert isinstance(built, store.models.Filter)
    assert len(built.should) == 2


def test_retriever_passes_video_id_and_year_filter_to_store():
    store = FakeVideoVectorStore()
    retriever = build_retriever(store, FakeEmbedder())

    retriever.retrieve_clips("query", video_ids=["video-9"], years=[2024])

    assert store.calls[0]["query_filter"] == {"video_ids": ["video-9"], "years": [2024]}


# ---------- VideoRetrievalService ----------


def build_service(store: FakeVideoVectorStore, embedder: FakeEmbedder) -> VideoRetrievalService:
    return VideoRetrievalService(retriever=build_retriever(store, embedder))


def test_service_embeds_query_once_for_both_collections():
    """Jina-CLIP v2 dùng chung không gian vector -> 1 lượt nhúng cho 2 collection."""
    store = FakeVideoVectorStore(
        {
            MEDIA_CLIP: [point(0.8, clip_payload())],
            VIDEO_TRANSCRIPT: [point(0.7, transcript_payload())],
        }
    )
    embedder = FakeEmbedder()
    service = build_service(store, embedder)

    result = service.retrieve("buổi sinh hoạt CLB")

    assert len(embedder.calls) == 1, "không được nhúng lại câu hỏi cho collection thứ hai"
    assert [call["collection_name"] for call in store.calls] == [MEDIA_CLIP, VIDEO_TRANSCRIPT]
    assert store.calls[0]["vector"] == store.calls[1]["vector"]
    assert result["total"] == 2
    assert result["found"] is True
    assert result["errors"] == []


def test_service_returns_not_found_on_empty_retrieval():
    service = build_service(FakeVideoVectorStore(), FakeEmbedder())

    result = service.retrieve("nội dung không tồn tại")

    assert result["found"] is False
    assert result["total"] == 0
    assert result["clips"] == []
    assert result["videos"] == []
    assert result["context"] == ""


def test_service_source_clip_skips_transcript_collection():
    store = FakeVideoVectorStore({MEDIA_CLIP: [point(0.8, clip_payload())]})
    service = build_service(store, FakeEmbedder())

    result = service.retrieve("thuyết trình", source="clip")

    assert [call["collection_name"] for call in store.calls] == [MEDIA_CLIP]
    assert result["source"] == "clip"
    assert result["videos"] == []


def test_service_source_transcript_skips_clip_collection():
    store = FakeVideoVectorStore({VIDEO_TRANSCRIPT: [point(0.7, transcript_payload())]})
    service = build_service(store, FakeEmbedder())

    result = service.retrieve("lời thoại", source="transcript")

    assert [call["collection_name"] for call in store.calls] == [VIDEO_TRANSCRIPT]
    assert result["clips"] == []


def test_service_keeps_transcript_results_when_clip_branch_fails():
    """Hai nhánh độc lập: keyframe lỗi vẫn phải trả về được transcript."""
    store = FakeVideoVectorStore({VIDEO_TRANSCRIPT: [point(0.7, transcript_payload())]})
    store.failing_collections.add(MEDIA_CLIP)
    service = build_service(store, FakeEmbedder())

    result = service.retrieve("sinh hoạt")

    assert result["clips"] == []
    assert len(result["videos"]) == 1
    assert result["found"] is True
    assert result["errors"] == ["media_clip: RuntimeError"]


def test_service_rejects_invalid_source():
    service = build_service(FakeVideoVectorStore(), FakeEmbedder())

    with pytest.raises(ValueError):
        service.retrieve("query", source="video")


def test_service_context_carries_timestamp_citations():
    store = FakeVideoVectorStore(
        {
            MEDIA_CLIP: [point(0.8, clip_payload(timestamp_sec=3675))],
            VIDEO_TRANSCRIPT: [point(0.7, transcript_payload(start_sec=10, end_sec=20))],
        }
    )
    service = build_service(store, FakeEmbedder())

    context = service.retrieve("sinh hoạt")["context"]

    assert "[1] video video-1 tại 01:01:15" in context
    assert "Thành viên CLB thuyết trình trên sân khấu" in context
    assert "Chữ trong hình: HIT-MIRA 2026" in context
    assert "[2] video video-1 (lời thoại)" in context
    assert "[00:10-00:20] Chào mừng các bạn đến với buổi sinh hoạt CLB" in context


def test_build_video_retrieval_service_is_exported():
    assert callable(build_video_retrieval_service)
