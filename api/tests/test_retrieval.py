"""Test retrieval — TC-301/306/307/308. Fake embedder + fake store (không cần Qdrant)."""
from __future__ import annotations

from types import SimpleNamespace

from app.domains.retrieval.service import (
    rank,
    retrieve_by_transcript,
    retrieve_media,
    retrieve_regulations,
)


class FakeTextEmbedder:
    dim = 4

    def embed(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def _store_with(hits):
    class _S:
        def __init__(self):
            self.last = None

        def search(self, vector, top_k=5, query_filter=None):
            self.last = {"vector": vector, "top_k": top_k}
            return hits[:top_k]

    return _S()


class FakeImageEmbedder:
    dim = 4

    def embed(self, image_paths):
        return [[0.0] * self.dim for _ in image_paths]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0, 0.0]


class FakeStore:
    """Ghi lại tham số search, trả điểm giảm dần như Qdrant thật."""

    def __init__(self):
        self.last = None

    def search(self, vector, top_k=5, query_filter=None):
        self.last = {"vector": vector, "top_k": top_k}
        hits = [
            SimpleNamespace(id=1, score=0.91, payload={"video_id": 7, "timestamp": 12, "caption": "khai giảng"}),
            SimpleNamespace(id=2, score=0.40, payload={"video_id": 9, "timestamp": 3, "caption": "workshop"}),
        ]
        return hits[:top_k]


def test_retrieve_media_maps_hits_and_sorted():
    store = FakeStore()
    out = retrieve_media("ảnh khai giảng", top_k=5,
                         embedder=FakeImageEmbedder(), store=store)
    assert [o["id"] for o in out] == [1, 2]
    assert out[0]["score"] >= out[1]["score"]        # đã theo score giảm dần
    assert out[0]["caption"] == "khai giảng"          # payload được trải phẳng
    assert store.last["vector"] == [1.0, 0.0, 0.0, 0.0]  # dùng embed_query


def test_retrieve_media_respects_top_k():
    out = retrieve_media("x", top_k=1, embedder=FakeImageEmbedder(), store=FakeStore())
    assert len(out) == 1


# ---- T-31 nội quy ----
def test_retrieve_regulations_maps_article_clause():
    store = _store_with([
        SimpleNamespace(id=1, score=0.8, payload={"article": 5, "clause": 2, "text": "cấm hút thuốc"}),
    ])
    out = retrieve_regulations("được hút thuốc không", embedder=FakeTextEmbedder(), store=store)
    assert out[0]["article"] == 5 and out[0]["clause"] == 2
    assert store.last["vector"] == [1.0, 0.0, 0.0, 0.0]   # dùng embed() của text embedder


# ---- T-32 transcript: gộp theo video_id (TC-308 không nhân đôi video) ----
def test_retrieve_by_transcript_merges_same_video():
    store = _store_with([
        SimpleNamespace(id=1, score=0.9, payload={"video_id": 7, "start_sec": 3, "text": "chào mừng"}),
        SimpleNamespace(id=2, score=0.7, payload={"video_id": 7, "start_sec": 40, "text": "kết thúc"}),
        SimpleNamespace(id=3, score=0.5, payload={"video_id": 9, "start_sec": 1, "text": "khác"}),
    ])
    out = retrieve_by_transcript("khai mạc", top_k=5, embedder=FakeTextEmbedder(), store=store)
    assert len(out) == 2                                  # video 7 gộp 1 lần, không nhân đôi
    top = out[0]
    assert top["video_id"] == 7 and top["score"] == 0.9   # giữ score cao nhất
    assert len(top["moments"]) == 2                        # gom 2 mốc thời gian


# ---- T-33 rank + ngưỡng ----
def test_rank_sorts_desc_and_applies_threshold():
    cands = [{"id": 1, "score": 0.2}, {"id": 2, "score": 0.9}, {"id": 3, "score": 0.5}]
    assert [c["id"] for c in rank(cands)] == [2, 3, 1]
    assert [c["id"] for c in rank(cands, threshold=0.55)] == [2]
    assert rank(cands, threshold=0.95) == []              # tất cả dưới ngưỡng → "không tìm thấy"
