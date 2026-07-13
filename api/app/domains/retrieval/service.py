"""Truy xuất multimodal (LÕI HỌC). [BR-301/302/303/306/307/308]

  retrieve_media        : text/ảnh → top-k media (embed query → search Qdrant media)   ✅ T-30
  retrieve_regulations  : text → top-k rule_chunk (search collection nội quy)           ✅ T-31
  retrieve_by_transcript: text → đoạn video khớp lời nói, gộp keyframe theo video_id     ✅ T-32
  rank                  : xếp hạng giảm dần; điểm < ngưỡng → loại (rỗng = "không tìm thấy") ✅ T-33
  Gợi ý : providers.embeddings + shared.vectorstore.qdrant.QdrantStore
  Pass  : tests/test_retrieval.py

`embedder`/`store` nhận qua tham số (mặc định lấy từ app.deps) → test inject fake.
"""
from __future__ import annotations

from typing import Any


def _hits_to_items(hits: list[Any]) -> list[dict[str, Any]]:
    """ScoredPoint Qdrant → dict phẳng {id, score, ...payload}."""
    return [{"id": h.id, "score": h.score, **(h.payload or {})} for h in hits]


def retrieve_media(query: str, top_k: int = 5, *,
                   embedder: Any | None = None, store: Any | None = None) -> list[dict[str, Any]]:
    """US-301.1: embed câu hỏi (Jina-CLIP text tower) → search Qdrant media.

    Trả list đã sắp theo score giảm dần (Qdrant lo), mỗi item = payload + id + score.
    """
    if embedder is None or store is None:
        from app.deps import get_image_embedder, get_media_store

        embedder = embedder or get_image_embedder()
        store = store or get_media_store()

    vector = embedder.embed_query(query)
    return _hits_to_items(store.search(vector, top_k=top_k))


def retrieve_regulations(query: str, top_k: int = 5, *,
                         embedder: Any | None = None, store: Any | None = None) -> list[dict[str, Any]]:
    """US-307.1: nhúng câu hỏi (Vietnamese_Embedding) → search collection nội quy."""
    if embedder is None or store is None:
        from app.deps import get_regulation_store, get_text_embedder

        embedder = embedder or get_text_embedder()
        store = store or get_regulation_store()

    vector = embedder.embed([query])[0]
    return _hits_to_items(store.search(vector, top_k=top_k))


def retrieve_by_transcript(query: str, top_k: int = 5, *,
                           embedder: Any | None = None, store: Any | None = None) -> list[dict[str, Any]]:
    """US-308.1: khớp transcript → GỘP theo video_id (không nhân đôi video).

    Mỗi video ra 1 item: giữ score cao nhất + gom các mốc thời gian khớp (`moments`).
    """
    if embedder is None or store is None:
        from app.deps import get_text_embedder, get_transcript_store

        embedder = embedder or get_text_embedder()
        store = store or get_transcript_store()

    vector = embedder.embed([query])[0]
    merged: dict[Any, dict[str, Any]] = {}
    for h in store.search(vector, top_k=top_k):
        p = h.payload or {}
        vid = p.get("video_id", h.id)
        moment = {
            "start_sec": p.get("start_sec", p.get("timestamp")),
            "end_sec": p.get("end_sec"),
            "text": p.get("text"),
        }
        entry = merged.get(vid)
        if entry is None:
            merged[vid] = {"video_id": vid, "score": h.score, "moments": [moment]}
        else:
            entry["score"] = max(entry["score"], h.score)
            entry["moments"].append(moment)
    return sorted(merged.values(), key=lambda x: x["score"], reverse=True)


def rank(candidates: list[dict[str, Any]], threshold: float = 0.0) -> list[dict[str, Any]]:
    """US-306.1: xếp hạng giảm dần + cắt ngưỡng. Rỗng ⇒ tầng answer coi là 'không tìm thấy'."""
    kept = [c for c in candidates if c.get("score", 0.0) >= threshold]
    return sorted(kept, key=lambda c: c.get("score", 0.0), reverse=True)
