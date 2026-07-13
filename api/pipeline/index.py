"""Nạp embedding vào Qdrant (OFFLINE). [BR-205/206 · US-205.1/206.1 · T-14]

Input : embedding + payload (video_id, timestamp, ...) + tên collection + dim
Output: dữ liệu vào đúng 1 trong 3 collection (media/transcript/nội quy)
Gợi ý : shared.vectorstore.qdrant.QdrantStore (ensure_collection → upsert)
Edge  : sai chiều vector → TỪ CHỐI (ValueError); upsert trùng id → ghi đè
Pass  : tests/test_pipeline.py::test_index_upsert
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def build_index(collection: str, dim: int, ids: list[int | str],
                vectors: list[list[float]], payloads: list[dict[str, Any]], *,
                store: object | None = None) -> None:
    if not (len(ids) == len(vectors) == len(payloads)):
        raise ValueError(
            f"Lệch số lượng: ids={len(ids)} vectors={len(vectors)} payloads={len(payloads)}",
        )
    for i, v in enumerate(vectors):
        if len(v) != dim:
            raise ValueError(
                f"Vector #{i} sai chiều: {len(v)} ≠ {dim} (collection {collection}).",
            )
    if not ids:
        log.info("build_index: không có điểm nào để nạp vào %s", collection)
        return

    if store is None:
        from app.config import settings
        from shared.vectorstore.qdrant import QdrantStore

        store = QdrantStore(settings.qdrant_url, collection, dim)

    store.ensure_collection()
    store.upsert(ids, vectors, payloads)
    log.info("Đã nạp %d điểm vào collection %s", len(ids), collection)
