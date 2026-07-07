"""Wrapper Qdrant (hạ tầng — CHO SẴN).

Học phần nằm ở *dùng* store này trong retrieval/pipeline (embed gì, xếp hạng
ra sao), không phải ở cú pháp qdrant-client. 3 collection: media/transcript/nội quy.
"""
from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient, models


class QdrantStore:
    def __init__(self, url: str, collection: str, dim: int, api_key: str | None = None) -> None:
        self.client = QdrantClient(url=url, api_key=api_key or None)
        self.collection = collection
        self.dim = dim

    def ensure_collection(self, distance: models.Distance = models.Distance.COSINE) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=self.dim, distance=distance),
            )

    def upsert(self, ids: list[int | str], vectors: list[list[float]],
               payloads: list[dict[str, Any]]) -> None:
        points = [
            models.PointStruct(id=i, vector=v, payload=p)
            for i, v, p in zip(ids, vectors, payloads, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, vector: list[float], top_k: int = 5,
               query_filter: models.Filter | None = None) -> list[models.ScoredPoint]:
        return self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,
            query_filter=query_filter,
        )
