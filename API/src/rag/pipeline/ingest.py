from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any

from src.common_utils.time_utils import get_system_time
from src.log.logger import logger
from src.rag.embedding.embedding_service import EmbeddingService
from src.rag.vector_store.vector_store import QdrantVectorStore


class IngestService:
    """Embed chunks and upsert them into the vector store."""

    def __init__(self, embedding_service: EmbeddingService, vector_store: QdrantVectorStore) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def ingest_chunks(self, chunks: list[dict[str, Any]]) -> dict:
        validated_chunks = self._validate_chunks(chunks)
        texts = [chunk["text"] for chunk in validated_chunks]
        embeddings = self.embedding_service.embed_documents(texts)
        self._validate_embeddings(embeddings, expected_count=len(validated_chunks))

        vector_size = len(embeddings[0])
        self.vector_store.ensure_collection(vector_size)

        enriched_chunks = self._enrich_chunks(validated_chunks)
        result = self.vector_store.upsert_chunks(enriched_chunks, embeddings)
        return {**result, "embedded": len(embeddings)}

    @staticmethod
    def _validate_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(chunks, list):
            raise TypeError("chunks must be a list[dict]")
        if not chunks:
            raise ValueError("chunks must not be empty")

        seen_chunk_ids: set[str] = set()
        validated: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise TypeError(f"chunks[{index}] must be a dict")
            chunk_id = str(chunk.get("chunk_id", "")).strip()
            document_id = str(chunk.get("document_id", "")).strip()
            text = str(chunk.get("text", "")).strip()
            if not chunk_id:
                raise ValueError(f"chunks[{index}] missing chunk_id")
            if chunk_id in seen_chunk_ids:
                raise ValueError(f"duplicate chunk_id detected: {chunk_id}")
            if not document_id:
                raise ValueError(f"chunks[{index}] missing document_id")
            if not text:
                raise ValueError(f"chunks[{index}] text must not be empty")
            seen_chunk_ids.add(chunk_id)
            copied = deepcopy(chunk)
            copied["text"] = text
            validated.append(copied)
        return validated

    @staticmethod
    def _validate_embeddings(embeddings: list[list[float]], expected_count: int) -> None:
        if len(embeddings) != expected_count:
            raise ValueError(f"embedding count mismatch: expected {expected_count}, got {len(embeddings)}")
        dimension: int | None = None
        for index, embedding in enumerate(embeddings):
            if not embedding:
                raise ValueError(f"embeddings[{index}] must not be empty")
            if dimension is None:
                dimension = len(embedding)
            elif len(embedding) != dimension:
                raise ValueError("embedding dimensions are not consistent")

    @staticmethod
    def _enrich_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        created_at = get_system_time()
        enriched: list[dict[str, Any]] = []
        for chunk in chunks:
            copied = deepcopy(chunk)
            copied["content_hash"] = hashlib.sha256(copied["text"].encode("utf-8")).hexdigest()
            copied["created_at"] = copied.get("created_at") or created_at
            enriched.append(copied)
        logger.info(f"Prepared {len(enriched)} chunk payload(s) for vector upsert")
        return enriched
