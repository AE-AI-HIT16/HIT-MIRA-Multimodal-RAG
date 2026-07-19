from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.configuration import AppConfig
from src.log.logger import logger
from src.rag.embedding.embedding_service import EmbeddingService
from src.rag.vector_store.vector_store import QdrantVectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    document: Document
    score: float

    def as_dict(self) -> dict[str, Any]:
        metadata = self.document.metadata
        return {
            "score": self.score,
            "text": self.document.page_content,
            "metadata": metadata,
            "document_id": metadata.get("document_id"),
            "chunk_id": metadata.get("chunk_id"),
            "chunk_index": metadata.get("chunk_index"),
            "filename": metadata.get("filename") or metadata.get("source"),
            "source": metadata.get("source"),
            "page": metadata.get("page") or metadata.get("start_page"),
            "start_page": metadata.get("start_page"),
            "end_page": metadata.get("end_page"),
            "section": metadata.get("section"),
        }


class LangChainEmbeddingAdapter(Embeddings):
    """Expose the existing embedding service through LangChain's Embeddings API."""

    def __init__(self, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedding_service.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embedding_service.embed_query(text)


class VectorRetriever:
    """Retrieve, rerank, and deduplicate semantic document chunks."""

    CANDIDATE_MULTIPLIER = 3
    MAX_CANDIDATES = 100
    KEYWORD_BOOST_PER_MATCH = 0.02
    MAX_KEYWORD_BOOST = 0.08
    STOP_WORDS = frozenset(
        {
            "ai",
            "các",
            "có",
            "cho",
            "của",
            "được",
            "khi",
            "là",
            "một",
            "những",
            "và",
            "về",
            "với",
        }
    )

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
        top_k: int | None = None,
        score_threshold: float | None = None,
        config: AppConfig | None = None,
    ) -> None:
        retrieval_config = (config or AppConfig()).retrieval
        self.embeddings = LangChainEmbeddingAdapter(embedding_service)
        self.vector_store = vector_store
        self.top_k = int(top_k or retrieval_config.top_k or 5)
        # KEYWORD_THRESHOLD is a legacy keyword-search setting, not a calibrated
        # cosine-similarity threshold. Filter only when explicitly requested.
        self.score_threshold = score_threshold
        self._store = None

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        normalized_query = self._normalize_query(query)
        requested_k = int(top_k or self.top_k)
        candidate_k = min(
            max(requested_k * self.CANDIDATE_MULTIPLIER, requested_k),
            self.MAX_CANDIDATES,
        )
        docs_with_scores = self._langchain_store().similarity_search_with_score(
            query=normalized_query,
            k=candidate_k,
            filter=self._metadata_filter(document_ids),
        )
        candidates = [
            RetrievedChunk(
                document=self._normalize_document(document), score=float(score)
            )
            for document, score in docs_with_scores
            if self.score_threshold is None
            or float(score) >= float(self.score_threshold)
        ]
        chunks = self._rerank_and_deduplicate(normalized_query, candidates, requested_k)
        logger.info(
            f"Retrieved {len(chunks)} chunk(s) from {len(candidates)} candidate(s)"
        )
        return chunks

    def _rerank_and_deduplicate(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Prefer exact query terms and discard duplicate overlap text."""
        query_terms = self._terms(query)
        seen_content: set[str] = set()
        ranked: list[tuple[float, int, RetrievedChunk]] = []

        for position, chunk in enumerate(candidates):
            normalized_content = self._normalise_content(chunk.document.page_content)
            if not normalized_content or normalized_content in seen_content:
                continue
            seen_content.add(normalized_content)
            keyword_matches = len(
                query_terms.intersection(self._terms(normalized_content))
            )
            keyword_boost = min(
                keyword_matches * self.KEYWORD_BOOST_PER_MATCH,
                self.MAX_KEYWORD_BOOST,
            )
            ranked.append((chunk.score + keyword_boost, -position, chunk))

        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [chunk for _, _, chunk in ranked[:top_k]]

    @classmethod
    def _terms(cls, value: str) -> set[str]:
        return {
            token
            for token in re.findall(r"\w+", value.casefold(), flags=re.UNICODE)
            if len(token) > 1 and token not in cls.STOP_WORDS
        }

    @staticmethod
    def _normalise_content(value: str) -> str:
        return " ".join(value.casefold().split())

    def _langchain_store(self):
        if self._store is not None:
            return self._store
        try:
            from langchain_qdrant import QdrantVectorStore as LangChainQdrantVectorStore
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency 'langchain-qdrant'. Install it to use retrieval."
            ) from exc

        self._store = LangChainQdrantVectorStore(
            client=self.vector_store.client,
            collection_name=self.vector_store.collection_name,
            embedding=self.embeddings,
            content_payload_key="text",
            metadata_payload_key="metadata",
        )
        return self._store

    def _metadata_filter(self, document_ids: list[str] | None) -> Any | None:
        document_ids = [
            item.strip() for item in document_ids or [] if item and item.strip()
        ]
        if not document_ids:
            return None

        models = self.vector_store.models
        conditions = [
            models.FieldCondition(
                key="document_id", match=models.MatchValue(value=document_id)
            )
            for document_id in document_ids
        ]
        return (
            models.Filter(must=conditions)
            if len(conditions) == 1
            else models.Filter(should=conditions)
        )

    @staticmethod
    def _normalize_document(document: Document) -> Document:
        metadata = dict(document.metadata or {})
        nested_metadata = metadata.pop("metadata", None)
        if isinstance(nested_metadata, dict):
            metadata = {**nested_metadata, **metadata}
        return Document(page_content=document.page_content, metadata=metadata)

    @staticmethod
    def _normalize_query(query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        return " ".join(query.split())
