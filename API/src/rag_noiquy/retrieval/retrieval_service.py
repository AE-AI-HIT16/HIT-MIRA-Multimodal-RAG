from __future__ import annotations

from typing import Any

from src.configuration import AppConfig
from src.log.logger import logger
from src.rag_noiquy.embedding.embedding_service import EmbeddingService
from src.rag_noiquy.retrieval.query_rewriter import QueryRewriter
from src.rag_noiquy.retrieval.retriever import VectorRetriever
from src.rag_noiquy.vector_store.vector_store import QdrantVectorStore


class RetrievalService:
    """Coordinate query rewriting and vector retrieval."""

    def __init__(
        self,
        query_rewriter: QueryRewriter,
        retriever: VectorRetriever,
    ) -> None:
        self.query_rewriter = query_rewriter
        self.retriever = retriever

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
        rewrite: bool = True,
    ) -> dict[str, Any]:
        query = self._normalize_query(query)
        search_query = self.query_rewriter.rewrite(query) if rewrite else query
        chunks = self.retriever.retrieve(
            search_query, top_k=top_k, document_ids=document_ids
        )
        logger.info(f"Retrieval flow completed with {len(chunks)} result(s)")
        return {
            "query": query,
            "rewritten_query": search_query,
            "results": [chunk.as_dict() for chunk in chunks],
            "context": self._format_context(chunks),
            "total": len(chunks),
        }

    @staticmethod
    def _format_context(chunks: list[Any]) -> str:
        parts: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            metadata = dict(chunk.document.metadata or {})
            source = metadata.get("filename") or metadata.get("source") or "unknown"
            page = metadata.get("page") or metadata.get("start_page")
            label = f"[{index}] {source}"
            if page:
                label = f"{label}, page {page}"
            section = metadata.get("section")
            if section:
                label = f"{label} | {section}"
            parts.append(f"{label}\n{chunk.document.page_content}")
        return "\n\n".join(parts)

    @staticmethod
    def _normalize_query(query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        return " ".join(query.split())


def build_retrieval_service(config: AppConfig | None = None) -> RetrievalService:
    app_config = config or AppConfig()
    embedding_service = EmbeddingService(config=app_config)
    vector_store = QdrantVectorStore(config=app_config)
    return RetrievalService(
        query_rewriter=QueryRewriter(config=app_config),
        retriever=VectorRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            config=app_config,
        ),
    )
