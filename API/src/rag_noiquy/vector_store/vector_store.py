from __future__ import annotations

import uuid
from typing import Any

from src.configuration import AppConfig
from src.log.logger import logger


class VectorStoreConfigurationError(ValueError):
    """Raised when vector store configuration is incomplete or invalid."""


class VectorStoreError(RuntimeError):
    """Raised when Qdrant rejects an operation or returns invalid metadata."""


class QdrantVectorStore:
    """Qdrant vector store for regulation chunks."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str | None = None,
        distance: str | None = None,
        client: Any | None = None,
        config: AppConfig | None = None,
    ) -> None:
        app_config = config or AppConfig()
        qdrant_config = app_config.qdrant

        self.url = url if url is not None else qdrant_config.url
        self.api_key = api_key if api_key is not None else qdrant_config.api_key
        self.collection_name = (
            collection_name
            if collection_name is not None
            else qdrant_config.collection_name
        )
        self.distance_name = (
            distance or getattr(qdrant_config, "distance", None) or "COSINE"
        ).upper()

        if self._is_missing(self.collection_name):
            raise VectorStoreConfigurationError(
                "Missing Qdrant collection name. Set QDRANT_COLLECTION_NAME."
            )

        self.client = client or self._create_client()
        self.models = self._load_models()

    def ensure_collection(self, vector_size: int) -> None:
        if not isinstance(vector_size, int) or vector_size <= 0:
            raise ValueError("vector_size must be a positive integer")

        if not self._collection_exists():
            distance = getattr(self.models.Distance, self.distance_name, None)
            if distance is None:
                raise VectorStoreConfigurationError(
                    f"Unsupported Qdrant distance: {self.distance_name}"
                )
            logger.info(
                f"Creating Qdrant collection '{self.collection_name}' with vector size {vector_size}"
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.models.VectorParams(
                    size=vector_size, distance=distance
                ),
            )
            return

        existing_size = self._collection_vector_size()
        if existing_size is not None and existing_size != vector_size:
            raise VectorStoreConfigurationError(
                "Qdrant collection vector dimension mismatch: "
                f"collection '{self.collection_name}' has {existing_size}, requested {vector_size}."
            )

    def upsert_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> dict:
        self._validate_upsert_inputs(chunks, embeddings)
        vector_size = len(embeddings[0])
        self.ensure_collection(vector_size)

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point_id = self._point_id(chunk)
            payload = self._payload(chunk)
            points.append(
                self.models.PointStruct(id=point_id, vector=embedding, payload=payload)
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(
            f"Upserted {len(points)} chunk point(s) into Qdrant collection '{self.collection_name}'"
        )
        return {"collection_name": self.collection_name, "upserted": len(points)}

    def delete_document(self, document_id: str) -> None:
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        filter_condition = self.models.Filter(
            must=[
                self.models.FieldCondition(
                    key="document_id",
                    match=self.models.MatchValue(value=document_id),
                )
            ]
        )
        points_selector = self.models.FilterSelector(filter=filter_condition)
        self.client.delete(
            collection_name=self.collection_name, points_selector=points_selector
        )
        logger.info(f"Deleted Qdrant points for document_id '{document_id}'")

    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as exc:  # Qdrant client exposes several transport exceptions.
            logger.warning(f"Qdrant health check failed: {exc.__class__.__name__}")
            return False

    def _create_client(self):
        if self._is_missing(self.url):
            raise VectorStoreConfigurationError("Missing Qdrant URL. Set QDRANT_URL.")
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise VectorStoreConfigurationError(
                "Missing dependency 'qdrant-client'."
            ) from exc

        kwargs: dict[str, Any] = {"url": self.url}
        if not self._is_missing(self.api_key):
            kwargs["api_key"] = self.api_key
        return QdrantClient(**kwargs)

    @staticmethod
    def _load_models():
        try:
            from qdrant_client import models
        except ImportError as exc:
            raise VectorStoreConfigurationError(
                "Missing dependency 'qdrant-client'."
            ) from exc
        return models

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        if not isinstance(value, str):
            return False
        stripped = value.strip()
        return (
            not stripped
            or (stripped.startswith("${") and stripped.endswith("}"))
            or (stripped.startswith("your_") and stripped.endswith("_here"))
        )

    def _collection_exists(self) -> bool:
        if hasattr(self.client, "collection_exists"):
            return bool(
                self.client.collection_exists(collection_name=self.collection_name)
            )
        try:
            self.client.get_collection(collection_name=self.collection_name)
            return True
        except Exception:
            return False

    def _collection_vector_size(self) -> int | None:
        collection = self.client.get_collection(collection_name=self.collection_name)
        vectors = getattr(getattr(collection, "config", None), "params", None)
        vectors = getattr(vectors, "vectors", None)
        if vectors is None:
            return None
        if hasattr(vectors, "size"):
            return int(vectors.size)
        if isinstance(vectors, dict):
            sizes = [getattr(value, "size", None) for value in vectors.values()]
            sizes = [int(size) for size in sizes if size is not None]
            if len(set(sizes)) == 1:
                return sizes[0]
        return None

    @staticmethod
    def _validate_upsert_inputs(
        chunks: list[dict], embeddings: list[list[float]]
    ) -> None:
        if not chunks:
            raise ValueError("chunks must not be empty")
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks/embeddings length mismatch: {len(chunks)} != {len(embeddings)}"
            )

        dimension: int | None = None
        for index, embedding in enumerate(embeddings):
            if not isinstance(embedding, list) or not embedding:
                raise ValueError(f"embeddings[{index}] must be a non-empty list")
            if not all(isinstance(value, (int, float)) for value in embedding):
                raise ValueError(f"embeddings[{index}] contains non-numeric values")
            if dimension is None:
                dimension = len(embedding)
            elif len(embedding) != dimension:
                raise ValueError("embedding dimensions are not consistent")

    @staticmethod
    def _point_id(chunk: dict) -> str:
        document_id = str(chunk.get("document_id", "")).strip()
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if not document_id or not chunk_id:
            raise ValueError("chunk must include document_id and chunk_id")
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{chunk_id}"))

    @staticmethod
    def _payload(chunk: dict) -> dict:
        metadata = (
            chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        )
        payload_metadata = {
            **metadata,
            "document_id": chunk.get("document_id"),
            "chunk_id": chunk.get("chunk_id"),
            "chunk_index": chunk.get("chunk_index"),
            "filename": chunk.get("filename"),
            "source": chunk.get("source"),
            "start_page": chunk.get("start_page"),
            "end_page": chunk.get("end_page"),
            "section": chunk.get("section"),
        }
        return {
            "document_id": chunk.get("document_id"),
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk.get("text"),
            "filename": chunk.get("filename"),
            "source": chunk.get("source"),
            "page": chunk.get("page"),
            "start_page": chunk.get("start_page"),
            "end_page": chunk.get("end_page"),
            "section": chunk.get("section"),
            "category": chunk.get("category") or metadata.get("category"),
            "subject": chunk.get("subject") or metadata.get("subject"),
            "chunk_index": chunk.get("chunk_index"),
            "content_hash": chunk.get("content_hash"),
            "created_at": chunk.get("created_at"),
            "metadata": payload_metadata,
        }
