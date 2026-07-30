"""Qdrant vector-store adapter for video retrieval collections."""

from __future__ import annotations

import uuid
from typing import Any

from src.configuration import AppConfig
from src.log.logger import logger


class VideoVectorStoreConfigurationError(ValueError):
    """Raised when Qdrant configuration is incomplete or invalid."""


class VideoVectorStoreError(RuntimeError):
    """Raised when Qdrant rejects an operation or returns invalid metadata."""


class QdrantVideoVectorStore:
    """Qdrant store for media_clip and video_transcript retrieval points."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        distance: str | None = None,
        client: Any | None = None,
        config: AppConfig | None = None,
    ) -> None:
        app_config = config or AppConfig()
        qdrant_config = app_config.qdrant
        self.url = url if url is not None else qdrant_config.url
        self.api_key = api_key if api_key is not None else qdrant_config.api_key
        self.distance_name = (distance or getattr(qdrant_config, "distance", None) or "COSINE").upper()
        self.client = client or self._create_client()
        self.models = self._load_models()

    def upsert_points(
        self,
        *,
        collection_name: str,
        point_ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self._validate_upsert_inputs(point_ids, vectors, payloads)
        vector_size = len(vectors[0])
        self.ensure_collection(collection_name=collection_name, vector_size=vector_size)
        points = [
            self.models.PointStruct(
                id=self._point_id(point_id),
                vector=vector,
                payload=payload,
            )
            for point_id, vector, payload in zip(point_ids, vectors, payloads)
        ]
        self.client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Upserted {len(points)} point(s) into Qdrant collection '{collection_name}'")
        return {"collection_name": collection_name, "upserted": len(points)}

    def search_points(
        self,
        *,
        collection_name: str,
        vector: list[float],
        limit: int,
        query_filter: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Tìm điểm gần nhất trong một collection.

        Collection chưa tồn tại (video chưa được index) là trạng thái hợp lệ ->
        trả về [] để tầng trên gọi luồng "không tìm thấy", không bịa kết quả.
        """
        self._validate_search_inputs(vector, limit)
        if not self._collection_exists(collection_name):
            logger.warning(
                f"Qdrant collection '{collection_name}' does not exist yet; returning no results"
            )
            return []

        raw_points = self._query_points(
            collection_name=collection_name,
            vector=vector,
            limit=limit,
            query_filter=query_filter,
        )
        results = [self._as_search_result(point) for point in raw_points]
        logger.info(f"Qdrant search on '{collection_name}' returned {len(results)} point(s)")
        return results

    def video_id_filter(self, video_ids: list[str] | None) -> Any | None:
        """Build filter giới hạn theo payload key `video_id` (None nếu không lọc)."""
        clean_ids = [str(item).strip() for item in video_ids or [] if str(item or "").strip()]
        if not clean_ids:
            return None
        conditions = [
            self.models.FieldCondition(key="video_id", match=self.models.MatchValue(value=video_id))
            for video_id in clean_ids
        ]
        return self.models.Filter(should=conditions)

    def _query_points(
        self,
        *,
        collection_name: str,
        vector: list[float],
        limit: int,
        query_filter: Any | None,
    ) -> list[Any]:
        # qdrant-client >= 1.10 dùng query_points, bản cũ chỉ có search.
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=collection_name,
                query=vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )
            points = getattr(response, "points", response)
            return list(points or [])
        response = self.client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        return list(response or [])

    @staticmethod
    def _as_search_result(point: Any) -> dict[str, Any]:
        payload = getattr(point, "payload", None)
        if not isinstance(payload, dict):
            payload = {}
        return {
            "id": str(getattr(point, "id", "") or ""),
            "score": float(getattr(point, "score", 0.0) or 0.0),
            "payload": dict(payload),
        }

    @staticmethod
    def _validate_search_inputs(vector: list[float], limit: int) -> None:
        if not isinstance(vector, list) or not vector:
            raise ValueError("vector must be a non-empty list")
        if not all(isinstance(value, (int, float)) for value in vector):
            raise ValueError("vector contains non-numeric values")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

    def ensure_collection(self, *, collection_name: str, vector_size: int) -> None:
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("collection_name must be a non-empty string")
        if not isinstance(vector_size, int) or vector_size <= 0:
            raise ValueError("vector_size must be a positive integer")

        if not self._collection_exists(collection_name):
            distance = getattr(self.models.Distance, self.distance_name, None)
            if distance is None:
                raise VideoVectorStoreConfigurationError(f"Unsupported Qdrant distance: {self.distance_name}")
            logger.info(f"Creating Qdrant collection '{collection_name}' with vector size {vector_size}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=self.models.VectorParams(size=vector_size, distance=distance),
            )
            return

        existing_size = self._collection_vector_size(collection_name)
        if existing_size is not None and existing_size != vector_size:
            raise VideoVectorStoreConfigurationError(
                "Qdrant collection vector dimension mismatch: "
                f"collection '{collection_name}' has {existing_size}, requested {vector_size}."
            )

    def _create_client(self) -> Any:
        if self._is_missing(self.url):
            raise VideoVectorStoreConfigurationError("Missing Qdrant URL. Set QDRANT_URL.")
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise VideoVectorStoreConfigurationError("Missing dependency 'qdrant-client'.") from exc

        kwargs: dict[str, Any] = {"url": self.url}
        if not self._is_missing(self.api_key):
            kwargs["api_key"] = self.api_key
        return QdrantClient(**kwargs)

    @staticmethod
    def _load_models() -> Any:
        try:
            from qdrant_client import models
        except ImportError as exc:
            raise VideoVectorStoreConfigurationError("Missing dependency 'qdrant-client'.") from exc
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

    def _collection_exists(self, collection_name: str) -> bool:
        if hasattr(self.client, "collection_exists"):
            return bool(self.client.collection_exists(collection_name=collection_name))
        try:
            self.client.get_collection(collection_name=collection_name)
            return True
        except Exception:
            return False

    def _collection_vector_size(self, collection_name: str) -> int | None:
        collection = self.client.get_collection(collection_name=collection_name)
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

    @classmethod
    def _validate_upsert_inputs(
        cls,
        point_ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        if not point_ids:
            raise ValueError("point_ids must not be empty")
        if len(point_ids) != len(vectors) or len(point_ids) != len(payloads):
            raise ValueError(
                "point_ids/vectors/payloads length mismatch: "
                f"{len(point_ids)} != {len(vectors)} != {len(payloads)}"
            )
        dimension: int | None = None
        for index, embedding in enumerate(vectors):
            if not isinstance(embedding, list) or not embedding:
                raise ValueError(f"vectors[{index}] must be a non-empty list")
            if not all(isinstance(value, (int, float)) for value in embedding):
                raise ValueError(f"vectors[{index}] contains non-numeric values")
            if dimension is None:
                dimension = len(embedding)
            elif len(embedding) != dimension:
                raise ValueError("vector dimensions are not consistent")
        for index, point_id in enumerate(point_ids):
            if not str(point_id or "").strip():
                raise ValueError(f"point_ids[{index}] must be non-empty")
        for index, payload in enumerate(payloads):
            if not isinstance(payload, dict):
                raise TypeError(f"payloads[{index}] must be a dict")

    @staticmethod
    def _point_id(value: str) -> str:
        text = str(value).strip()
        try:
            return str(uuid.UUID(text))
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, text))
