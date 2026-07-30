"""Media embedding services for video retrieval indexing."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any, Callable

from src.configuration import AppConfig
from src.log.logger import logger


class ImageEmbeddingConfigurationError(ValueError):
    """Raised when image embedding configuration is missing or invalid."""


class ImageEmbeddingServiceError(RuntimeError):
    """Raised when the image embedding provider returns invalid vectors."""


class ImageEmbeddingService:
    """Jina Embeddings API adapter for keyframe image and transcript text vectors."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
        timeout: float | None = None,
        config: AppConfig | None = None,
        http_post: Callable[..., Any] | None = None,
    ) -> None:
        model_config = (config or AppConfig()).media_models
        self.model_name = (
            model_name
            or os.getenv("MEDIA_IMAGE_EMBEDDING_MODEL")
            or getattr(model_config, "clip_model_name", None)
            or "jina-clip-v2"
        )
        self.api_key = api_key if api_key is not None else getattr(model_config, "clip_api_key", None)
        self.base_url = (
            base_url
            or os.getenv("MEDIA_IMAGE_EMBEDDING_BASE_URL")
            or getattr(model_config, "clip_api_base_url", None)
            or "https://api.jina.ai/v1/embeddings"
        )
        self.dimensions = dimensions if dimensions is not None else getattr(model_config, "clip_embedding_dimensions", 1024)
        self.timeout = timeout if timeout is not None else getattr(model_config, "clip_api_timeout", 60.0)
        self._http_post = http_post

    def embed_images(self, image_paths: list[str | Path]) -> list[list[float]]:
        paths = self._validate_image_paths(image_paths)
        logger.info(f"Embedding {len(paths)} keyframe image(s) with Jina model '{self.model_name}'")
        payload = {
            "model": self.model_name,
            "input": [{"image": self._image_as_base64(path)} for path in paths],
            "embedding_type": "float",
            "dimensions": self.dimensions,
            "normalized": True,
        }
        return self._embed_payload(payload, expected_count=len(paths), label="Image")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        clean_texts = self._validate_texts(texts)
        logger.info(f"Embedding {len(clean_texts)} transcript text chunk(s) with Jina model '{self.model_name}'")
        payload = {
            "model": self.model_name,
            "input": [{"text": text} for text in clean_texts],
            "task": "retrieval.query",
            "embedding_type": "float",
            "dimensions": self.dimensions,
            "normalized": True,
        }
        return self._embed_payload(payload, expected_count=len(clean_texts), label="Text")

    def _embed_payload(self, payload: dict[str, Any], *, expected_count: int, label: str) -> list[list[float]]:
        self._validate_configuration()
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = self._post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                embeddings = self._response_embeddings(response.json())
            except ImageEmbeddingServiceError:
                raise
            except Exception as exc:
                if attempt < max_attempts - 1 and self._is_retryable_provider_error(exc):
                    delay = self._retry_delay_seconds(exc, attempt)
                    logger.warning(
                        f"{label} embedding provider returned a retryable error; "
                        f"retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue
                raise ImageEmbeddingServiceError(
                    f"{label} embedding provider request failed: {exc.__class__.__name__}."
                ) from exc
            self._validate_embeddings(embeddings, expected_count=expected_count)
            return embeddings
        raise ImageEmbeddingServiceError(f"{label} embedding provider request failed after retries.")

    def _validate_configuration(self) -> None:
        if self._is_missing(self.api_key):
            raise ImageEmbeddingConfigurationError("Missing image embedding configuration. Set JINA_API_KEY.")
        if self._is_missing(self.base_url):
            raise ImageEmbeddingConfigurationError("Missing image embedding configuration. Set MEDIA_IMAGE_EMBEDDING_BASE_URL.")
        if self._is_missing(self.model_name):
            raise ImageEmbeddingConfigurationError("Missing image embedding configuration. Set MEDIA_IMAGE_EMBEDDING_MODEL.")
        if not isinstance(self.dimensions, int) or self.dimensions <= 0:
            raise ImageEmbeddingConfigurationError("Image embedding dimensions must be a positive integer.")

    def _post(self, *args: Any, **kwargs: Any) -> Any:
        if self._http_post is not None:
            return self._http_post(*args, **kwargs)
        try:
            import httpx
        except ImportError as exc:
            raise ImageEmbeddingConfigurationError("Missing dependency 'httpx' for Jina image embedding API.") from exc
        return httpx.post(*args, **kwargs)

    @staticmethod
    def _is_retryable_provider_error(exc: Exception) -> bool:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        return status_code in {429, 500, 502, 503, 504}

    @staticmethod
    def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", {}) or {}
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), 30.0)
            except ValueError:
                pass
        return float(2 ** attempt)

    @staticmethod
    def _image_as_base64(path: Path) -> str:
        return base64.b64encode(path.read_bytes()).decode("ascii")

    @classmethod
    def _response_embeddings(cls, response_data: Any) -> list[list[float]]:
        if not isinstance(response_data, dict) or not isinstance(response_data.get("data"), list):
            raise ImageEmbeddingServiceError("Image embedding API response does not contain a data list.")
        rows = response_data["data"]
        try:
            rows = sorted(rows, key=lambda row: int(row.get("index", 0)))
            return [cls._as_vector(row.get("embedding")) for row in rows]
        except (AttributeError, TypeError, ValueError) as exc:
            raise ImageEmbeddingServiceError("Image embedding API response contains invalid vectors.") from exc

    @staticmethod
    def _validate_image_paths(image_paths: list[str | Path]) -> list[Path]:
        if not isinstance(image_paths, list):
            raise TypeError("image_paths must be a list")
        if not image_paths:
            raise ValueError("image_paths must not be empty")
        paths = [Path(path) for path in image_paths]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"image file does not exist: {missing[0]}")
        return paths

    @staticmethod
    def _validate_texts(texts: list[str]) -> list[str]:
        if not isinstance(texts, list):
            raise TypeError("texts must be a list")
        if not texts:
            raise ValueError("texts must not be empty")
        clean_texts = [str(text).strip() for text in texts]
        if any(not text for text in clean_texts):
            raise ValueError("texts must not contain empty values")
        return clean_texts

    @staticmethod
    def _as_vector(embedding: Any) -> list[float]:
        if not isinstance(embedding, list):
            raise TypeError("embedding must be a list")
        return [float(value) for value in embedding]

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

    def _validate_embeddings(self, embeddings: list[list[float]], expected_count: int) -> None:
        if len(embeddings) != expected_count:
            raise ImageEmbeddingServiceError(
                f"Image embedding count mismatch: expected {expected_count}, got {len(embeddings)}."
            )
        dimensions = {len(embedding) for embedding in embeddings if embedding}
        if len(dimensions) != 1:
            raise ImageEmbeddingServiceError("Image embedding dimensions are not consistent.")
        observed_dimension = dimensions.pop()
        if observed_dimension != self.dimensions:
            raise ImageEmbeddingServiceError(
                f"Image embedding dimension mismatch: expected {self.dimensions}, got {observed_dimension}."
            )
