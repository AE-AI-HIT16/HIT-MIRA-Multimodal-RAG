from __future__ import annotations

from typing import Any

from src.configuration import AppConfig
from src.log.logger import logger


class EmbeddingConfigurationError(ValueError):
    """Raised when embedding configuration is incomplete or invalid."""


class EmbeddingServiceError(RuntimeError):
    """Raised when the embedding provider returns an invalid response."""


class EmbeddingService:
    """OpenAI-compatible HTTP embedding service.

    The service does not call the provider during import or construction. API calls
    happen only in embed_documents/embed_query.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        timeout: float = 60.0,
        provider: str = "openai-compatible",
        config: AppConfig | None = None,
    ) -> None:
        app_config = config or AppConfig()
        embedding_config = app_config.embedding

        self.api_key = api_key if api_key is not None else embedding_config.api_key
        self.base_url = (base_url if base_url is not None else embedding_config.base_url) or ""
        self.model = model if model is not None else embedding_config.model
        self._dimension = dimensions if dimensions is not None else embedding_config.dimensions
        self.timeout = timeout
        self.provider = provider

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        normalized_texts = self._validate_texts(texts)
        self._validate_configuration()
        logger.info(f"Embedding {len(normalized_texts)} document chunk(s) with model '{self.model}'")

        response_data = self._request_embeddings(normalized_texts)
        embeddings = self._extract_embeddings(response_data)
        self._validate_embeddings(embeddings, expected_count=len(normalized_texts))
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        return self.embed_documents([query])[0]

    def _validate_configuration(self) -> None:
        if self._is_missing(self.api_key):
            raise EmbeddingConfigurationError(
                "Missing embedding API key. Set EMBEDDING_API_KEY in the environment or secret manager."
            )
        if self._is_missing(self.base_url):
            raise EmbeddingConfigurationError("Missing embedding base URL. Set EMBEDDING_BASE_URL.")
        if self._is_missing(self.model):
            raise EmbeddingConfigurationError("Missing embedding model. Set EMBEDDING_MODEL.")

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

    @staticmethod
    def _validate_texts(texts: list[str]) -> list[str]:
        if not isinstance(texts, list):
            raise TypeError("texts must be a list[str]")
        if not texts:
            raise ValueError("texts must not be empty")

        normalized: list[str] = []
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"texts[{index}] must be a string")
            cleaned = text.strip()
            if not cleaned:
                raise ValueError(f"texts[{index}] must not be empty")
            normalized.append(cleaned)
        return normalized

    def _request_embeddings(self, texts: list[str]) -> dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:
            raise EmbeddingConfigurationError("Missing dependency 'httpx' for embedding HTTP calls.") from exc

        endpoint = self._embedding_endpoint()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        if self._dimension is not None:
            payload["dimensions"] = int(self._dimension)

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            raise EmbeddingServiceError(f"Embedding provider returned HTTP {status_code}.") from exc
        except httpx.HTTPError as exc:
            raise EmbeddingServiceError("Embedding provider request failed.") from exc
        except ValueError as exc:
            raise EmbeddingServiceError("Embedding provider returned non-JSON response.") from exc

    def _embedding_endpoint(self) -> str:
        base_url = str(self.base_url).rstrip("/")
        if base_url.endswith("/embeddings"):
            return base_url
        return f"{base_url}/embeddings"

    @staticmethod
    def _extract_embeddings(response_data: dict[str, Any]) -> list[list[float]]:
        data = response_data.get("data")
        if not isinstance(data, list):
            raise EmbeddingServiceError("Embedding response missing 'data' list.")

        ordered = sorted(data, key=lambda item: item.get("index", 0) if isinstance(item, dict) else 0)
        embeddings: list[list[float]] = []
        for item in ordered:
            if not isinstance(item, dict) or "embedding" not in item:
                raise EmbeddingServiceError("Embedding response item missing 'embedding'.")
            embedding = item["embedding"]
            if not isinstance(embedding, list):
                raise EmbeddingServiceError("Embedding must be a list of floats.")
            embeddings.append(embedding)
        return embeddings

    def _validate_embeddings(self, embeddings: list[list[float]], expected_count: int) -> None:
        if len(embeddings) != expected_count:
            raise EmbeddingServiceError(
                f"Embedding count mismatch: expected {expected_count}, got {len(embeddings)}."
            )

        observed_dimension: int | None = None
        for index, embedding in enumerate(embeddings):
            if not embedding:
                raise EmbeddingServiceError(f"Embedding at index {index} is empty.")
            if not all(isinstance(value, (int, float)) for value in embedding):
                raise EmbeddingServiceError(f"Embedding at index {index} contains non-numeric values.")
            current_dimension = len(embedding)
            if observed_dimension is None:
                observed_dimension = current_dimension
            elif current_dimension != observed_dimension:
                raise EmbeddingServiceError("Embedding dimensions are not consistent.")

        if self._dimension is not None and observed_dimension != int(self._dimension):
            raise EmbeddingServiceError(
                f"Embedding dimension mismatch: expected {self._dimension}, got {observed_dimension}."
            )
