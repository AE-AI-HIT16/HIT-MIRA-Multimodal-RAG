from __future__ import annotations

from typing import Any

from langfuse import observe

from src.configuration import AppConfig
from src.log.logger import logger


class EmbeddingConfigurationError(ValueError):
    """Raised when embedding configuration is incomplete or invalid."""


class EmbeddingServiceError(RuntimeError):
    """Raised when the embedding provider returns an invalid response."""


class EmbeddingService:
    """LangChain OpenAI-compatible embedding adapter."""

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
        embedding_config = (config or AppConfig()).embedding
        self.api_key = api_key if api_key is not None else embedding_config.api_key
        self.base_url = (base_url if base_url is not None else embedding_config.base_url) or ""
        self.model = model if model is not None else embedding_config.model
        self._dimension = dimensions if dimensions is not None else embedding_config.dimensions
        self.timeout = timeout
        self.provider = provider
        self._client = None

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        normalized_texts = self._validate_texts(texts)
        self._validate_configuration()
        logger.info(f"Embedding {len(normalized_texts)} document chunk(s) with model '{self.model}'")

        try:
            embeddings = self._embedding_provider_request(normalized_texts)
        except Exception as exc:
            raise EmbeddingServiceError(f"Embedding provider request failed: {exc.__class__.__name__}.") from exc

        self._validate_embeddings(embeddings, expected_count=len(normalized_texts))
        return embeddings

    @observe(name="embedding_provider_request")
    def _embedding_provider_request(self, texts: list[str]) -> list[list[float]]:
        return self._embedding_client().embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        return self.embed_documents([query])[0]

    def _embedding_client(self):
        if self._client is not None:
            return self._client
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise EmbeddingConfigurationError("Missing dependency 'langchain-openai'.") from exc

        kwargs: dict[str, Any] = {
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "tiktoken_enabled": False,
            "check_embedding_ctx_length": False,
        }
        if self._dimension is not None:
            kwargs["dimensions"] = int(self._dimension)
        self._client = OpenAIEmbeddings(**kwargs)
        return self._client

    def _validate_configuration(self) -> None:
        missing = {
            "EMBEDDING_API_KEY": self.api_key,
            "EMBEDDING_BASE_URL": self.base_url,
            "EMBEDDING_MODEL": self.model,
        }
        for env_name, value in missing.items():
            if self._is_missing(value):
                raise EmbeddingConfigurationError(f"Missing embedding configuration. Set {env_name}.")

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
        normalized = [text.strip() for text in texts if isinstance(text, str) and text.strip()]
        if len(normalized) != len(texts):
            raise ValueError("texts must contain only non-empty strings")
        return normalized

    def _validate_embeddings(self, embeddings: list[list[float]], expected_count: int) -> None:
        if len(embeddings) != expected_count:
            raise EmbeddingServiceError(
                f"Embedding count mismatch: expected {expected_count}, got {len(embeddings)}."
            )
        dimensions = {len(embedding) for embedding in embeddings if embedding}
        if len(dimensions) != 1:
            raise EmbeddingServiceError("Embedding dimensions are not consistent.")
        observed_dimension = dimensions.pop()
        if self._dimension is not None and observed_dimension != int(self._dimension):
            raise EmbeddingServiceError(
                f"Embedding dimension mismatch: expected {self._dimension}, got {observed_dimension}."
            )
