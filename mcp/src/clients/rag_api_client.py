from __future__ import annotations

from typing import Any

import httpx

from config.configs import config_object


class RAGApiError(RuntimeError):
    """Raised when the FastAPI RAG backend rejects a request."""


class RAGApiClient:
    """Async HTTP client for the existing FastAPI RAG endpoints."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.base_url = (base_url or config_object.API.BASE_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds or config_object.API.TIMEOUT_SECONDS

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.request(
                method,
                url,
                json=json,
            )

        if response.is_error:
            try:
                detail: Any = response.json()
            except ValueError:
                detail = response.text[:1000]
            raise RAGApiError(
                f"FastAPI {method} {path} trả HTTP {response.status_code}: {detail}"
            )

        try:
            return response.json()
        except ValueError as exc:
            raise RAGApiError(
                f"FastAPI {method} {path} không trả về JSON hợp lệ"
            ) from exc

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
        rewrite: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": query,
            "rewrite": rewrite,
        }
        if top_k is not None:
            payload["top_k"] = top_k
        if document_ids:
            payload["document_ids"] = document_ids
        return await self._request("POST", "/api/retrieval/search", json=payload)

    async def search_media(
        self,
        query: str,
        top_k: int | None = None,
        video_ids: list[str] | None = None,
        source: str = "both",
        years: list[int] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": query,
            "source": source,
        }
        if top_k is not None:
            payload["top_k"] = top_k
        if video_ids:
            payload["video_ids"] = video_ids
        if years:
            payload["years"] = years
        return await self._request("POST", "/api/media/search", json=payload)


_client: RAGApiClient | None = None


def get_rag_api_client() -> RAGApiClient:
    global _client
    if _client is None:
        _client = RAGApiClient()
    return _client
