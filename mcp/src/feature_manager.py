from __future__ import annotations

from clients.rag_api_client import get_rag_api_client
from logger import get_logger, log_exceptions


logger = get_logger(__name__)


class FeatureManager:
    """Handlers whose names must match `name_tool` in Resources/tools.yaml."""

    @staticmethod
    @log_exceptions(logger)
    async def search_regulations(
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
        rewrite: bool = True,
    ) -> dict:
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("query không được để trống")
        if top_k is not None and not 1 <= top_k <= 50:
            raise ValueError("top_k phải nằm trong khoảng 1..50")

        result = await get_rag_api_client().search(
            query=normalized_query,
            top_k=top_k,
            document_ids=document_ids,
            rewrite=rewrite,
        )
        return {"success": True, **result}

    @staticmethod
    @log_exceptions(logger)
    async def search_media(
        query: str,
        top_k: int | None = None,
        video_ids: list[str] | None = None,
        source: str = "both",
    ) -> dict:
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("query không được để trống")
        if top_k is not None and not 1 <= top_k <= 50:
            raise ValueError("top_k phải nằm trong khoảng 1..50")
        normalized_source = str(source or "both").strip().lower()
        if normalized_source not in {"clip", "transcript", "both"}:
            raise ValueError("source phải là 'clip', 'transcript' hoặc 'both'")

        result = await get_rag_api_client().search_media(
            query=normalized_query,
            top_k=top_k,
            video_ids=video_ids,
            source=normalized_source,
        )
        return {"success": True, **result}
