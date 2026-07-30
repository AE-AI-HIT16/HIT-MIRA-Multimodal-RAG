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
