"""MediaTool — bọc năng lực truy xuất media cho Router RAG. [BR-301/302/303/308]"""
from __future__ import annotations

from typing import Any

from app.tools.base import BaseTool, ToolResult


class MediaTool(BaseTool):
    name = "media"
    description = "Tìm ảnh/sự kiện/clip video theo text hoặc ảnh, gồm cả lời nói (transcript)."

    def run(self, query: str, *, top_k: int = 5, threshold: float = 0.0,
            **kwargs: Any) -> ToolResult:
        """Gọi retrieval media (+ transcript) rồi đóng gói kết quả. [T-34]"""
        from app.domains.retrieval import service as retrieval

        hits = retrieval.rank(retrieval.retrieve_media(query, top_k=top_k), threshold=threshold)
        return ToolResult(source="media", items=hits)
