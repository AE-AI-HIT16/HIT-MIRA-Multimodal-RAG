"""RegulationTool — bọc truy xuất nội quy cho Router RAG. [BR-307/407]"""
from __future__ import annotations

from typing import Any

from app.tools.base import BaseTool, ToolResult


class RegulationTool(BaseTool):
    name = "regulation"
    description = "Tra cứu điều/khoản nội quy CLB để trả lời câu hỏi tra cứu/tình huống."

    def run(self, query: str, *, top_k: int = 5, threshold: float = 0.0,
            **kwargs: Any) -> ToolResult:
        """Gọi retrieval nội quy rồi đóng gói kết quả. [T-34]"""
        from app.domains.retrieval import service as retrieval

        hits = retrieval.rank(
            retrieval.retrieve_regulations(query, top_k=top_k), threshold=threshold,
        )
        return ToolResult(source="regulation", items=hits)
