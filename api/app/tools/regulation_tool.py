"""RegulationTool — bọc truy xuất nội quy cho Router RAG. [BR-307/407]"""
from __future__ import annotations

from typing import Any

from app.tools.base import BaseTool, ToolResult


class RegulationTool(BaseTool):
    name = "regulation"
    description = "Tra cứu điều/khoản nội quy CLB để trả lời câu hỏi tra cứu/tình huống."

    def run(self, query: str, **kwargs: Any) -> ToolResult:
        """Gọi retrieval nội quy rồi đóng gói kết quả.

        TODO(sinh viên): gọi domains.retrieval.service.retrieve_regulations(...)
        và map sang ToolResult(source="regulation", items=[rule_chunk...]).
        """
        raise NotImplementedError("US-307.1: nối RegulationTool với retrieval nội quy")


# TODO(sinh viên): registry.register(RegulationTool()) khi retrieval sẵn sàng.
