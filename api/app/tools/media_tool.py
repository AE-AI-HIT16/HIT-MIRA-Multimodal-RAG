"""MediaTool — bọc năng lực truy xuất media cho Router RAG. [BR-301/302/303/308]"""
from __future__ import annotations

from typing import Any

from app.tools.base import BaseTool, ToolResult


class MediaTool(BaseTool):
    name = "media"
    description = "Tìm ảnh/sự kiện/clip video theo text hoặc ảnh, gồm cả lời nói (transcript)."

    def run(self, query: str, **kwargs: Any) -> ToolResult:
        """Gọi retrieval media rồi đóng gói kết quả.

        TODO(sinh viên): gọi domains.retrieval.service.retrieve_media(...) và
        map sang ToolResult(source="media", items=[...]).
        """
        raise NotImplementedError("US-301.1: nối MediaTool với retrieval media")


# TODO(sinh viên): registry.register(MediaTool()) khi retrieval sẵn sàng.
