"""Tool abstraction cho Router RAG.

Router (app/routing) chọn một hoặc nhiều tool đã đăng ký; mỗi tool bọc một
năng lực truy xuất (media / nội quy). Đây cũng là đường nâng cấp lên Agentic
RAG (v2): khi đó một LLM-agent dùng CHUNG registry này để tự chọn tool.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    source: str                       # nhãn nguồn đã dùng: "media" | "regulation"
    items: list[Any] = field(default_factory=list)   # kết quả thô cho answer tổng hợp


class BaseTool(ABC):
    name: str
    description: str                   # để classifier/agent hiểu khi nào dùng tool

    @abstractmethod
    def run(self, query: str, **kwargs: Any) -> ToolResult:
        """Chạy tool và trả về kết quả thô kèm nhãn nguồn."""
        raise NotImplementedError
