"""Interface LLM sinh câu trả lời (HỢP ĐỒNG cho sẵn). [BR-401/407]

Đặt sau abstraction để đổi Gemini ↔ GPT ↔ Claude không sửa domain answer.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, temperature: float = 0.2) -> str:
        """prompt (đã chèn ngữ cảnh truy xuất) → câu trả lời text."""
        raise NotImplementedError


# TODO(sinh viên): impl 1 provider free-tier (vd google-generativeai / Gemini).
