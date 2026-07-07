"""Sinh câu trả lời RAG (LÕI HỌC). [BR-401/405/406/407]

Hợp đồng — sinh viên implement:
  synthesize_answer  : câu hỏi + ngữ cảnh media → câu trả lời BÁM nguồn + trích dẫn
  answer_regulation  : câu hỏi + rule_chunk → trả lời neo Điều/Khoản + DISCLAIMER
  handle_not_found   : ngữ cảnh rỗng/điểm thấp → thông báo "không tìm thấy", KHÔNG bịa
  Ràng buộc: ngữ cảnh rỗng → không gọi LLM sinh tự do; LLM lỗi → fallback kết quả thô.
  Gợi ý : providers.llm.LLMClient.generate(prompt)
  Pass  : tests/test_answer.py
"""
from __future__ import annotations

from typing import Any


def synthesize_answer(question: str, context: list[dict[str, Any]]) -> str:
    raise NotImplementedError("US-401.1: tổng hợp câu trả lời bám nguồn + trích dẫn")


def answer_regulation(question: str, rule_chunks: list[dict[str, Any]]) -> str:
    raise NotImplementedError("US-407.1: neo Điều/Khoản + disclaimer; không quy định → báo rõ")


def handle_not_found(question: str) -> str:
    raise NotImplementedError("US-406.1: thông báo không tìm thấy + gợi ý")
