"""Orchestrate hội thoại Router RAG (LÕI HỌC). [BR-501/507]

Đây là nơi RÁP Router RAG lại:
  1. routing.intent.route(text, has_image, override) → chọn nguồn
  2. tools.registry.get(source).run(query)          → gọi tool tương ứng
  3. domains.answer.synthesize_answer / answer_regulation → câu trả lời + trích dẫn
  4. lưu messages (conversation)

Hợp đồng — sinh viên implement handle_message(...) theo 4 bước trên.
  Edge: ý định mơ hồ → hỏi làm rõ / chạy cả hai; ngữ cảnh rỗng → handle_not_found.
  Pass : tests/test_router.py + tests/test_answer.py
"""
from __future__ import annotations

from typing import Any


def handle_message(text: str | None, image_path: str | None = None,
                   override: str | None = None) -> dict[str, Any]:
    raise NotImplementedError("US-507.1: ráp routing → tools → answer")
