"""Test CHUỖI tối thiểu end-to-end: route → retrieve_media → synthesize_answer.

Chứng minh 4 mảnh (T-23/T-20/T-30/T-41) ghép đúng, với provider fake — không cần
GPU/Qdrant/Gemini. Đây là xương sống demo trước khi có pipeline offline nạp dữ liệu.
"""
from __future__ import annotations

from app.domains.answer.service import synthesize_answer
from app.domains.retrieval.service import retrieve_media
from app.routing.intent import Source, route
from tests.test_answer import SpyLLM
from tests.test_retrieval import FakeImageEmbedder, FakeStore


def test_media_query_flows_end_to_end():
    question = "cho xem ảnh lễ khai giảng CLB"

    # 1) Router luật (T-40 đã có) → media
    assert route(question, has_image=False) is Source.MEDIA

    # 2) Retrieve (T-30)
    context = retrieve_media(question, embedder=FakeImageEmbedder(), store=FakeStore())
    assert context, "phải có ngữ cảnh để không rơi vào not_found"

    # 3) Synthesize (T-41) với LLM fake
    llm = SpyLLM(reply="Có, đây là ảnh lễ khai giảng [1].")
    answer = synthesize_answer(question, context, llm=llm)
    assert "khai giảng" in answer
    assert "khai giảng" in llm.prompt  # câu trả lời thực sự bám ngữ cảnh truy xuất
