"""Test answer — TC-401/406/407. Fake LLM (không gọi Gemini thật)."""
from __future__ import annotations

import pytest

from app.domains.answer.service import (
    answer_regulation,
    handle_not_found,
    synthesize_answer,
)

RULES = [
    {"id": 10, "article": 5, "clause": 2, "text": "Cấm hút thuốc trong phòng lab."},
    {"id": 11, "article": 8, "text": "Giữ trật tự khi sinh hoạt."},
]

CONTEXT = [
    {"id": 1, "score": 0.9, "video_id": 7, "timestamp": 12, "caption": "lễ khai giảng CLB"},
    {"id": 2, "score": 0.6, "video_id": 9, "timestamp": 3, "caption": "workshop Python"},
]


class SpyLLM:
    def __init__(self, reply="Trả lời có căn cứ [1]."):
        self.reply = reply
        self.prompt = None

    def generate(self, prompt, *, temperature=0.2):
        self.prompt = prompt
        return self.reply


class BoomLLM:
    def generate(self, prompt, *, temperature=0.2):
        raise RuntimeError("hết quota")


def test_answer_uses_context_and_returns_llm_text():
    llm = SpyLLM()
    out = synthesize_answer("có ảnh khai giảng không?", CONTEXT, llm=llm)
    assert out == "Trả lời có căn cứ [1]."
    assert "lễ khai giảng CLB" in llm.prompt      # ngữ cảnh được nhồi vào prompt
    assert "[1]" in llm.prompt and "[2]" in llm.prompt  # đánh số nguồn để trích dẫn


def test_empty_context_does_not_call_llm():
    llm = SpyLLM()
    out = synthesize_answer("câu ngoài miền", [], llm=llm)
    assert out == handle_not_found("câu ngoài miền")
    assert llm.prompt is None                      # không bịa khi rỗng


def test_llm_error_falls_back_to_raw_sources():
    out = synthesize_answer("x", CONTEXT, llm=BoomLLM())
    assert "video 7 @ 12s" in out                  # fallback vẫn có nguồn
    assert "lễ khai giảng CLB" in out


@pytest.mark.parametrize("q", ["", "abcxyz không liên quan"])
def test_not_found_message_no_hallucination(q):
    msg = handle_not_found(q)
    assert "chưa tìm thấy" in msg.lower()


# ---- T-42 answer_regulation (TC-407: citation + disclaimer) ----
def test_regulation_answer_has_citation_context_and_disclaimer():
    llm = SpyLLM(reply="Không được hút thuốc [Điều 5 Khoản 2].")
    out = answer_regulation("được hút thuốc trong lab không?", RULES, llm=llm)
    assert "Điều 5 Khoản 2" in llm.prompt          # neo điều/khoản vào prompt
    assert "tham khảo" in out                        # disclaimer luôn được nối
    assert out.startswith("Không được hút thuốc")


def test_regulation_empty_context_not_found():
    llm = SpyLLM()
    assert answer_regulation("hỏi linh tinh", [], llm=llm) == handle_not_found("hỏi linh tinh")
    assert llm.prompt is None


def test_regulation_llm_error_falls_back_with_disclaimer():
    out = answer_regulation("x", RULES, llm=BoomLLM())
    assert "Điều 5 Khoản 2" in out                   # fallback vẫn neo điều/khoản
    assert "tham khảo" in out                          # vẫn có disclaimer
