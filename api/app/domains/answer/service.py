"""Sinh câu trả lời RAG (LÕI HỌC). [BR-401/405/406/407]

Hợp đồng — sinh viên implement:
  synthesize_answer  : câu hỏi + ngữ cảnh media → câu trả lời BÁM nguồn + trích dẫn   ✅ T-41
  answer_regulation  : câu hỏi + rule_chunk → trả lời neo Điều/Khoản + DISCLAIMER      ⏳ T-42
  handle_not_found   : ngữ cảnh rỗng/điểm thấp → thông báo "không tìm thấy", KHÔNG bịa  ✅ T-43(min)
  Ràng buộc: ngữ cảnh rỗng → không gọi LLM sinh tự do; LLM lỗi → fallback kết quả thô.
  Gợi ý : providers.llm.LLMClient.generate(prompt)
  Pass  : tests/test_answer.py
"""
from __future__ import annotations

from typing import Any

_SYSTEM = (
    "Bạn là trợ lý của CLB Tin học HIT. CHỈ trả lời dựa trên NGỮ CẢNH cho dưới đây, "
    "bằng tiếng Việt, ngắn gọn. Nếu ngữ cảnh không đủ để trả lời, hãy nói rõ là chưa "
    "tìm thấy — TUYỆT ĐỐI không bịa. Cuối mỗi ý, trích nguồn dạng [n] theo số thứ tự."
)


def _source_label(item: dict[str, Any], idx: int) -> str:
    """Nhãn nguồn để neo trích dẫn: ưu tiên video_id+timestamp, fallback id."""
    vid = item.get("video_id")
    ts = item.get("timestamp")
    if vid is not None and ts is not None:
        return f"video {vid} @ {ts}s"
    if vid is not None:
        return f"video {vid}"
    return f"media#{item.get('id', idx)}"


def _snippet(item: dict[str, Any]) -> str:
    """Nội dung mô tả một item ngữ cảnh (caption/transcript/text)."""
    for key in ("caption", "transcript", "text", "summary"):
        val = item.get(key)
        if val:
            return str(val)
    return "(không có mô tả text)"


def _build_prompt(question: str, context: list[dict[str, Any]]) -> str:
    lines = [f"[{i}] ({_source_label(it, i)}) {_snippet(it)}" for i, it in enumerate(context, 1)]
    ctx = "\n".join(lines)
    return f"{_SYSTEM}\n\nNGỮ CẢNH:\n{ctx}\n\nCÂU HỎI: {question}\n\nTRẢ LỜI:"


def _raw_fallback(context: list[dict[str, Any]]) -> str:
    """LLM lỗi → trả kết quả truy xuất thô kèm nguồn, không sinh tự do."""
    head = "Không tạo được câu trả lời (LLM lỗi). Kết quả liên quan nhất:"
    lines = [f"[{i}] {_source_label(it, i)}: {_snippet(it)}" for i, it in enumerate(context, 1)]
    return "\n".join([head, *lines])


def synthesize_answer(question: str, context: list[dict[str, Any]], *,
                      llm: Any | None = None) -> str:
    """US-401.1: tổng hợp câu trả lời bám nguồn + trích dẫn.

    Ngữ cảnh rỗng → không gọi LLM (không bịa). LLM lỗi → fallback kết quả thô.
    """
    if not context:
        return handle_not_found(question)

    if llm is None:
        from app.deps import get_llm

        llm = get_llm()

    prompt = _build_prompt(question, context)
    try:
        return llm.generate(prompt)
    except Exception:  # noqa: BLE001 — LLM lỗi/hết quota → hạ cấp mượt, không vỡ request
        return _raw_fallback(context)


_DISCLAIMER = (
    "\n\n⚠️ Trích từ nội quy CLB, chỉ mang tính tham khảo; khi có tranh chấp căn cứ "
    "văn bản gốc do Ban chủ nhiệm ban hành."
)

_RULE_SYSTEM = (
    "Bạn là trợ lý tra cứu nội quy CLB Tin học HIT. CHỈ trả lời dựa trên các ĐIỀU/KHOẢN "
    "cho dưới đây, bằng tiếng Việt. Neo mỗi ý bằng số điều/khoản dạng [Điều X Khoản Y]. "
    "Nếu nội quy KHÔNG quy định điều được hỏi, nói rõ 'nội quy không quy định' — không suy diễn."
)


def _rule_label(item: dict[str, Any], idx: int) -> str:
    dieu = item.get("article") or item.get("dieu")
    khoan = item.get("clause") or item.get("khoan")
    if dieu and khoan:
        return f"Điều {dieu} Khoản {khoan}"
    if dieu:
        return f"Điều {dieu}"
    return f"nội quy #{item.get('id', idx)}"


def _build_regulation_prompt(question: str, rule_chunks: list[dict[str, Any]]) -> str:
    lines = [f"[{_rule_label(it, i)}] {_snippet(it)}" for i, it in enumerate(rule_chunks, 1)]
    ctx = "\n".join(lines)
    return f"{_RULE_SYSTEM}\n\nNỘI QUY:\n{ctx}\n\nCÂU HỎI: {question}\n\nTRẢ LỜI:"


def _raw_regulation_fallback(rule_chunks: list[dict[str, Any]]) -> str:
    head = "Không tạo được câu trả lời (LLM lỗi). Điều/khoản liên quan nhất:"
    lines = [f"[{_rule_label(it, i)}] {_snippet(it)}" for i, it in enumerate(rule_chunks, 1)]
    return "\n".join([head, *lines]) + _DISCLAIMER


def answer_regulation(question: str, rule_chunks: list[dict[str, Any]], *,
                      llm: Any | None = None) -> str:
    """US-407.1: neo Điều/Khoản + disclaimer; ngữ cảnh rỗng → không bịa; LLM lỗi → thô."""
    if not rule_chunks:
        return handle_not_found(question)

    if llm is None:
        from app.deps import get_llm

        llm = get_llm()

    prompt = _build_regulation_prompt(question, rule_chunks)
    try:
        answer = llm.generate(prompt)
    except Exception:  # noqa: BLE001 — hạ cấp mượt, luôn kèm disclaimer
        return _raw_regulation_fallback(rule_chunks)
    return answer + _DISCLAIMER


def handle_not_found(question: str) -> str:
    """US-406.1: thông báo không tìm thấy, không bịa."""
    return (
        "Mình chưa tìm thấy dữ liệu phù hợp trong kho ảnh/video/nội quy của CLB cho câu hỏi "
        "này. Bạn thử diễn đạt lại, thêm từ khóa cụ thể (tên sự kiện, năm), hoặc gửi kèm ảnh nhé."
    )
