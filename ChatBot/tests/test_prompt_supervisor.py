"""System prompt là nơi chứa các ràng buộc chống bịa — sửa nhầm là hỏng thầm.

Prompt không có compiler nào kiểm. Một lần sửa câu chữ cho "mượt hơn" đủ để
mất dòng disclaimer bắt buộc của US-407.1, hoặc mất luật cấm gán mốc thời gian
cho ảnh tĩnh. Không ai thấy gì cả cho tới lúc chatbot trả lời sai trước hội
đồng. Vài phép so chuỗi ở đây rẻ hơn nhiều so với cái giá đó.
"""

from __future__ import annotations

import asyncio

import pytest

from src.config.configs import config_prompts
from src.graph.middlewares.simple_middlewares import SafeDict
from src.graph.utils import load_sys_prompt

# Chép nguyên văn từ US-407.1. Đây là câu bắt buộc, không phải gợi ý diễn đạt.
DISCLAIMER = "Thông tin trên chỉ mang tính tham khảo, quyết định cuối cùng thuộc về Ban Chủ nhiệm CLB."


@pytest.fixture(scope="module")
def prompt() -> str:
    """Đọc prompt qua đúng đường mà runtime dùng, nên cũng kiểm luôn cấu hình đường dẫn."""
    return asyncio.run(load_sys_prompt(config_prompts.PROFILE_GRAPH.PROMPT_SUPERVISOR_PATH))


def test_co_dong_disclaimer_dung_nguyen_van(prompt: str) -> None:
    """US-407.1: câu trả lời nội quy phải kết thúc bằng đúng câu này."""
    assert DISCLAIMER in prompt


def test_mo_ta_du_ca_hai_tool(prompt: str) -> None:
    """Agent được cấp tool nhưng prompt không tả cách dùng thì nó sẽ không gọi."""
    assert "search_regulations" in prompt
    assert "search_media" in prompt


def test_cam_gan_moc_thoi_gian_cho_anh_tinh(prompt: str) -> None:
    """Ảnh tĩnh cố tình không mang `timestamp_sec` — gán vào là bịa ra một khoảnh khắc."""
    assert "timestamp_sec" in prompt
    assert "null" in prompt


def test_co_luat_viet_lai_loi_thoai_asr(prompt: str) -> None:
    """Zipformer trả về chữ in hoa không dấu câu; phải dặn viết lại mà giữ nguyên từ."""
    assert "viết hoa toàn bộ" in prompt


def test_prompt_khong_lam_vo_buoc_format_map(prompt: str) -> None:
    """Middleware chạy `format_map` lên prompt trước mỗi lượt gọi model.

    Chỉ cần một dấu `{` lẻ trong prompt (ví dụ dán một mẫu JSON vào) là mọi
    lượt chat ném `ValueError` ngay tại middleware, trước khi kịp gọi model.
    `SafeDict` chỉ cứu được key thiếu, không cứu được ngoặc lệch.
    """
    ket_qua = prompt.format_map(SafeDict({"memories": "", "messages": []}))
    assert DISCLAIMER in ket_qua
