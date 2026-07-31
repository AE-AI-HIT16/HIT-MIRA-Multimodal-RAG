"""Supervisor phải được cấp đúng bộ tool, và timeout không được làm sập lượt chat.

Sự cố gốc: `required_tools` chỉ liệt kê `search_regulations`, nên MCP server có
`search_media` mà agent không hề nhìn thấy. Mọi câu hỏi về ảnh/video đều bị
Mira từ chối bằng câu "em chưa có công cụ tra cứu ảnh/video trong phiên này" —
tức là một nửa đồ án đa phương thức không tới được người dùng, mà không có lỗi
nào được ném ra để ai đó nhận ra.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from src.graph.agents import base_agent as base_agent_module
from src.graph.agents.base_agent import BaseAgent
from src.graph.agents.supervisor_agent import (
    REQUIRED_TOOLS,
    SupervisorAgent,
    ToolResultCollector,
)


def test_supervisor_duoc_cap_ca_hai_tool() -> None:
    """Thiếu `search_media` là mất toàn bộ nhánh ảnh/video."""
    assert "search_regulations" in REQUIRED_TOOLS
    assert "search_media" in REQUIRED_TOOLS


def test_chi_lay_dung_nhung_tool_da_liet_ke(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool lạ trên MCP server không được tự động chui vào tay supervisor.

    Danh sách được khai báo tường minh để một tool mới thêm vào
    `mcp/Resources/tools.yaml` không lặng lẽ tới tay agent trước khi có ai viết
    hướng dẫn dùng nó trong system prompt.
    """
    tools_gia = [
        SimpleNamespace(name="search_regulations"),
        SimpleNamespace(name="search_media"),
        SimpleNamespace(name="xoa_toan_bo_du_lieu"),
    ]

    async def get_tools_gia(url: str, transport: str = "streamable_http"):
        return tools_gia

    monkeypatch.setattr(
        base_agent_module,
        "mcp_client",
        SimpleNamespace(get_tools=get_tools_gia),
        raising=True,
    )

    ket_qua = asyncio.run(
        BaseAgent.init_tools(url_mcp_servers="http://khong-ton-tai", required_tools=REQUIRED_TOOLS)
    )
    assert sorted(tool.name for tool in ket_qua) == sorted(REQUIRED_TOOLS)


def test_timeout_tra_ve_cau_tra_loi_thay_vi_nem_loi() -> None:
    """Quá giờ thì người dùng phải nhận được câu trả lời, không phải stack trace."""
    phan_hoi = SupervisorAgent._timeout_response(45)
    tin_nhan = phan_hoi["messages"][0]
    assert "45" in tin_nhan.content, "phải nói rõ đã chờ bao lâu"


def test_timeout_van_kem_theo_ket_qua_tho() -> None:
    """US-401.1 AC-2: chờ hết giờ rồi nhận về con số không là mất trắng công truy xuất.

    Phần truy xuất đã chạy xong và đã tốn quota nhúng — vứt đi là lãng phí đúng
    thứ đắt nhất của lượt hỏi.
    """
    collector = ToolResultCollector()
    collector.record(
        "search_media",
        json.dumps({"success": True, "context": "[1] video abc tại 02:00 — nguồn: https://fb.com/x"}),
    )

    phan_hoi = SupervisorAgent._timeout_response(45, collector)
    noi_dung = phan_hoi["messages"][0].content

    assert "45" in noi_dung, "vẫn phải báo lỗi nhẹ"
    assert "[1] video abc tại 02:00" in noi_dung, "phải kèm kết quả truy xuất thô"
    assert "chưa kịp tổng hợp" in noi_dung, "phải nói rõ đây chưa phải câu trả lời hoàn chỉnh"


def test_timeout_khong_co_gi_de_khoe_thi_chi_xin_loi() -> None:
    """Hết giờ trước cả khi gọi được tool: đừng dựng một mục 'nguồn' rỗng."""
    noi_dung = SupervisorAgent._timeout_response(30, ToolResultCollector())["messages"][0].content
    assert "nguồn" not in noi_dung.lower()
    assert "thử lại" in noi_dung.lower()


def test_collector_rut_dung_phan_nguoi_doc_duoc() -> None:
    """Tool trả JSON; thứ đáng đưa cho người dùng là `context` đã đánh số sẵn."""
    collector = ToolResultCollector()
    collector.record("search_regulations", json.dumps({"success": True, "context": "[1] Nội quy CLB, trang 2"}))
    collector.record("tool_la", "không phải JSON")
    collector.record("tool_rong", "   ")

    ket_qua = collector.as_text()
    assert "[1] Nội quy CLB, trang 2" in ket_qua
    assert "không phải JSON" in ket_qua
    assert "tool_rong" not in ket_qua, "kết quả rỗng thì không ghi gì cả"
