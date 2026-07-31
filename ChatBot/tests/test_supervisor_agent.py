"""Supervisor phải được cấp đúng bộ tool, và timeout không được làm sập lượt chat.

Sự cố gốc: `required_tools` chỉ liệt kê `search_regulations`, nên MCP server có
`search_media` mà agent không hề nhìn thấy. Mọi câu hỏi về ảnh/video đều bị
Mira từ chối bằng câu "em chưa có công cụ tra cứu ảnh/video trong phiên này" —
tức là một nửa đồ án đa phương thức không tới được người dùng, mà không có lỗi
nào được ném ra để ai đó nhận ra.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.graph.agents import base_agent as base_agent_module
from src.graph.agents.base_agent import BaseAgent
from src.graph.agents.supervisor_agent import REQUIRED_TOOLS, SupervisorAgent


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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "US-401.1 AC-2 chưa làm: khi timeout, fallback phải kèm kết quả truy xuất thô "
        "chứ không chỉ một lời xin lỗi. Khi ai đó làm xong, test này sẽ XPASS và "
        "bắt buộc phải gỡ marker — đó là ý đồ."
    ),
)
def test_timeout_van_kem_theo_ket_qua_tho() -> None:
    """US-401.1 AC-2: chờ 60 giây rồi nhận về con số không là mất trắng công truy xuất."""
    phan_hoi = SupervisorAgent._timeout_response(45)
    tin_nhan = phan_hoi["messages"][0]
    assert phan_hoi.get("retrieved") or "nguồn" in tin_nhan.content.lower()
