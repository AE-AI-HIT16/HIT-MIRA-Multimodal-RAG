"""Đồ thị phải dựng được ngay cả khi đã có sẵn một vòng lặp sự kiện.

Bản trước của `init_root_graph` khai báo `async` rồi tự gọi `asyncio.run(...)`
ngay lúc import. Hậu quả: mọi tiến trình đã có event loop — LangGraph server,
FastAPI, notebook, hay chỉ một script `asyncio.run` — đều chết ngay dòng
import với "asyncio.run() cannot be called from a running event loop".

Nhập module trong một hàm đồng bộ thì không bao giờ lộ ra lỗi này, nên test
phải nhập *bên trong* vòng lặp mới đúng hiện trường.
"""

from __future__ import annotations

import asyncio
import importlib
import sys

MODULE_NAME = "src.graph.graph"


def _import_lai_sach() -> None:
    """Xoá module khỏi cache để lần import sau chạy lại từ đầu."""
    for name in list(sys.modules):
        if name == MODULE_NAME or name.startswith(MODULE_NAME + "."):
            del sys.modules[name]


def test_import_duoc_trong_vong_lap_su_kien() -> None:
    """Import giữa lúc event loop đang chạy: không được ném lỗi."""

    async def nhap_module():
        _import_lai_sach()
        return importlib.import_module(MODULE_NAME)

    module = asyncio.run(nhap_module())
    assert module.graph is not None, "đồ thị phải được biên dịch sẵn khi import"


def test_do_thi_co_dung_nut_supervisor() -> None:
    """Đồ thị v1 chỉ có một nút xử lý; điểm vào phải trỏ vào nó."""
    from src.config.configs import config_agents

    module = importlib.import_module(MODULE_NAME)
    ten_nut = config_agents.PROFILE_GRAPH.AGENT_SUPERVISOR
    assert ten_nut in module.graph.nodes


def test_init_root_graph_la_ham_dong_bo() -> None:
    """Chốt lại bản chất của cách sửa: hàm không được là coroutine."""
    module = importlib.import_module(MODULE_NAME)
    assert not asyncio.iscoroutinefunction(module.init_root_graph), (
        "init_root_graph phải đồng bộ — thân hàm không await gì cả, "
        "để async lại là mở đường cho asyncio.run() lúc import quay về"
    )
