"""Test tool wiring — T-34. tool.run trả ToolResult đúng nguồn (monkeypatch retrieval)."""
from __future__ import annotations

from app.domains.retrieval import service as retrieval
from app.tools.media_tool import MediaTool
from app.tools.regulation_tool import RegulationTool


def test_media_tool_returns_media_result(monkeypatch):
    monkeypatch.setattr(retrieval, "retrieve_media",
                        lambda q, top_k=5: [{"id": 1, "score": 0.9, "video_id": 7}])
    out = MediaTool().run("ảnh khai giảng")
    assert out.source == "media"
    assert out.items[0]["video_id"] == 7


def test_regulation_tool_returns_regulation_result(monkeypatch):
    monkeypatch.setattr(retrieval, "retrieve_regulations",
                        lambda q, top_k=5: [{"id": 1, "score": 0.8, "article": 5}])
    out = RegulationTool().run("được phép không")
    assert out.source == "regulation"
    assert out.items[0]["article"] == 5


def test_tool_applies_threshold(monkeypatch):
    monkeypatch.setattr(retrieval, "retrieve_media",
                        lambda q, top_k=5: [{"id": 1, "score": 0.1}])
    out = MediaTool().run("x", threshold=0.5)
    assert out.items == []                      # dưới ngưỡng → rỗng (không tìm thấy)
