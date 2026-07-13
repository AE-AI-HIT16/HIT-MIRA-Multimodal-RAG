"""Registry tool — router tra tool theo tên."""
from __future__ import annotations

from .base import BaseTool

_REGISTRY: dict[str, BaseTool] = {}


def register(tool: BaseTool) -> BaseTool:
    _REGISTRY[tool.name] = tool
    return tool


def get(name: str) -> BaseTool:
    return _REGISTRY[name]


def all_tools() -> dict[str, BaseTool]:
    return dict(_REGISTRY)
