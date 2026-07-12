"""Tiện ích tools cho ChatBot."""

from src.log import logger


def log_tool_call(tool_name: str, tool_args: dict):
    """Log thông tin tool call."""
    logger.info(f"Tool call: {tool_name} | args: {tool_args}")
