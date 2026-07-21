"""
Entry point MCP server. Chạy: python src/server.py
"""
import asyncio

from config.configs import config_object
from tools.registry import ToolRegistry
from logger import get_logger

logger = get_logger(__name__)

mcp = ToolRegistry(name=config_object.MCP.SERVER_NAME)


async def main():
    server_mcp = await mcp.register_tools(category="mcp")
    return server_mcp


if __name__ == "__main__":
    server_mcp = asyncio.run(main())
    logger.info("Máy chủ MCP đã sẵn sàng và bắt đầu lắng nghe yêu cầu...")
    server_mcp.run(transport=config_object.MCP.TRANSPORT)
