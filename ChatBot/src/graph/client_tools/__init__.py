from src.graph.client_tools.base_mcp_client import BaseMCPClient
from src.log import logger

try:
    mcp_client = BaseMCPClient()
    logger.info("---INIT MCPClientService successful---")
except Exception as e:
    logger.error("---INIT MCPClientService Unsuccessful---")
    logger.error(e)
