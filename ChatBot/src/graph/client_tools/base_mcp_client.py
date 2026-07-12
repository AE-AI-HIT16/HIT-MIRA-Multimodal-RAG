from langchain_mcp_adapters.client import MultiServerMCPClient

from src.graph.client_tools.mcp_interceptors import inject_langgraph_runtime


class BaseMCPClient:
    """
    Base class for MCP clients.
    """

    @staticmethod
    async def get_tools(url: str, transport: str = "streamable_http"):
        client = MultiServerMCPClient(
            {
                "MCP Server":
                    {
                        "url": url,
                        "transport": transport,
                        "timeout": 600,
                        "sse_read_timeout": 600,
                    }
            },
            tool_interceptors=[inject_langgraph_runtime]
        )
        tools = await client.get_tools()
        return tools
