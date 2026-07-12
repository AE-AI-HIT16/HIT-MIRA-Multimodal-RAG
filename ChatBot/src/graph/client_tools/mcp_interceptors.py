from langchain_mcp_adapters.interceptors import MCPToolCallRequest

async def inject_langgraph_runtime(
    request: MCPToolCallRequest,
    handler,
):
    """
    Interceptor pass-through: forward tool call đến MCP server.
    Không inject thêm field vào args vì các tool MCP (search_archives,
    get_profile_detail) chỉ nhận đúng các parameter đã khai báo trong
    tools.yaml — inject field lạ sẽ gây validation error tại FastMCP.
    """
    return await handler(request)
