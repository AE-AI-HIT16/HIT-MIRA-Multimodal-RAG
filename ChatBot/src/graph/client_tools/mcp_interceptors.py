from langchain_mcp_adapters.interceptors import MCPToolCallRequest

async def inject_langgraph_runtime(
    request: MCPToolCallRequest,
    handler,
):
    return await handler(request)
