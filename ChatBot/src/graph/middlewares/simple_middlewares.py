from typing import Callable

from langchain.agents.middleware import AgentMiddleware, wrap_model_call, ModelResponse
from src.graph.state import State
from src.common_utils.time_utils import get_current_date_info
from langchain.agents.middleware import ModelRequest


class SafeDict(dict):
    def __missing__(self, key):
        # Khi key không có trong dict, giữ nguyên placeholder
        return "{" + key + "}"


class MemoryMiddleware(AgentMiddleware):
    async def abefore_agent(self, state, runtime):
        return {}


@wrap_model_call(state_schema=State)
async def inject_config_prompt(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    base_prompt = request.system_prompt.format_map(SafeDict(request.state))
    current_time, date_of_time = get_current_date_info()
    time_prompt = f"## Thời gian hiện tại\t- {current_time}\t- {date_of_time}"

    new_content = f"{base_prompt}\n{time_prompt}"

    return await handler(request.override(system_prompt=new_content))

from langchain.agents.middleware import wrap_tool_call

@wrap_tool_call
async def ToolStandardizationMiddleware(request, handler):
    """Chuẩn hóa dữ liệu giữa output của LLM với input của tool."""
    # Tiền xử lý (chuẩn hóa đầu vào của tool)
    modified_args = {}
    for k, v in request.tool_call["args"].items():
        if isinstance(v, str):
            modified_args[k] = v.strip() # Ví dụ: chuẩn hóa bằng cách strip khoảng trắng
        else:
            modified_args[k] = v

    modified_call = {
        **request.tool_call,
        "args": modified_args
    }
    modified_request = request.override(tool_call=modified_call)
    
    # Thực thi tool
    response = await handler(modified_request)
    
    # Hậu xử lý (chuẩn hóa đầu ra của tool)
    # response có thể là ToolMessage, trả về nguyên trạng hoặc format lại nếu cần
    return response
