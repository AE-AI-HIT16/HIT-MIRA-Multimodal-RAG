import asyncio

import httpx
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from src.graph.configuration import Context
from src.graph.agents.base_agent import BaseAgent
from src.config.configs import config_object


class SupervisorAgent(BaseAgent):
    def __init__(self,
                 system_prompt_path: str,
                 runtime: Runtime[Context]):
        super().__init__(system_prompt_path, runtime)

    async def ainvoke(self, state, runtime, config=None):
        input_data = {
            "messages": state.messages,
            "memories": state.memories
        }
        if self.agent is None:
            await self.setup_agent(
                url_mcp_servers=config_object.MCP.BASE_MCP_SERVER_URL,
                required_tools=["search_regulations"]
            )
        try:
            response = await asyncio.wait_for(
                self.agent.ainvoke(input=input_data, runtime=runtime.context, config=config),
                timeout=runtime.context.response_timeout,
            )
        except asyncio.TimeoutError:
            response = self._timeout_response(runtime.context.response_timeout)
        except httpx.TimeoutException:
            response = self._timeout_response(runtime.context.llm_config.timeout)
        return response

    @staticmethod
    def _timeout_response(timeout: float):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Xin lỗi, yêu cầu xử lý quá lâu nên hệ thống đã dừng sau "
                        f"{timeout:g} giây. "
                        "Vui lòng thử lại hoặc đặt câu hỏi cụ thể hơn."
                    )
                )
            ]
        }
