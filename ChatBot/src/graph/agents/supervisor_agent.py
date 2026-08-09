import asyncio
import json

import httpx
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from src.config.configs import config_object
from src.graph.agents.base_agent import BaseAgent
from src.graph.configuration import Context

# Hai tool này là toàn bộ khả năng tra cứu của Mira: nội quy và kho media.
# Liệt kê tường minh thay vì lấy hết tool của MCP server, để một tool mới thêm
# vào Resources/tools.yaml không tự động chui vào tay supervisor mà chưa có ai
# viết hướng dẫn dùng nó trong system prompt.
REQUIRED_TOOLS = ["search_regulations", "search_media"]

# Cắt bớt phần thô cho vừa một câu trả lời đọc được, không phải một bãi JSON.
MAX_RAW_SNIPPETS = 3
MAX_SNIPPET_CHARS = 800


class ToolResultCollector:
    """Giữ lại kết quả tool của lượt hiện tại.

    US-401.1 AC-2: khi hết giờ, người dùng phải nhận được kết quả truy xuất thô
    kèm một lời báo lỗi nhẹ. Nếu không giữ gì, phần truy xuất đã chạy xong (và
    đã tốn quota nhúng) bị vứt đi sạch, người dùng chờ hết `response_timeout`
    rồi nhận về đúng một lời xin lỗi.

    Mỗi lượt gọi `SupervisorAgent.ainvoke` dùng một collector riêng, vì node
    dựng agent mới cho từng lượt.
    """

    def __init__(self) -> None:
        self.snippets: list[tuple[str, str]] = []

    def record(self, tool_name: str, content) -> None:
        text = self._readable(content)
        if text:
            self.snippets.append((tool_name or "tool", text))

    @staticmethod
    def _readable(content) -> str:
        """Rút phần người đọc được ra khỏi kết quả tool.

        Tool trả về JSON. `context` là chuỗi trích dẫn đã đánh số sẵn
        `[1] [2] [3]` — đúng thứ cần đưa cho người dùng xem. Không phân tích
        được thì lấy nguyên văn, cắt ngắn.
        """
        if content is None:
            return ""
        if not isinstance(content, str):
            content = str(content)
        text = content.strip()
        if not text:
            return ""
        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            return text[:MAX_SNIPPET_CHARS]
        if isinstance(payload, dict):
            context = str(payload.get("context") or "").strip()
            if context:
                return context[:MAX_SNIPPET_CHARS]
        return text[:MAX_SNIPPET_CHARS]

    def as_text(self) -> str:
        if not self.snippets:
            return ""
        lines: list[str] = []
        for tool_name, snippet in self.snippets[:MAX_RAW_SNIPPETS]:
            lines.append(f"Từ {tool_name}:\n{snippet}")
        return "\n\n".join(lines)


def build_tool_capture_middleware(collector: ToolResultCollector):
    """Middleware chỉ ghi lại kết quả tool, không đổi gì trên đường đi."""

    @wrap_tool_call
    async def capture_tool_result(request, handler):
        response = await handler(request)
        try:
            tool_name = (request.tool_call or {}).get("name", "")
            collector.record(tool_name, getattr(response, "content", None))
        except Exception:
            # Ghi hụt một kết quả thì tệ, nhưng làm hỏng lượt chat còn tệ hơn.
            pass
        return response

    return capture_tool_result


class SupervisorAgent(BaseAgent):
    def __init__(self,
                 system_prompt_path: str,
                 runtime: Runtime[Context]):
        super().__init__(system_prompt_path, runtime)
        self.tool_results = ToolResultCollector()

    async def ainvoke(self, state, runtime, config=None):
        input_data = {
            "messages": state.messages,
            "memories": state.memories
        }
        if self.agent is None:
            await self.setup_agent(
                url_mcp_servers=config_object.MCP.BASE_MCP_SERVER_URL,
                required_tools=REQUIRED_TOOLS,
                middlewares=[build_tool_capture_middleware(self.tool_results)],
            )
        try:
            response = await asyncio.wait_for(
                self.agent.ainvoke(input=input_data, runtime=runtime.context, config=config),
                timeout=runtime.context.response_timeout,
            )
        except asyncio.TimeoutError:
            response = self._timeout_response(runtime.context.response_timeout, self.tool_results)
        except httpx.TimeoutException:
            response = self._timeout_response(runtime.context.llm_config.timeout, self.tool_results)
        return response

    @staticmethod
    def _timeout_response(timeout: float, collector: "ToolResultCollector | None" = None):
        apology = (
            "Xin lỗi, yêu cầu xử lý quá lâu nên hệ thống đã dừng sau "
            f"{timeout:g} giây. "
        )
        raw = collector.as_text() if collector is not None else ""
        if not raw:
            content = apology + "Vui lòng thử lại hoặc đặt câu hỏi cụ thể hơn."
        else:
            # Nêu rõ đây là kết quả thô chưa tổng hợp, để không bị đọc nhầm
            # thành câu trả lời hoàn chỉnh.
            content = (
                apology
                + "Em chưa kịp tổng hợp thành câu trả lời, nhưng đây là các nguồn "
                "em đã tìm được:\n\n"
                + raw
            )
        return {"messages": [AIMessage(content=content)]}
