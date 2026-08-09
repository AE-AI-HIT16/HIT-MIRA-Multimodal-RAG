import asyncio
import json

import httpx
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from src.config.configs import config_object
from src.graph.agents.base_agent import BaseAgent
from src.graph.configuration import Context
from src.graph.utils import load_sys_prompt

# Hai tool này là toàn bộ khả năng tra cứu của Mira: nội quy và kho media.
# Liệt kê tường minh thay vì lấy hết tool của MCP server, để một tool mới thêm
# vào Resources/tools.yaml không tự động chui vào tay supervisor mà chưa có ai
# viết hướng dẫn dùng nó trong system prompt.
REQUIRED_TOOLS = ["search_regulations", "search_media"]

# Ép chế độ thủ công (US-507.1 Edge / TC-507) bằng cách CẮT DANH SÁCH TOOL, chứ
# không phải bằng một câu nhắc trong prompt. Nhắc thì model vẫn có thể gọi tool
# kia khi nó thấy hợp lý hơn — mà "ép" nghĩa là người dùng đã nói router chọn
# sai rồi, nên để model tự quyết lần nữa là bỏ qua đúng thứ họ vừa sửa. Không
# cầm tool thì không gọi được: đó là bảo đảm, không phải gợi ý.
TOOL_THEO_CHE_DO: dict[str, list[str]] = {
    "media": ["search_media"],
    "regulation": ["search_regulations"],
    # "both" giữ cả hai tool — cắt tool không ép được việc gọi ĐỦ hai, nên phần
    # đó phải nhờ prompt bên dưới.
    "both": REQUIRED_TOOLS,
}

# Phần nối thêm vào system prompt cho mỗi chế độ.
NHAC_THEO_CHE_DO: dict[str, str] = {
    "media": (
        "## Phạm vi bị giới hạn cho lượt này\n"
        "Người dùng đã chọn **chỉ tìm trong kho ảnh/video**. Bạn chỉ có tool "
        "`search_media`. Nếu câu hỏi thuộc về nội quy, hãy nói rõ là lượt này "
        "chỉ tra tư liệu ảnh/video và mời họ chuyển sang chế độ Nội quy — "
        "TUYỆT ĐỐI không trả lời nội quy từ trí nhớ."
    ),
    "regulation": (
        "## Phạm vi bị giới hạn cho lượt này\n"
        "Người dùng đã chọn **chỉ tra nội quy**. Bạn chỉ có tool "
        "`search_regulations`. Nếu câu hỏi thuộc về ảnh/video, hãy nói rõ là "
        "lượt này chỉ tra nội quy và mời họ chuyển sang chế độ Ảnh & video — "
        "TUYỆT ĐỐI không mô tả ảnh/video từ trí nhớ."
    ),
    "both": (
        "## Phạm vi bị giới hạn cho lượt này\n"
        "Người dùng đã chọn **tra cả hai nguồn**. Hãy gọi CẢ `search_media` và "
        "`search_regulations` rồi tổng hợp, kể cả khi bạn nghĩ một nguồn là đủ."
    ),
}

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
            che_do = self._che_do_hop_le(getattr(runtime.context, "override", None))
            if che_do:
                # Nạp prompt sẵn ở đây rồi nối phần nhắc: `setup_agent` chỉ nạp
                # khi `system_prompt` còn None, nên gán trước là điểm móc có sẵn.
                self.system_prompt = await load_sys_prompt(self.system_prompt_path)
                self.system_prompt = f"{self.system_prompt}\n\n{NHAC_THEO_CHE_DO[che_do]}"
            await self.setup_agent(
                url_mcp_servers=config_object.MCP.BASE_MCP_SERVER_URL,
                required_tools=TOOL_THEO_CHE_DO.get(che_do, REQUIRED_TOOLS),
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
    def _che_do_hop_le(gia_tri) -> str | None:
        """Chuẩn hoá `override`, trả `None` cho mọi thứ không nhận ra.

        Giá trị lạ rơi về định tuyến tự động chứ không ném lỗi: đây là tuỳ chọn
        giao diện, và làm hỏng cả lượt chat vì một chuỗi gõ sai thì đắt hơn
        nhiều so với việc lặng lẽ dùng đường chính.
        """
        if not isinstance(gia_tri, str):
            return None
        che_do = gia_tri.strip().lower()
        return che_do if che_do in TOOL_THEO_CHE_DO else None

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
