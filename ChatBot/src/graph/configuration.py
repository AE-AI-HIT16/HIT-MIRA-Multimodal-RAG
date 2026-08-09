"""Define the configurable parameters for the chat bot."""

from typing import Optional
from pydantic import BaseModel

from src.config.configs import config_models


class LLMConfig(BaseModel):
    """The configurable fields for the model llm."""
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 0.1
    max_tokens: Optional[int] = None
    model_name: Optional[str] = config_models.OPENAI_LLM_MODEL.MODEL_PATH
    timeout: Optional[float] = 30
    max_retries: Optional[int] = 3
    base_url: Optional[str] = getattr(config_models.OPENAI_LLM_MODEL, "BASE_URL", None)
    api_key: Optional[str] = getattr(config_models.OPENAI_LLM_MODEL, "API_KEY", None)

class Context(BaseModel):
    conversation_id: Optional[str] = None
    customer_id: Optional[str] = None
    attachments: Optional[list[dict]] = None
    response_timeout: Optional[float] = 120
    llm_config: Optional[LLMConfig] = LLMConfig()
    # Ép chế độ thủ công — US-507.1 Edge / TC-507. `None` là đường chính: router
    # tự chọn nguồn. Ba giá trị hợp lệ: "media", "regulation", "both".
    #
    # Trường này TỪNG THIẾU trong khi giao diện vẫn gửi `configurable.override`
    # mỗi lượt, nên nút chọn chế độ trông như hoạt động mà thực ra bị bỏ qua
    # hoàn toàn — gửi override="media" cho câu hỏi nội quy vẫn gọi
    # search_regulations. Kiểu hỏng khó thấy nhất: không lỗi, không cảnh báo.
    override: Optional[str] = None
