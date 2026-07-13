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
    base_url: Optional[str] = None

class Context(BaseModel):
    conversation_id: Optional[str] = None
    customer_id: Optional[str] = None
    attachments: Optional[list[dict]] = None
    llm_config: Optional[LLMConfig] = LLMConfig()
