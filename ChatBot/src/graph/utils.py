"""Utility & helper functions."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from src.graph.configuration import LLMConfig
from src.config.configs import config_object


def load_sys_prompt(prompt_path: str) -> str:
    """Đọc system prompt từ file."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    return system_prompt


def load_model(llm_config: LLMConfig) -> BaseChatModel:
    """Initialize the configured chat model."""
    if ":" in llm_config.model_name:
        provider, model = llm_config.model_name.split(":", maxsplit=1)
    else:
        provider = None
        model = llm_config.model_name
    llm_model = init_chat_model(
        model=model,
        model_provider=provider,
        temperature=llm_config.temperature,
        timeout=llm_config.timeout,
        max_retries=llm_config.max_retries,
        max_tokens=llm_config.max_tokens
    )
    return llm_model
