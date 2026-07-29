from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langfuse import observe

from src.config.config import config_prompts, resolve_project_path
from src.configuration import AppConfig
from src.log.logger import logger


class QueryRewriter:
    """Rewrite user questions into search-friendly queries."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        prompt_path = resolve_project_path(config_prompts.PROMPT.PROMPT_REWRITING)
        self.chain = (
            PromptTemplate.from_file(prompt_path)
            | self._llm()
            | StrOutputParser()
        )

    @observe(name="query_rewrite")
    def rewrite(self, query: str) -> str:
        query = self._normalize_query(query)
        try:
            return self.chain.invoke({"query": query}).strip() or query
        except Exception as exc:
            logger.warning(f"Query rewrite failed, using original query: {exc.__class__.__name__}")
            return query

    def _llm(self):
        llm_config = self.config.llm
        kwargs = {
            "temperature": llm_config.temperature,
            "timeout": llm_config.timeout,
            "max_retries": llm_config.max_retries,
        }
        if llm_config.max_tokens is not None:
            kwargs["max_tokens"] = llm_config.max_tokens
        if llm_config.base_url:
            kwargs["base_url"] = llm_config.base_url
        if llm_config.api_key:
            kwargs["api_key"] = llm_config.api_key
        return init_chat_model(llm_config.model_name, **kwargs)

    @staticmethod
    def _normalize_query(query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        return " ".join(query.split())
