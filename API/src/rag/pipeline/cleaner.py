from __future__ import annotations

import re
import unicodedata
from importlib import import_module
from typing import Any

from src.log.logger import logger


class DocumentCleaner:
    """Clean LangChain Documents without changing meaning or metadata."""

    def clean_documents(self, documents: list[Any]) -> list[Any]:
        if not isinstance(documents, list):
            raise TypeError("documents must be a list")

        cleaned_documents: list[Any] = []
        for index, document in enumerate(documents):
            normalized_document = self._as_document(document, index)
            cleaned_text = self.clean_text(normalized_document.page_content)
            if cleaned_text:
                cleaned_documents.append(
                    self._document(cleaned_text, dict(normalized_document.metadata or {}))
                )

        logger.info(f"Cleaned {len(cleaned_documents)} non-empty document unit(s)")
        return cleaned_documents

    @staticmethod
    def _document(page_content: str, metadata: dict[str, Any]) -> Any:
        document_cls = getattr(import_module("langchain_core.documents"), "Document")
        return document_cls(page_content=page_content, metadata=metadata)

    def _as_document(self, document: Any, index: int) -> Any:
        if hasattr(document, "page_content") and hasattr(document, "metadata"):
            return document
        if isinstance(document, dict):
            text = document.get("text", "")
            if not isinstance(text, str):
                raise TypeError(f"documents[{index}]['text'] must be a string")
            metadata = dict(document.get("metadata") or {})
            metadata.setdefault("page", document.get("page"))
            metadata.setdefault("source", document.get("source"))
            return self._document(text, metadata)
        raise TypeError(f"documents[{index}] must be a LangChain Document or dict")

    @staticmethod
    def clean_text(text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[\t\f\v]+", " ", text)

        lines: list[str] = []
        previous_blank = False
        for raw_line in text.split("\n"):
            line = re.sub(r"[ ]{2,}", " ", raw_line).strip()
            if not line:
                if not previous_blank and lines:
                    lines.append("")
                previous_blank = True
                continue
            lines.append(line)
            previous_blank = False

        return "\n".join(lines).strip()
