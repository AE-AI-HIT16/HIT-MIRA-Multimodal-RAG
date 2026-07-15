from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from typing import Any

from src.log.logger import logger


class DocumentCleaner:
    """Clean parsed documents without changing their meaning or metadata."""

    def clean_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(documents, list):
            raise TypeError("documents must be a list[dict]")

        cleaned_documents: list[dict[str, Any]] = []
        for index, document in enumerate(documents):
            if not isinstance(document, dict):
                raise TypeError(f"documents[{index}] must be a dict")
            text = document.get("text", "")
            if not isinstance(text, str):
                raise TypeError(f"documents[{index}]['text'] must be a string")

            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                continue

            cleaned_document = deepcopy(document)
            cleaned_document["text"] = cleaned_text
            cleaned_documents.append(cleaned_document)

        logger.info(f"Cleaned {len(cleaned_documents)} non-empty document unit(s)")
        return cleaned_documents

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
