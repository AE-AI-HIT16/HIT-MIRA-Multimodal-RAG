from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from importlib import import_module
from typing import Any

from src.configuration import AppConfig
from src.log.logger import logger


class StructureAwareChunker:
    """Structure-aware chunker using LangChain's recursive splitter for overflow text."""

    HEADING_RE = re.compile(
        r"^(Chuong|Muc|Dieu|Khoan|Chương|Mục|Điều|Khoản|\d+(?:\.\d+)*[\).]?|"
        r"Doi voi|Đối với|Hinh thuc xu phat|Hình thức xử phạt)\b",
        re.IGNORECASE,
    )
    CATEGORY_RE = re.compile(
        r"^(Chương|CHƯƠNG|Mục|MỤC|Điều|ĐIỀU|Khoản|KHOẢN)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        minimum_chunk_size: int | None = None,
        config: AppConfig | None = None,
    ) -> None:
        pipeline_config = (config or AppConfig()).pipeline
        if pipeline_config is None:
            raise ValueError("pipeline config is not available")
        self.chunk_size = int(chunk_size or pipeline_config.chunk_size or 1200)
        configured_overlap = pipeline_config.chunk_overlap or 150
        self.chunk_overlap = int(chunk_overlap if chunk_overlap is not None else configured_overlap)
        self.minimum_chunk_size = int(minimum_chunk_size or pipeline_config.minimum_chunk_size or 250)
        self._validate_config()
        self.splitter = self._build_splitter()

    def chunk_documents(
        self,
        documents: list[dict[str, Any]],
        document_id: str,
        filename: str | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        if not documents:
            raise ValueError("documents must not be empty")

        units = self._section_documents(documents)
        split_units = self.splitter.split_documents(units)
        merged_units = self._merge_small_units(split_units)
        chunks = [
            self._format_chunk(unit, document_id, filename, index)
            for index, unit in enumerate(merged_units)
        ]
        logger.info(f"Created {len(chunks)} chunk(s) for document_id '{document_id}'")
        return chunks

    def _build_splitter(self) -> Any:
        splitter_module = import_module("langchain_text_splitters")
        splitter_cls = getattr(splitter_module, "RecursiveCharacterTextSplitter")
        return splitter_cls(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "; ", " ", ""],
        )

    @staticmethod
    def _document(page_content: str, metadata: dict[str, Any]) -> Any:
        document_module = import_module("langchain_core.documents")
        document_cls = getattr(document_module, "Document")
        return document_cls(page_content=page_content, metadata=metadata)

    def _section_documents(self, documents: list[Any]) -> list[Any]:
        section: str | None = None
        units: list[Any] = []
        for document in documents:
            langchain_doc = self._as_document(document)
            metadata = dict(langchain_doc.metadata or {})
            base_metadata = {
                **metadata,
                "start_page": metadata.get("page"),
                "end_page": metadata.get("page"),
                "source": metadata.get("source"),
            }
            for paragraph in self._paragraphs(langchain_doc.page_content):
                first_line = paragraph.split("\n", 1)[0].strip()
                if self._is_heading(first_line):
                    section = first_line
                units.append(self._document(paragraph, {**base_metadata, "section": section}))
        return units

    def _as_document(self, document: Any) -> Any:
        if hasattr(document, "page_content") and hasattr(document, "metadata"):
            return document
        if isinstance(document, dict):
            metadata = dict(document.get("metadata") or {})
            metadata.setdefault("page", document.get("page"))
            metadata.setdefault("source", document.get("source"))
            return self._document(str(document.get("text", "")), metadata)
        raise TypeError("documents must contain LangChain Document or dict items")

    @staticmethod
    def _paragraphs(text: str) -> list[str]:
        return [paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if paragraph.strip()]

    def _merge_small_units(self, units: list[Any]) -> list[Any]:
        merged: list[Any] = []
        for unit in units:
            text = unit.page_content.strip()
            if not text:
                continue
            if not merged or len(merged[-1].page_content) >= self.minimum_chunk_size:
                merged.append(unit)
                continue

            previous = merged[-1]
            combined = f"{previous.page_content}\n\n{text}"
            if len(combined) <= self.chunk_size:
                merged[-1] = self._document(
                    combined,
                    {
                        **previous.metadata,
                        "end_page": (
                            unit.metadata.get("end_page")
                            or previous.metadata.get("end_page")
                        ),
                    },
                )
            else:
                merged.append(unit)
        return merged

    def _format_chunk(
        self,
        unit: Any,
        document_id: str,
        filename: str | None,
        chunk_index: int,
    ) -> dict[str, Any]:
        text = unit.page_content.strip()
        metadata = dict(unit.metadata or {})
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        category, subject = self._infer_category_subject(metadata)
        source = metadata.get("source") or filename
        return {
            "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{chunk_index}:{content_hash}")),
            "document_id": document_id,
            "text": text,
            "page": metadata.get("start_page"),
            "start_page": metadata.get("start_page"),
            "end_page": metadata.get("end_page"),
            "section": metadata.get("section"),
            "category": category,
            "subject": subject,
            "chunk_index": chunk_index,
            "source": source,
            "filename": filename or source,
            "metadata": {**metadata, "category": category, "subject": subject},
        }

    def _is_heading(self, line: str) -> bool:
        return bool(self.HEADING_RE.match(line) or self.HEADING_RE.match(self._strip_accents(line)))

    @staticmethod
    def _strip_accents(value: str) -> str:
        normalized = unicodedata.normalize("NFD", value)
        return "".join(char for char in normalized if unicodedata.category(char) != "Mn")

    def _infer_category_subject(self, metadata: dict[str, Any]) -> tuple[str | None, str | None]:
        if metadata.get("category") or metadata.get("subject"):
            return metadata.get("category"), metadata.get("subject")
        section = metadata.get("section")
        if not section:
            return None, None
        category_match = self.CATEGORY_RE.match(str(section).strip())
        return category_match.group(1) if category_match else None, section

    def _validate_config(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        if self.minimum_chunk_size < 0:
            raise ValueError("minimum_chunk_size must be >= 0")
