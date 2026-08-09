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
    """Chunk documents by section, paragraph, and sentence boundaries."""

    HEADING_RE = re.compile(
        r"^(Chuong|Muc|Dieu|Khoan|Chương|Mục|Điều|Khoản|Nội quy|Nội Quy|Quy định|Quy Định|"
        r"\d+(?:\.\d+)*[\).]?|Doi voi|Đối với|Hinh thuc xu phat|Hình thức xử phạt)\b",
        re.IGNORECASE,
    )
    CATEGORY_RE = re.compile(
        r"^(Chương|CHƯƠNG|Mục|MỤC|Điều|ĐIỀU|Khoản|KHOẢN)\b",
        re.IGNORECASE,
    )
    # `HEADING_RE` chỉ khớp phần ĐẦU dòng, nên thiếu chặn độ dài thì cả một đoạn
    # dài mở đầu bằng "Nội quy..." cũng bị nhận là tiêu đề mục — và vì tên mục
    # được ghép vào trước mọi chunk của mục đó, văn bản phình lên theo cấp số.
    # Đo 07/08/2026 trên "Nội Quy CLB 2022.docx": Docx2txtLoader không giữ dòng
    # trống giữa các gạch đầu dòng, cho ra một đoạn 1.461 ký tự; nó thành tên
    # mục và 1.697 ký tự tài liệu nở thành 73 chunk / 107.220 ký tự — gấp 63
    # lần, tức 63 lần tiền nhúng và một collection đầy chữ lặp. Bản PDF cùng nội
    # dung ra 4 chunk / 2.076 ký tự.
    MAX_HEADING_CHARS = 120

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
        self.chunk_overlap = int(
            chunk_overlap if chunk_overlap is not None else configured_overlap
        )
        self.minimum_chunk_size = int(
            minimum_chunk_size or pipeline_config.minimum_chunk_size or 250
        )
        self._validate_config()

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
        merged_units = self._pack_semantic_units(units)
        chunks = [
            self._format_chunk(unit, document_id, filename, index)
            for index, unit in enumerate(merged_units)
        ]
        logger.info(
            f"Created {len(chunks)} semantic chunk(s) for document_id '{document_id}'"
        )
        return chunks

    @staticmethod
    def _document(page_content: str, metadata: dict[str, Any]) -> Any:
        document_module = import_module("langchain_core.documents")
        document_cls = getattr(document_module, "Document")
        return document_cls(page_content=page_content, metadata=metadata)

    def _section_documents(self, documents: list[Any]) -> list[Any]:
        section: str | None = None
        document_title: str | None = None
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
                is_heading = self._is_heading(first_line) and not self._is_clause_item(
                    first_line, section
                )
                if is_heading:
                    if self._is_document_title(first_line):
                        document_title = first_line
                        section = first_line
                    elif document_title and re.match(r"^\d{1,3}[.)]\s+", first_line):
                        section = f"{document_title} — {first_line}"
                    else:
                        section = first_line
                    # A standalone heading is metadata/context, never a vector
                    # by itself. Its text is prefixed to the following content.
                    if self._is_heading_only(paragraph):
                        continue
                for part in self._split_at_meaningful_boundaries(paragraph, section):
                    units.append(
                        self._document(part, {**base_metadata, "section": section})
                    )
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
        return [
            paragraph.strip()
            for paragraph in re.split(r"\n{2,}", text)
            if paragraph.strip()
        ]

    def _pack_semantic_units(self, units: list[Any]) -> list[Any]:
        """Pack complete paragraphs/sentences; never use character-based splitting."""
        chunks: list[Any] = []
        current: list[Any] = []
        current_section: str | None = None

        def emit(items: list[Any], section: str | None) -> None:
            if not items:
                return
            body = "\n\n".join(item.page_content.strip() for item in items).strip()
            prefix = (
                f"{section}\n\n" if section and not body.startswith(section) else ""
            )
            first, last = items[0], items[-1]
            chunks.append(
                self._document(
                    f"{prefix}{body}",
                    {
                        **first.metadata,
                        "section": section,
                        "start_page": first.metadata.get("start_page"),
                        "end_page": last.metadata.get("end_page")
                        or first.metadata.get("end_page"),
                    },
                )
            )

        for unit in units:
            section = unit.metadata.get("section")
            # Do not mix the end of one article with the next article.
            if current and section != current_section:
                emit(current, current_section)
                current = []
            current_section = section
            if (
                not current
                or self._rendered_length(current + [unit], section) <= self.chunk_size
            ):
                current.append(unit)
                continue
            emit(current, current_section)
            current = self._overlap_units(current, current_section)
            if (
                self._rendered_length(current + [unit], current_section)
                > self.chunk_size
            ):
                current = []
            current.append(unit)
        emit(current, current_section)
        return self._merge_small_units(chunks)

    def _rendered_length(self, units: list[Any], section: str | None) -> int:
        body = "\n\n".join(unit.page_content.strip() for unit in units)
        prefix = len(section) + 2 if section and not body.startswith(section) else 0
        return prefix + len(body)

    def _overlap_units(self, units: list[Any], section: str | None) -> list[Any]:
        if self.chunk_overlap == 0:
            return []
        overlap: list[Any] = []
        for unit in reversed(units):
            candidate = [unit] + overlap
            if (
                overlap
                and len("\n\n".join(item.page_content for item in candidate))
                > self.chunk_overlap
            ):
                break
            overlap = candidate
        return (
            overlap if self._rendered_length(overlap, section) < self.chunk_size else []
        )

    def _split_at_meaningful_boundaries(
        self, paragraph: str, section: str | None
    ) -> list[str]:
        """Split at sentences, then clauses; fall back to words only if unavoidable."""
        paragraph = re.sub(r"[ \t]*\n[ \t]*", " ", paragraph).strip()
        heading_length = (
            len(section) + 2 if section and not paragraph.startswith(section) else 0
        )
        budget = max(1, self.chunk_size - heading_length)
        if len(paragraph) <= budget:
            return [paragraph]
        sentences = re.split(r"(?<=[.!?…])(?=\s+(?:[A-ZÀ-ỴĐ0-9\"“'‘(\[]))", paragraph)
        parts: list[str] = []
        for sentence in (item.strip() for item in sentences if item.strip()):
            if len(sentence) <= budget:
                parts.append(sentence)
                continue
            clauses = re.split(r"(?<=[,;:])(?=\s+)", sentence)
            current = ""
            for clause in (item.strip() for item in clauses if item.strip()):
                candidate = f"{current} {clause}".strip()
                if current and len(candidate) > budget:
                    parts.append(current)
                    current = clause
                else:
                    current = candidate
            if current:
                parts.append(current)
        result: list[str] = []
        for part in parts:
            if len(part) <= budget:
                result.append(part)
                continue
            line = ""
            for word in part.split():
                candidate = f"{line} {word}".strip()
                if line and len(candidate) > budget:
                    result.append(line)
                    line = word
                else:
                    line = candidate
            if line:
                result.append(line)
        return result

    def _merge_small_units(self, units: list[Any]) -> list[Any]:
        merged: list[Any] = []
        for unit in units:
            text = unit.page_content.strip()
            if not text:
                continue
            if not merged or len(text) >= self.minimum_chunk_size:
                merged.append(unit)
                continue
            previous = merged[-1]
            if previous.metadata.get("section") != unit.metadata.get("section"):
                merged.append(unit)
                continue
            section = unit.metadata.get("section")
            prefix = (
                f"{section}\n\n"
                if section and text.startswith(f"{section}\n\n")
                else ""
            )
            combined = f"{previous.page_content}\n\n{text[len(prefix) :]}".strip()
            if len(combined) <= self.chunk_size:
                merged[-1] = self._document(
                    combined,
                    {
                        **previous.metadata,
                        "end_page": unit.metadata.get("end_page")
                        or previous.metadata.get("end_page"),
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
            "chunk_id": str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL, f"{document_id}:{chunk_index}:{content_hash}"
                )
            ),
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

    @staticmethod
    def _is_document_title(line: str) -> bool:
        return bool(re.match(r"^(Nội quy|Quy định)\b", line, re.IGNORECASE))

    @staticmethod
    def _is_heading_only(paragraph: str) -> bool:
        line = " ".join(paragraph.split())
        if "\n" in paragraph or len(line) > 180:
            return False
        letters = re.sub(r"[^A-Za-zÀ-ỹĐđ]", "", line)
        if (
            len(letters) >= 5
            and letters == letters.upper()
            and letters != letters.lower()
        ):
            return True
        return bool(
            re.match(
                r"^(?:Chương|Mục|Điều|Khoản)\s+\d+|^\d{1,3}[.)]\s+",
                line,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _is_clause_item(line: str, active_section: str | None) -> bool:
        """Treat numbered lines as clauses only inside an article or clause."""
        return bool(
            active_section
            and re.match(r"^(Điều|Khoản)\b", active_section, re.IGNORECASE)
            and re.match(r"^\(?\d{1,3}[.)]\s+", line)
        )

    def _is_heading(self, line: str) -> bool:
        # Tiêu đề là thứ ngắn. Cắt theo độ dài trước khi khớp mẫu, vì mẫu chỉ
        # nhìn phần đầu dòng nên nó không tự phân biệt được "Nội quy sử dụng
        # phòng" với cả một mục dài mở đầu bằng đúng mấy chữ đó (MAX_HEADING_CHARS).
        if len(line) > self.MAX_HEADING_CHARS:
            return False
        return bool(
            self.HEADING_RE.match(line)
            or self.HEADING_RE.match(self._strip_accents(line))
        )

    @staticmethod
    def _strip_accents(value: str) -> str:
        normalized = unicodedata.normalize("NFD", value)
        return "".join(
            char for char in normalized if unicodedata.category(char) != "Mn"
        )

    def _infer_category_subject(
        self, metadata: dict[str, Any]
    ) -> tuple[str | None, str | None]:
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
