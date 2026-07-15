from __future__ import annotations

import hashlib
import re
import uuid
from typing import Any

from src.configuration import AppConfig
from src.log.logger import logger


class StructureAwareChunker:
    """Chunk club regulation documents while preserving legal-style structure."""

    STRUCTURE_RE = re.compile(
        r"^(\s*(Chuong|CHUONG|Muc|MUC|Dieu|DIEU|Khoan|KHOAN)\b.*|"
        r"\s*(\d+(?:\.\d+)*[\).]?\s+.+)|"
        r"\s*(Doi voi nguoi su dung phong|Doi voi nguoi quan ly phong|Hinh thuc xu phat|"
        r"Doi voi ca nhan|Doi voi nhom hoc|Doi voi viec muon va duoc cap khoa)\b.*)",
        re.IGNORECASE,
    )

    VIETNAMESE_STRUCTURE_RE = re.compile(
        r"^(\s*(Chương|CHƯƠNG|Mục|MỤC|Điều|ĐIỀU|Khoản|KHOẢN)\b.*|"
        r"\s*(Đối với người sử dụng phòng|Đối với người quản lý phòng|Hình thức xử phạt|"
        r"Đối với cá nhân|Đối với nhóm học|Đối với việc mượn và được cấp khóa)\b.*)",
        re.IGNORECASE,
    )

    CATEGORY_RE = re.compile(r"^(Chương|CHƯƠNG|Mục|MỤC|Điều|ĐIỀU|Khoản|KHOẢN)\b", re.IGNORECASE)

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        minimum_chunk_size: int | None = None,
        config: AppConfig | None = None,
    ) -> None:
        app_config = config or AppConfig()
        pipeline_config = app_config.pipeline
        self.chunk_size = int(chunk_size or pipeline_config.chunk_size or 1200)
        self.chunk_overlap = int(chunk_overlap if chunk_overlap is not None else pipeline_config.chunk_overlap or 150)
        self.minimum_chunk_size = int(minimum_chunk_size or pipeline_config.minimum_chunk_size or 250)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        if self.minimum_chunk_size < 0:
            raise ValueError("minimum_chunk_size must be >= 0")

    def chunk_documents(
        self,
        documents: list[dict[str, Any]],
        document_id: str,
        filename: str | None = None,
    ) -> list[dict[str, Any]]:
        if not document_id or not isinstance(document_id, str):
            raise ValueError("document_id must be a non-empty string")
        if not documents:
            raise ValueError("documents must not be empty")

        units = self._structure_units(documents)
        merged = self._merge_units(units)
        chunks: list[dict[str, Any]] = []
        for chunk_index, unit in enumerate(merged):
            text = unit["text"].strip()
            chunk_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            source = unit.get("source") or filename
            metadata = dict(unit.get("metadata") or {})
            category, subject = self._infer_category_subject(unit.get("section"), metadata)
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{chunk_index}:{chunk_hash}"))
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "text": text,
                    "page": unit.get("start_page"),
                    "start_page": unit.get("start_page"),
                    "end_page": unit.get("end_page"),
                    "section": unit.get("section"),
                    "category": category,
                    "subject": subject,
                    "chunk_index": chunk_index,
                    "source": source,
                    "filename": filename or source,
                    "metadata": {**metadata, "category": category, "subject": subject},
                }
            )

        logger.info(f"Created {len(chunks)} chunk(s) for document_id '{document_id}'")
        return chunks

    def _structure_units(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        units: list[dict[str, Any]] = []
        current_section: str | None = None

        for document in documents:
            text = str(document.get("text", "")).strip()
            if not text:
                continue
            page = document.get("page")
            source = document.get("source")
            metadata = dict(document.get("metadata") or {})
            paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if paragraph.strip()]

            for paragraph in paragraphs:
                lines = paragraph.split("\n")
                first_line = lines[0].strip()
                if self._is_structure_heading(first_line):
                    current_section = first_line
                units.extend(
                    self._split_large_unit(
                        {
                            "text": paragraph,
                            "start_page": page,
                            "end_page": page,
                            "section": current_section,
                            "source": source,
                            "metadata": metadata,
                        }
                    )
                )
        return units

    def _merge_units(self, units: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None

        for unit in units:
            if current is None:
                current = dict(unit)
                continue

            proposed_text = f"{current['text']}\n\n{unit['text']}"
            starts_new_section = self._is_structure_heading(unit["text"].split("\n", 1)[0])
            should_flush = len(proposed_text) > self.chunk_size and len(current["text"]) >= self.minimum_chunk_size
            if should_flush and starts_new_section:
                chunks.append(current)
                current = dict(unit)
                continue
            if should_flush:
                chunks.append(current)
                current = dict(unit)
                continue

            current["text"] = proposed_text
            current["end_page"] = unit.get("end_page") or current.get("end_page")
            if current.get("section") is None:
                current["section"] = unit.get("section")

        if current is not None:
            if chunks and len(current["text"]) < self.minimum_chunk_size:
                previous = chunks[-1]
                combined_text = f"{previous['text']}\n\n{current['text']}"
                if len(combined_text) <= self.chunk_size:
                    previous["text"] = combined_text
                    previous["end_page"] = current.get("end_page") or previous.get("end_page")
                else:
                    chunks.append(current)
            else:
                chunks.append(current)
        return chunks

    def _split_large_unit(self, unit: dict[str, Any]) -> list[dict[str, Any]]:
        text = unit["text"]
        if len(text) <= self.chunk_size:
            return [unit]

        paragraphs = [paragraph.strip() for paragraph in text.split("\n") if paragraph.strip()]
        pieces: list[dict[str, Any]] = []
        buffer = ""
        for paragraph in paragraphs:
            proposed = paragraph if not buffer else f"{buffer}\n{paragraph}"
            if len(proposed) <= self.chunk_size:
                buffer = proposed
                continue
            if buffer:
                split_unit = dict(unit)
                split_unit["text"] = buffer
                pieces.append(split_unit)
            buffer = paragraph

        if buffer:
            split_unit = dict(unit)
            split_unit["text"] = buffer
            pieces.append(split_unit)
        return pieces

    def _is_structure_heading(self, line: str) -> bool:
        normalized = self._strip_accents(line)
        return bool(self.STRUCTURE_RE.match(normalized) or self.VIETNAMESE_STRUCTURE_RE.match(line))

    @staticmethod
    def _strip_accents(value: str) -> str:
        import unicodedata

        normalized = unicodedata.normalize("NFD", value)
        return "".join(char for char in normalized if unicodedata.category(char) != "Mn")

    def _infer_category_subject(self, section: str | None, metadata: dict[str, Any]) -> tuple[str | None, str | None]:
        if metadata.get("category") or metadata.get("subject"):
            return metadata.get("category"), metadata.get("subject")
        if not section:
            return None, None
        category_match = self.CATEGORY_RE.match(section.strip())
        category = category_match.group(1) if category_match else None
        return category, section
