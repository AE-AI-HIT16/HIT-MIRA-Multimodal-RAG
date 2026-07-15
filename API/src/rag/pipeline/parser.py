from __future__ import annotations

from pathlib import Path
from typing import Any

from src.log.logger import logger


class DocumentParser:
    """Parse PDF and DOCX files into ordered text units."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

    def parse(self, file_path: str | Path, filename: str | None = None) -> list[dict[str, Any]]:
        path = self._validate_file(file_path)
        source_name = filename or path.name
        extension = path.suffix.lower()

        if extension == ".pdf":
            documents = self._parse_pdf(path, source_name)
        elif extension == ".docx":
            documents = self._parse_docx(path, source_name)
        else:
            raise ValueError(f"Unsupported file type '{extension}'. Only PDF and DOCX are supported.")

        logger.info(f"Parsed {len(documents)} text unit(s) from '{source_name}'")
        return documents

    @classmethod
    def _validate_file(cls, file_path: str | Path) -> Path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a regular file: {path}")
        if path.suffix.lower() not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{path.suffix}'. Only PDF and DOCX are supported.")
        return path

    @staticmethod
    def _parse_pdf(path: Path, source_name: str) -> list[dict[str, Any]]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Missing dependency 'pypdf' required to parse PDF files.") from exc

        reader = PdfReader(str(path))
        documents: list[dict[str, Any]] = []
        for page_index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            documents.append(
                {
                    "text": text,
                    "page": page_index,
                    "source": source_name,
                    "metadata": {"file_type": "pdf"},
                }
            )
        return documents

    @staticmethod
    def _parse_docx(path: Path, source_name: str) -> list[dict[str, Any]]:
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("Missing dependency 'python-docx' required to parse DOCX files.") from exc

        document = Document(str(path))
        lines: list[str] = []
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            style_name = getattr(getattr(paragraph, "style", None), "name", "") or ""
            if ("List" in style_name or "Bullet" in style_name) and not text.startswith(("-", "*", "+")):
                text = f"- {text}"
            lines.append(text)

        if not lines:
            return []
        return [
            {
                "text": "\n".join(lines),
                "page": 1,
                "source": source_name,
                "metadata": {"file_type": "docx"},
            }
        ]
