from __future__ import annotations

from pathlib import Path
from typing import Any

from src.log.logger import logger


class DocumentParser:
    """Parse PDF and DOCX files with LangChain loaders."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

    def parse(self, file_path: str | Path, filename: str | None = None) -> list[dict[str, Any]]:
        path = self._validate_file(file_path)
        source_name = filename or path.name
        documents = [
            self._to_pipeline_document(doc, source_name, path.suffix.lower())
            for doc in self._load_documents(path)
            if doc.page_content.strip()
        ]
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
    def _load_documents(path: Path):
        try:
            from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
        except ImportError as exc:
            raise RuntimeError(
                "Missing LangChain parser dependencies. Install langchain-community, pypdf, and docx2txt."
            ) from exc

        loader = PyPDFLoader(str(path), mode="page") if path.suffix.lower() == ".pdf" else Docx2txtLoader(str(path))
        return loader.load()

    @staticmethod
    def _to_pipeline_document(document, source_name: str, extension: str) -> dict[str, Any]:
        metadata = dict(document.metadata or {})
        page = metadata.get("page", 0)
        page_number = int(page) + 1 if extension == ".pdf" else 1
        return {
            "text": document.page_content.strip(),
            "page": page_number,
            "source": source_name,
            "metadata": {**metadata, "file_type": extension.lstrip("."), "source": source_name},
        }
