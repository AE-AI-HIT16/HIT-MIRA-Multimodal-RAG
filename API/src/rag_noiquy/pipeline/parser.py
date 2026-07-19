from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from src.log.logger import logger


class DocumentParser:
    """Parse PDF and DOCX files into LangChain Document objects."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

    def parse(self, file_path: str | Path, filename: str | None = None) -> list[Any]:
        path = self._validate_file(file_path)
        source_name = filename or path.name
        extension = path.suffix.lower()
        documents = [
            self._normalize_document(document, source_name, extension)
            for document in self._load_documents(path)
            if document.page_content.strip()
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
    def _load_documents(path: Path) -> list[Any]:
        try:
            loaders = import_module("langchain_community.document_loaders")
        except ImportError as exc:
            raise RuntimeError(
                "Missing LangChain parser dependencies. Install langchain-community, pypdf, and docx2txt."
            ) from exc

        loader_cls = getattr(loaders, "PyPDFLoader") if path.suffix.lower() == ".pdf" else getattr(loaders, "Docx2txtLoader")
        loader = loader_cls(str(path), mode="page") if path.suffix.lower() == ".pdf" else loader_cls(str(path))
        return loader.load()

    @staticmethod
    def _normalize_document(document: Any, source_name: str, extension: str) -> Any:
        document_cls = getattr(import_module("langchain_core.documents"), "Document")
        metadata = dict(document.metadata or {})
        page = metadata.get("page", 0)
        page_number = int(page) + 1 if extension == ".pdf" else 1
        return document_cls(
            page_content=document.page_content.strip(),
            metadata={
                **metadata,
                "page": page_number,
                "source": source_name,
                "file_type": extension.lstrip("."),
            },
        )
