from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from src.log.logger import logger


class RAGPipelineService:
    """Orchestrates parse, clean, chunk, embed, and vector upsert."""

    def __init__(self, parser, cleaner, chunker, ingest_service) -> None:
        self.parser = parser
        self.cleaner = cleaner
        self.chunker = chunker
        self.ingest_service = ingest_service

    def ingest_document(
        self,
        file_path: str,
        filename: str | None = None,
        document_id: str | None = None,
    ) -> dict[str, Any]:
        errors: list[str] = []
        resolved_filename = filename or Path(file_path).name
        resolved_document_id = document_id or str(uuid.uuid5(uuid.NAMESPACE_URL, str(Path(file_path).resolve())))

        try:
            parsed_documents = self.parser.parse(file_path, filename=resolved_filename)
            cleaned_documents = self.cleaner.clean_documents(parsed_documents)
            chunks = self.chunker.chunk_documents(
                cleaned_documents,
                document_id=resolved_document_id,
                filename=resolved_filename,
            )
            ingest_result = self.ingest_service.ingest_chunks(chunks)
            total_pages = self._total_pages(cleaned_documents)
            logger.info(
                f"Completed ingest for document_id '{resolved_document_id}' with {len(chunks)} chunk(s)"
            )
            return {
                "document_id": resolved_document_id,
                "filename": resolved_filename,
                "total_pages": total_pages,
                "total_documents": len(cleaned_documents),
                "total_chunks": len(chunks),
                "status": "completed",
                "errors": errors,
                "ingest": ingest_result,
            }
        except (FileNotFoundError, ValueError, TypeError, RuntimeError) as exc:
            step_error = f"ingest_document failed for '{resolved_filename}': {exc}"
            logger.error(step_error)
            errors.append(step_error)
            return {
                "document_id": resolved_document_id,
                "filename": resolved_filename,
                "total_pages": 0,
                "total_documents": 0,
                "total_chunks": 0,
                "status": "failed",
                "errors": errors,
            }

    @staticmethod
    def _total_pages(documents: list[dict[str, Any]]) -> int:
        pages = {document.get("page") for document in documents if document.get("page") is not None}
        return len(pages)
