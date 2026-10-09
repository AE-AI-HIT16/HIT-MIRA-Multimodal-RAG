from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from src.configuration import AppConfig
from src.rag_noiquy.embedding.embedding_service import EmbeddingService
from src.rag_noiquy.pipeline import DocumentCleaner, DocumentParser, RAGPipelineService, StructureAwareChunker
from src.rag_noiquy.pipeline.ingest import IngestService
from src.rag_noiquy.vector_store.vector_store import QdrantVectorStore
from src.routers.auth import yeu_cau_admin

router = APIRouter(prefix="/documents", tags=["documents"])


class IngestDocumentRequest(BaseModel):
    file_path: str = Field(..., description="Path to a PDF or DOCX file on the API server.")
    filename: str | None = None
    document_id: str | None = None


class DeleteDocumentResponse(BaseModel):
    document_id: str
    status: str


def build_pipeline_service(config: AppConfig | None = None) -> RAGPipelineService:
    app_config = config or AppConfig()
    embedding_service = EmbeddingService(config=app_config)
    vector_store = QdrantVectorStore(config=app_config)
    return RAGPipelineService(
        parser=DocumentParser(),
        cleaner=DocumentCleaner(),
        chunker=StructureAwareChunker(config=app_config),
        ingest_service=IngestService(
            embedding_service=embedding_service,
            vector_store=vector_store,
        ),
    )


@lru_cache(maxsize=1)
def get_pipeline_service() -> RAGPipelineService:
    return build_pipeline_service()


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore(config=AppConfig())


def ensure_completed(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") == "failed":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result)
    return result


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    _admin: Any = Depends(yeu_cau_admin),
    file: UploadFile = File(...),
    document_id: str | None = Form(default=None),
    pipeline_service: RAGPipelineService = Depends(get_pipeline_service),
) -> dict[str, Any]:
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="filename is required")
    if Path(filename).suffix.lower() not in DocumentParser.SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF and DOCX files are supported")

    with tempfile.TemporaryDirectory(prefix="hit-mira-upload-") as temp_dir:
        temp_path = Path(temp_dir) / filename
        temp_path.write_bytes(await file.read())
        result = await run_in_threadpool(
            pipeline_service.ingest_document,
            str(temp_path),
            filename,
            document_id,
        )
        return ensure_completed(result)


@router.post("/ingest", status_code=status.HTTP_201_CREATED)
async def ingest_document(
    request: IngestDocumentRequest,
    _admin: Any = Depends(yeu_cau_admin),
    pipeline_service: RAGPipelineService = Depends(get_pipeline_service),
) -> dict[str, Any]:
    result = await run_in_threadpool(
        pipeline_service.ingest_document,
        request.file_path,
        request.filename,
        request.document_id,
    )
    return ensure_completed(result)


@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
async def delete_document(
    document_id: str,
    _admin: Any = Depends(yeu_cau_admin),
    vector_store: QdrantVectorStore = Depends(get_vector_store),
) -> DeleteDocumentResponse:
    if not document_id.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="document_id is required")
    await run_in_threadpool(vector_store.delete_document, document_id.strip())
    return DeleteDocumentResponse(document_id=document_id.strip(), status="deleted")
