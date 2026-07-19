from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from src.rag_noiquy.retrieval.retrieval_service import RetrievalService, build_retrieval_service


router = APIRouter(prefix="/retrieval", tags=["retrieval"])


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, gt=0, le=50)
    document_ids: list[str] | None = None
    rewrite: bool = True


@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:
    return build_retrieval_service()


@router.post("/search")
async def search(
    request: RetrievalRequest,
    service: RetrievalService = Depends(get_retrieval_service),
) -> dict[str, Any]:
    return await run_in_threadpool(
        service.retrieve,
        request.query,
        request.top_k,
        request.document_ids,
        request.rewrite,
    )


@router.get("/search")
async def search_get(
    query: str = Query(..., min_length=1),
    top_k: int | None = Query(default=None, gt=0, le=50),
    rewrite: bool = Query(default=True),
    service: RetrievalService = Depends(get_retrieval_service),
) -> dict[str, Any]:
    return await run_in_threadpool(service.retrieve, query, top_k, None, rewrite)
