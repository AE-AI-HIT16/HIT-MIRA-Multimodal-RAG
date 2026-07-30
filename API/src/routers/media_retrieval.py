from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from src.rag_video_anh.embedding.embedding_service import ImageEmbeddingConfigurationError
from src.rag_video_anh.retrieval.retrieval_service import (
    ALLOWED_SOURCES,
    SOURCE_BOTH,
    VideoRetrievalService,
    build_video_retrieval_service,
)
from src.rag_video_anh.vector_store.vector_store import VideoVectorStoreConfigurationError

router = APIRouter(prefix="/media", tags=["media-retrieval"])


class MediaRetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, gt=0, le=50)
    video_ids: list[str] | None = None
    source: str = Field(default=SOURCE_BOTH, description=f"Một trong {ALLOWED_SOURCES}")


@lru_cache(maxsize=1)
def get_media_retrieval_service() -> VideoRetrievalService:
    return build_video_retrieval_service()


async def _search(
    service: VideoRetrievalService,
    query: str,
    top_k: int | None,
    video_ids: list[str] | None,
    source: str,
) -> dict[str, Any]:
    try:
        return await run_in_threadpool(service.retrieve, query, top_k, video_ids, source)
    except (ImageEmbeddingConfigurationError, VideoVectorStoreConfigurationError) as exc:
        # Hai lỗi này kế thừa ValueError nên PHẢI bắt trước, nếu không thiếu
        # JINA_API_KEY/QDRANT_URL phía server sẽ bị báo thành lỗi của client.
        raise HTTPException(
            status_code=503,
            detail=f"Dịch vụ truy hồi media chưa được cấu hình đầy đủ: {exc}",
        ) from exc
    except ValueError as exc:
        # query rỗng / source không hợp lệ là lỗi phía client.
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/search")
async def search_media(
    request: MediaRetrievalRequest,
    service: VideoRetrievalService = Depends(get_media_retrieval_service),
) -> dict[str, Any]:
    return await _search(
        service,
        request.query,
        request.top_k,
        request.video_ids,
        request.source,
    )


@router.get("/search")
async def search_media_get(
    query: str = Query(..., min_length=1),
    top_k: int | None = Query(default=None, gt=0, le=50),
    video_ids: list[str] | None = Query(default=None),
    source: str = Query(default=SOURCE_BOTH),
    service: VideoRetrievalService = Depends(get_media_retrieval_service),
) -> dict[str, Any]:
    return await _search(service, query, top_k, video_ids, source)
