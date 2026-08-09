from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
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

# Giữ đúng bộ định dạng và hạn mức mà web đã chặn phía client (`web/lib/image.ts`).
# Client chặn để báo lỗi nhanh, server chặn vì client nào cũng có thể bị bỏ qua.
ALLOWED_IMAGE_CONTENT_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
MAX_QUERY_IMAGE_BYTES = 8 * 1024 * 1024
IMAGE_READ_CHUNK_BYTES = 64 * 1024


class MediaRetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, gt=0, le=50)
    video_ids: list[str] | None = None
    source: str = Field(default=SOURCE_BOTH, description=f"Một trong {ALLOWED_SOURCES}")
    years: list[int] | None = Field(
        default=None,
        description=(
            "Lọc theo năm của BÀI ĐĂNG (nhiều năm = HOẶC), tính theo lịch Việt Nam. "
            "Ví dụ [2024, 2025]."
        ),
    )


@lru_cache(maxsize=1)
def get_media_retrieval_service() -> VideoRetrievalService:
    return build_video_retrieval_service()


async def _search(
    service: VideoRetrievalService,
    query: str | None,
    top_k: int | None,
    video_ids: list[str] | None,
    source: str,
    image: bytes | None = None,
    years: list[int] | None = None,
) -> dict[str, Any]:
    try:
        return await run_in_threadpool(
            service.retrieve, query, top_k, video_ids, source, image, years
        )
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
        years=request.years,
    )


async def _read_query_image(upload: UploadFile) -> bytes:
    """Đọc ảnh truy vấn với hạn mức, trả lỗi client rõ ràng (US-502.1 AC-2).

    Đọc theo từng khúc chứ không `await upload.read()` một phát: bản đọc một
    phát nạp trọn file vào RAM trước rồi mới biết nó vượt hạn mức, nên hạn mức
    hoá ra không bảo vệ được đúng thứ nó sinh ra để bảo vệ.
    """
    content_type = (upload.content_type or "").split(";")[0].strip().lower()
    if content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail="Chỉ nhận ảnh JPG, PNG hoặc WEBP.",
        )

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(IMAGE_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_QUERY_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Ảnh quá lớn (tối đa {MAX_QUERY_IMAGE_BYTES // (1024 * 1024)}MB).",
            )
        chunks.append(chunk)

    data = b"".join(chunks)
    if not data:
        raise HTTPException(status_code=422, detail="Ảnh rỗng.")
    return data


@router.post("/search-image")
async def search_media_by_image(
    image: UploadFile = File(..., description="Ảnh truy vấn: JPG, PNG hoặc WEBP"),
    query: str | None = Form(default=None),
    top_k: int | None = Form(default=None, gt=0, le=50),
    video_ids: list[str] | None = Form(default=None),
    source: str = Form(default=SOURCE_BOTH),
    years: list[int] | None = Form(default=None),
    service: VideoRetrievalService = Depends(get_media_retrieval_service),
) -> dict[str, Any]:
    """US-302.1 / US-303.1: tìm ảnh và keyframe tương tự từ MỘT ảnh truy vấn.

    Multipart chứ không phải JSON base64: base64 phình 33% và bắt cả hai đầu
    mã hoá/giải mã một khối vài MB, trong khi trình duyệt gửi `FormData` sẵn.
    """
    data = await _read_query_image(image)
    return await _search(service, query, top_k, video_ids, source, data, years)


@router.get("/search")
async def search_media_get(
    query: str = Query(..., min_length=1),
    top_k: int | None = Query(default=None, gt=0, le=50),
    video_ids: list[str] | None = Query(default=None),
    source: str = Query(default=SOURCE_BOTH),
    years: list[int] | None = Query(default=None, description="Lọc theo năm bài đăng, ví dụ ?years=2024&years=2025"),
    service: VideoRetrievalService = Depends(get_media_retrieval_service),
) -> dict[str, Any]:
    return await _search(service, query, top_k, video_ids, source, None, years)
