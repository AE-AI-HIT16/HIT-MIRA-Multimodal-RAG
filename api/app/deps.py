"""Dependencies dùng chung cho các router (CHO SẴN).

Ngoài DI cho FastAPI, đây là nơi WIRING provider (LLM/embedder/vectorstore) —
để `shared/` không phụ thuộc ngược vào `app.config`. Provider tạo 1 lần (lru_cache),
lazy import bên trong để `import app.deps` không kéo theo torch/genai.
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.config import settings
from app.domains.consent import service as consent_service
from shared.db.session import get_session


# ===== Provider factories (T-20/T-23) =====
@lru_cache
def get_llm():
    """Gemini free-tier sau lớp LLMClient. [T-23]"""
    from shared.providers.llm import GeminiClient

    return GeminiClient(
        settings.llm_api_key,
        model=settings.llm_model or "gemini-2.5-flash",
    )


@lru_cache
def get_text_embedder():
    """Vietnamese_Embedding cho transcript & nội quy. [T-20]"""
    from shared.providers.embeddings import VietnameseTextEmbedder

    return VietnameseTextEmbedder()


@lru_cache
def get_image_embedder():
    """Jina-CLIP v2 cho ảnh + query text chung không gian. [T-20]"""
    from shared.providers.embeddings import JinaClipEmbedder

    return JinaClipEmbedder()


@lru_cache
def get_asr():
    """PhoWhisper-large qua backend faster-whisper. [T-21]"""
    from shared.providers.asr import FasterWhisperASR

    return FasterWhisperASR()


@lru_cache
def get_captioner():
    """Gemini 2.5 Flash Vision → caption tiếng Việt. [T-22]"""
    from shared.providers.captioner import GeminiVisionCaptioner

    return GeminiVisionCaptioner(
        settings.llm_api_key,
        model=settings.llm_model or "gemini-2.5-flash",
    )


@lru_cache
def get_media_store():
    """QdrantStore collection media, chiều = dim của image embedder. [T-30]"""
    from shared.vectorstore.qdrant import QdrantStore

    return QdrantStore(
        settings.qdrant_url,
        settings.qdrant_collection_media,
        get_image_embedder().dim,
    )


@lru_cache
def get_transcript_store():
    """QdrantStore collection transcript, chiều = dim của text embedder. [T-32]"""
    from shared.vectorstore.qdrant import QdrantStore

    return QdrantStore(
        settings.qdrant_url,
        settings.qdrant_collection_transcript,
        get_text_embedder().dim,
    )


@lru_cache
def get_regulation_store():
    """QdrantStore collection nội quy, chiều = dim của text embedder. [T-31]"""
    from shared.vectorstore.qdrant import QdrantStore

    return QdrantStore(
        settings.qdrant_url,
        settings.qdrant_collection_regulation,
        get_text_embedder().dim,
    )


def require_consent(session: Session = Depends(get_session)) -> None:
    """Cổng chặn ingestion: phải có consent hợp lệ mới cho nạp dữ liệu.

    [BR-101, BR-701 · US-101.1]  Không có consent còn hiệu lực → 403.
    """
    if consent_service.get_active_consent(session) is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Chưa có quyền sử dụng dữ liệu",
        )


def require_admin() -> None:
    """Chỉ cho phép role admin gọi endpoint quản trị.

    TODO(US-505 · NFR bảo mật): giải mã JWT, kiểm tra role == admin.
    Tạm no-op cho POC — sinh viên implement khi làm auth.
    """
    return None
