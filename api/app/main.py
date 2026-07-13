"""FastAPI app factory — đăng ký router của từng domain.

Chạy dev:  cd api && uvicorn app.main:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI

from app.config import settings


def create_app() -> FastAPI:
    app = FastAPI(title="HIT-MIRA · Multimodal RAG", version="0.1.0")

    # --- Domain đã implement ---
    from app.domains.consent.router import router as consent_router

    app.include_router(consent_router, prefix="/consents", tags=["consent"])

    # --- TODO: đăng ký khi sinh viên implement xong ---
    #   from app.domains.auth.router import router as auth_router
    #   app.include_router(auth_router, prefix="/auth", tags=["auth"])
    # Thứ tự dự kiến: auth · ingest · media · retrieval · chat · eval

    @app.get("/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok", "env": settings.app_env}

    return app


app = create_app()
