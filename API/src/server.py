from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute, APIRouter

from src.routers.documents import router as documents_router
from src.routers.retrieval import router as retrieval_router


def include_api_router(app: FastAPI, router: APIRouter, prefix: str = "") -> None:
    """Include router routes eagerly for FastAPI versions with deferred routers."""

    for route in router.routes:
        if isinstance(route, APIRoute):
            app.add_api_route(
                f"{prefix}{route.path}",
                route.endpoint,
                methods=route.methods,
                response_model=route.response_model,
                status_code=route.status_code,
                tags=route.tags,
            )


def create_app() -> FastAPI:
    app = FastAPI(
        title="HIT-MIRA Multimodal RAG API",
        version="0.1.0",
        description="Document ingest and retrieval API for the HIT-MIRA RAG system.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    include_api_router(app, documents_router, prefix="/api")
    include_api_router(app, retrieval_router, prefix="/api")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True)
