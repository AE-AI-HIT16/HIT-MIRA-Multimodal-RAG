"""Cấu hình tập trung (pydantic-settings, đọc .env)."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_env: str = "dev"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    jwt_secret: str = "change-me"

    # DB — mặc định sqlite cho dev (zero-config). Prod/docker override sang
    # postgres qua .env: postgresql+psycopg://hit:...@postgres:5432/hit_mira
    database_url: str = "sqlite:///./data/dev.db"

    # Qdrant — 3 collection riêng (media / transcript / nội quy)
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_media: str = "media_clip"
    qdrant_collection_transcript: str = "video_transcript"
    qdrant_collection_regulation: str = "regulation_text"

    # Storage — media gốc. filesystem (dev/test, zero-config) | minio (docker/prod).
    storage_backend: str = "filesystem"
    data_dir: str = "./data"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "hit-mira-media"
    minio_secure: bool = False

    # LLM provider (free-tier, sau lớp abstraction)
    llm_provider: str = "gemini"
    llm_api_key: str = ""
    llm_model: str = ""


settings = Settings()
