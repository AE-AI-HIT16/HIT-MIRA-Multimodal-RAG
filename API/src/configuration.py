from typing import Optional

from pydantic import BaseModel

from src.config.config import config_object, config_models


class QdrantConfig(BaseModel):
    """Qdrant connection settings."""
    url: Optional[str] = config_object.QDRANT.URL
    api_key: Optional[str] = config_object.QDRANT.API_KEY
    collection_name: Optional[str] = config_object.QDRANT.COLLECTION_NAME
    dense_vector_size: Optional[int] = config_object.QDRANT.DENSE_VECTOR_SIZE

class EmbeddingConfig(BaseModel):
    """Embedding model settings."""
    api_key: Optional[str] = config_object.EMBEDDING.API_KEY
    base_url: Optional[str] = config_object.EMBEDDING.EMBEDDING_BASE_URL
    dimensions: Optional[int] = config_object.EMBEDDING.EMBEDDING_DIMENSIONS
    model: Optional[str] = config_object.EMBEDDING.EMBEDDING_MODEL

class PipelineConfig(BaseModel):
    """Document pipeline settings."""
    chunk_size: Optional[int] = config_object.PIPELINE.CHUNK_SIZE
    chunk_overlap: Optional[int] = config_object.PIPELINE.CHUNK_OVERLAP
    minimum_chunk_size: Optional[int] = config_object.PIPELINE.MINIMUM_CHUNK_SIZE

class RetrievalConfig(BaseModel):
    """Retrieval pipeline settings."""
    top_k: Optional[int] = config_object.RETRIEVAL.TOP_K
    keyword_threshold: Optional[float] = config_object.RETRIEVAL.KEYWORD_THRESHOLD

class LLMConfig(BaseModel):
    """The configurable fields for the model llm."""
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = None
    model_name: Optional[str] = config_models.OPENAI_LLM_MODEL.MODEL_PATH
    timeout: Optional[float] = 30
    max_retries: Optional[int] = 3
    base_url: Optional[str] = None


class AppConfig(BaseModel):
    """Top-level application configuration."""
    qdrant: Optional[QdrantConfig] = QdrantConfig()
    embedding: Optional[EmbeddingConfig] = EmbeddingConfig()
    retrieval: Optional[RetrievalConfig] = RetrievalConfig()
    pipeline: Optional[PipelineConfig] = PipelineConfig()
    llm : Optional[LLMConfig] = LLMConfig()
