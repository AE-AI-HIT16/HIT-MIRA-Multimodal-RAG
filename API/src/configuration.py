from typing import Optional
from pydantic import BaseModel

from src.config.config import config_object, config_models

class QdrantConfig(BaseModel):
    """Qdrant connection settings."""
    url: Optional[str] = config_object.QDRANT.URL 
    collection_name: Optional[str] = config_object.QDRANT.COLLECTION_NAME 
    dense_vector_size: Optional[int] = config_object.QDRANT.DENSE_VECTOR_SIZE 

class EmbeddingConfig(BaseModel):
    """Embedding model settings."""
    dense_model: Optional[str] = config_object.EMBEDDING.DENSE_MODEL 
    sparse_model: Optional[str] = config_object.EMBEDDING.SPARSE_MODEL 

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
    llm : Optional[LLMConfig] = LLMConfig()
