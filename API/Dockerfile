FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/API \
    CONFIG_PATH=API/Resources/dev.yaml \
    MODELS_PATH=API/Resources/model.yaml \
    PROMPTS_PATH=API/Resources/prompt.yaml \
    LOG_DIR=/tmp/hit-mira-api-logs \
    QDRANT_URL=http://qdrant:6333 \
    QDRANT_COLLECTION_NAME=rag_documents \
    QDRANT_DENSE_VECTOR_SIZE=1536 \
    EMBEDDING_BASE_URL=https://openrouter.ai/api/v1 \
    EMBEDDING_DIMENSIONS=1024 \
    EMBEDDING_MODEL=baai/bge-m3 \
    RETRIEVAL_TOP_K=5 \
    RETRIEVAL_KEYWORD_THRESHOLD=0.5 \
    MINIO_ENDPOINT=localhost:9000 \
    MINIO_BUCKET=hit-mira-media \
    MINIO_SECURE=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app/API


FROM base AS runtime
EXPOSE 8000
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
