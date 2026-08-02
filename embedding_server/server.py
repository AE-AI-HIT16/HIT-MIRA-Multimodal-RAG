"""Endpoint nhúng tự host, nói đúng giao thức của api.jina.ai.

Giữ nguyên giao thức là chủ ý: `ImageEmbeddingService` bên `API/` không cần
sửa một dòng nào, chỉ đổi `MEDIA_MODELS.CLIP_API_BASE_URL` trong
`API/Resources/model.yaml` (hoặc biến môi trường `MEDIA_IMAGE_EMBEDDING_BASE_URL`)
là xong. Sửa client để hợp với server mới sẽ kéo theo sửa cả nhánh `rag_noiquy`
lẫn đống test đang xanh — không đáng, khi mà giao thức chỉ có ba trường.

Hợp đồng phải khớp:

* Nhận  `{"model", "input": [{"image": b64} | {"text": s}], "task", "dimensions", "normalized"}`
* Trả   `{"data": [{"index": i, "embedding": [...]}], "usage": {...}}`
* Client sắp lại theo `index` rồi ĐÒI ĐÚNG số vector bằng số phần tử `input` —
  trả thiếu một cái là nó ném `ImageEmbeddingServiceError`, không nuốt lặng.

Chạy:
    uvicorn server:app --host 0.0.0.0 --port 8100
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from encoder import EMBED_DIM, MODEL_NAME, EncoderError, JinaClipEncoder

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Bỏ trống thì chấp nhận mọi khoá. Endpoint này nằm trong mạng nội bộ hoặc sau
# proxy của RunPod; đặt khoá vào đây là lớp phòng thủ thêm, không phải lớp duy nhất.
API_KEY = os.getenv("EMBED_SERVER_API_KEY", "").strip()

encoder = JinaClipEncoder()


class InputItem(BaseModel):
    text: str | None = None
    image: str | None = None


class EmbeddingRequest(BaseModel):
    model: str | None = None
    # Hai dạng `input` vì hai client khác nhau cùng gọi vào đây:
    #   - nhánh media gửi kiểu Jina:   [{"image": b64}] hoặc [{"text": s}]
    #   - nhánh nội quy đi qua LangChain OpenAIEmbeddings, gửi kiểu OpenAI:
    #     ["chuỗi", "chuỗi"] — chuỗi trần, không bọc object.
    # Chỉ nhận một dạng thì một trong hai nhánh ăn 422 mà rất khó đoán ra vì sao.
    input: list[InputItem | str] = Field(min_length=1)
    task: str | None = None
    embedding_type: Literal["float"] | None = "float"
    # LangChain gửi thêm trường này; nó tự xử lý được cả float lẫn base64 nên
    # cứ trả float. Khai ra để pydantic không từ chối request.
    encoding_format: str | None = None
    dimensions: int | None = None
    normalized: bool | None = True

    def tach_text_va_anh(self) -> tuple[list[str], list[str]]:
        texts: list[str] = []
        images: list[str] = []
        for item in self.input:
            if isinstance(item, str):
                texts.append(item)
            elif item.text is not None:
                texts.append(item.text)
            elif item.image is not None:
                images.append(item.image)
        return texts, images


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Nạp sẵn nếu được yêu cầu, để request đầu tiên không phải chờ.

    Mặc định TẮT: pod trên RunPod nên trả lời `/health` sớm cho health check,
    còn máy CPU thì nạp mất hơn nửa phút.
    """
    if os.getenv("EMBED_PRELOAD", "").lower() in {"1", "true", "yes"}:
        encoder.load()
    yield


app = FastAPI(title="HIT-MIRA embedding server", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    """Trả lời được ngay cả khi model chưa nạp xong — `ready` nói rõ tình trạng."""
    return {"status": "ok", "model": MODEL_NAME, "device": encoder.device, "ready": encoder.ready}


@app.post("/v1/embeddings")
def embeddings(request: EmbeddingRequest, authorization: str = Header(default="")) -> dict[str, Any]:
    if API_KEY and authorization.removeprefix("Bearer ").strip() != API_KEY:
        raise HTTPException(status_code=401, detail="Sai hoặc thiếu API key")

    # Số chiều sai thì Qdrant sẽ từ chối tận lúc upsert, sau khi đã tốn cả mẻ
    # nhúng. Chặn ngay ở đây và nói rõ lý do.
    if request.dimensions is not None and request.dimensions != EMBED_DIM:
        raise HTTPException(
            status_code=422,
            detail=f"Server chỉ phục vụ {EMBED_DIM} chiều, client xin {request.dimensions}",
        )

    texts, images = request.tach_text_va_anh()
    if texts and images:
        # Trộn hai loại trong một request thì thứ tự trả về không còn khớp
        # `index` của client. Client thật không bao giờ trộn, nên từ chối thẳng
        # còn hơn trả về vector xếp sai chỗ mà không ai biết.
        raise HTTPException(status_code=422, detail="Một request chỉ được chứa toàn text hoặc toàn image")
    if not texts and not images:
        raise HTTPException(status_code=422, detail="input phải có ít nhất một 'text' hoặc 'image'")

    try:
        vectors = encoder.encode_texts(texts) if texts else encoder.encode_images(images)
    except EncoderError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Nhúng thất bại")
        raise HTTPException(status_code=500, detail=f"{exc.__class__.__name__}: {exc}") from exc

    so_luong = len(texts) if texts else len(images)
    return {
        "model": request.model or MODEL_NAME,
        "object": "list",
        "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)],
        # Client không đọc `usage`, nhưng giữ cho giống hợp đồng gốc để ai đó
        # cắm curl vào so sánh không thấy thiếu trường.
        "usage": {"total_tokens": 0, "prompt_tokens": 0, "items": so_luong},
    }
