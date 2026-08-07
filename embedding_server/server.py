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

from encoder import EMBED_DIM, MODEL_NAME, TEXT_TASK, EncoderError, JinaClipEncoder
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Bỏ trống thì chấp nhận mọi khoá. Endpoint này nằm trong mạng nội bộ hoặc sau
# proxy của RunPod; đặt khoá vào đây là lớp phòng thủ thêm, không phải lớp duy nhất.
API_KEY = os.getenv("EMBED_SERVER_API_KEY", "").strip()

encoder = JinaClipEncoder()


class EmbeddingRequestError(ValueError):
    """Request sai hợp đồng; HTTP và RunPod cùng ánh xạ lỗi này ở biên."""


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
        for index, item in enumerate(self.input):
            if isinstance(item, str):
                value = item.strip()
                if not value:
                    raise EmbeddingRequestError(f"input[{index}] chứa text rỗng")
                texts.append(value)
                continue
            has_text = item.text is not None
            has_image = item.image is not None
            if has_text == has_image:
                raise EmbeddingRequestError(
                    f"input[{index}] phải có đúng một trường 'text' hoặc 'image'"
                )
            value = (item.text if has_text else item.image) or ""
            value = value.strip()
            if not value:
                field = "text" if has_text else "image"
                raise EmbeddingRequestError(f"input[{index}].{field} rỗng")
            (texts if has_text else images).append(value)
        return texts, images


def encode_request(
    request: EmbeddingRequest,
    encoder_instance: JinaClipEncoder,
    *,
    max_batch_size: int | None = None,
) -> dict[str, Any]:
    """Xử lý một request độc lập với FastAPI hay RunPod.

    Chỉ giữ một bản logic contract để hai cách deploy không thể lệch thứ tự
    vector, số chiều hoặc quy tắc không trộn text/ảnh.
    """
    if request.dimensions is not None and request.dimensions != EMBED_DIM:
        raise EmbeddingRequestError(
            f"Server chỉ phục vụ {EMBED_DIM} chiều, client xin {request.dimensions}"
        )
    if request.task is not None and request.task != TEXT_TASK:
        raise EmbeddingRequestError(
            f"Server chỉ phục vụ task='{TEXT_TASK}', client xin '{request.task}'"
        )
    if request.normalized is False:
        raise EmbeddingRequestError("Server chỉ trả vector đã chuẩn hoá; normalized phải là true")
    if max_batch_size is not None and len(request.input) > max_batch_size:
        raise EmbeddingRequestError(
            f"Một job nhận tối đa {max_batch_size} phần tử, client gửi {len(request.input)}"
        )

    texts, images = request.tach_text_va_anh()
    if texts and images:
        raise EmbeddingRequestError("Một request chỉ được chứa toàn text hoặc toàn image")
    if not texts and not images:
        raise EmbeddingRequestError("input phải có ít nhất một 'text' hoặc 'image'")

    vectors = (
        encoder_instance.encode_texts(texts)
        if texts
        else encoder_instance.encode_images(images)
    )
    expected = len(request.input)
    if len(vectors) != expected:
        raise RuntimeError(f"Model trả {len(vectors)} vector cho {expected} input")
    if any(len(vector) != EMBED_DIM for vector in vectors):
        observed = sorted({len(vector) for vector in vectors})
        raise RuntimeError(f"Model trả sai số chiều: mong đợi {EMBED_DIM}, nhận {observed}")

    return {
        "model": request.model or MODEL_NAME,
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": vector}
            for i, vector in enumerate(vectors)
        ],
        "usage": {"total_tokens": 0, "prompt_tokens": 0, "items": expected},
    }


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

    try:
        return encode_request(request, encoder)
    except (EmbeddingRequestError, EncoderError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Nhúng thất bại")
        raise HTTPException(status_code=500, detail=f"{exc.__class__.__name__}: {exc}") from exc
