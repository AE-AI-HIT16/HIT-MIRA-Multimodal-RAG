"""Nạp jina-clip-v2 và sinh vector đúng như API của Jina vẫn trả về.

Đây là phần lõi, dùng chung cho cả hai chỗ chạy: GPU trên RunPod (nhúng ảnh
hàng loạt, ngoại tuyến) và CPU ngay trên máy chủ API (nhúng câu hỏi, trực
tuyến). Cùng một lớp để không có nguy cơ hai nơi lệch nhau về tiền xử lý —
lệch một chút thôi là toàn bộ index cũ trong Qdrant thành vô giá trị.

Ba tham số dưới đây phải khớp tuyệt đối với những gì `ImageEmbeddingService`
từng gửi lên api.jina.ai, nếu không vector mới sẽ không nằm cùng không gian
với 2.409 điểm đã có:

* `task="retrieval.query"` — chọn LoRA adapter. jina-clip-v2 CHỈ nhận giá trị
  này; nhúng bất đối xứng query/passage là của jina-embeddings-v3.
* `truncate_dim=1024` — model này là Matryoshka, cắt được xuống tận 64 chiều.
  Cắt nhầm là sai số chiều, Qdrant từ chối ngay (còn may hơn là im lặng sai).
* chuẩn hoá L2 — client cũ luôn gửi `"normalized": true`.

Ảnh đã được thu nhỏ về 512px phía client trước khi base64, nên ở đây không
đụng vào kích thước nữa.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jinaai/jina-clip-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
# jina-clip-v2 chỉ chấp nhận đúng task này (xem docstring trên).
TEXT_TASK = os.getenv("EMBED_TEXT_TASK", "retrieval.query")


class EncoderError(RuntimeError):
    """Đầu vào hỏng hoặc model từ chối — trả 4xx/5xx cho client, không nuốt."""


def _resolve_device() -> str:
    override = os.getenv("EMBED_DEVICE", "").strip()
    if override:
        return override
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(device: str) -> str:
    """Trả về TÊN kiểu dữ liệu, không phải đối tượng `torch.dtype`.

    Remote code của Jina làm `hasattr(torch, torch_dtype)` nên bắt buộc phải là
    chuỗi. transformers 5.x tự đổi chuỗi trong config.json thành `torch.dtype`
    trước khi dựng config, và đó chính là chỗ vỡ:
    `TypeError: attribute name must be string, not 'torch.dtype'`. Truyền
    tường minh một chuỗi vào `from_pretrained` thì ghi đè được giá trị đã bị
    đổi kiểu đó.
    """
    override = os.getenv("EMBED_DTYPE", "").strip()
    if override:
        return override
    # fp16 trên CPU chậm hơn fp32 vì phần lớn kernel phải ép kiểu qua lại;
    # trên GPU thì ngược lại, fp16 nhanh gấp đôi mà cosine lệch < 1e-3.
    return "float16" if device.startswith("cuda") else "float32"


class JinaClipEncoder:
    """Bọc model để phần server không phải biết gì về torch.

    Nạp trọng số lười (lazy): tiến trình lên được ngay và trả `/health` trong
    lúc còn đang nạp, thay vì im lặng vài chục giây rồi mới nghe cổng.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._device: str | None = None
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        return self._device or "chưa nạp"

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        # Hai request đầu tiên tới cùng lúc thì chỉ một cái được nạp; cái kia
        # chờ ở đây thay vì nạp bản thứ hai và nhân đôi bộ nhớ.
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModel

            device = _resolve_device()
            dtype_name = _resolve_dtype(device)
            logger.info(f"Đang nạp {MODEL_NAME} lên {device} ({dtype_name})")
            model = AutoModel.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                torch_dtype=dtype_name,
                # Nạp thẳng trọng số vào chỗ của model thay vì dựng model rỗng
                # rồi copy đè: cách mặc định ngốn gấp đôi bộ nhớ trong lúc nạp
                # (đo được đỉnh 4,95GB, đủ để bị OOM giết trên máy 7,6GB đang
                # chạy Docker). Cần gói `accelerate`.
                low_cpu_mem_usage=True,
            )
            model = model.to(device)
            model.eval()
            torch.set_grad_enabled(False)
            self._model = model
            self._device = device
            logger.info(f"Đã nạp {MODEL_NAME} lên {device}")

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        self.load()
        try:
            vectors = self._model.encode_text(
                texts,
                task=TEXT_TASK,
                truncate_dim=EMBED_DIM,
                normalize_embeddings=True,
            )
        except TypeError:
            # Bản remote code cũ hơn không có `normalize_embeddings`; tự chuẩn
            # hoá ở dưới nên bỏ tham số đi vẫn ra đúng vector.
            vectors = self._model.encode_text(texts, task=TEXT_TASK, truncate_dim=EMBED_DIM)
        return _as_unit_vectors(vectors)

    def encode_images(self, images_base64: list[str]) -> list[list[float]]:
        self.load()
        images = [_decode_image(item) for item in images_base64]
        try:
            vectors = self._model.encode_image(
                images,
                truncate_dim=EMBED_DIM,
                normalize_embeddings=True,
            )
        except TypeError:
            vectors = self._model.encode_image(images, truncate_dim=EMBED_DIM)
        return _as_unit_vectors(vectors)


def _decode_image(item: str):
    """Giải base64 thành ảnh PIL RGB.

    Client gửi base64 trần, nhưng chấp nhận luôn dạng data URL để ai gõ tay
    bằng curl không phải bực mình.
    """
    from PIL import Image

    if not isinstance(item, str) or not item.strip():
        raise EncoderError("input.image rỗng")
    payload = item.strip()
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise EncoderError("input.image không phải base64 hợp lệ") from exc
    try:
        with Image.open(io.BytesIO(raw)) as image:
            return image.convert("RGB")
    except Exception as exc:
        raise EncoderError(f"không mở được ảnh: {exc.__class__.__name__}") from exc


def _as_unit_vectors(vectors: Any) -> list[list[float]]:
    """Đưa về list[list[float]] đã chuẩn hoá L2.

    Chuẩn hoá lại lần nữa dù model có thể đã làm: rẻ, và nếu một bản remote
    code nào đó bỏ qua bước này thì mọi điểm số cosine sẽ lệch mà không có
    dấu hiệu nào lộ ra.
    """
    import numpy as np

    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    # Vector không thì để nguyên, chia cho 0 ra nan còn tệ hơn.
    norms[norms == 0] = 1.0
    return (array / norms).tolist()
