"""RunPod Serverless adapter cho ``JinaClipEncoder``.

Body gửi vào ``/run`` hoặc ``/runsync`` có dạng::

    {"input": {"model": "jina-clip-v2", "input": [{"text": "..."}], ...}}

RunPod tự bọc giá trị trả về dưới trường ``output``. Giá trị bên trong giữ
nguyên giao thức embeddings hiện có để benchmark và client chỉ phải tháo một
lớp envelope của RunPod, không phải hiểu thêm một schema vector mới.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from pydantic import ValidationError

from server import EmbeddingRequest, EmbeddingRequestError, encode_request, encoder

logger = logging.getLogger(__name__)
MAX_BATCH_SIZE = int(os.getenv("EMBED_MAX_BATCH_SIZE", "64"))


class RunPodEmbeddingInputError(ValueError):
    """Job sai schema; ném lỗi để RunPod đánh dấu job FAILED."""


def _request_from_job(job: dict[str, Any]) -> EmbeddingRequest:
    if not isinstance(job, dict):
        raise RunPodEmbeddingInputError("job phải là một object")
    payload = job.get("input")
    if not isinstance(payload, dict):
        raise RunPodEmbeddingInputError("job.input phải là một object")
    try:
        return EmbeddingRequest.model_validate(payload)
    except ValidationError as exc:
        # Không đưa `input` vào lỗi: với ảnh, trường đó là chuỗi base64 rất lớn
        # và không nên bị nhân bản sang log/response của RunPod.
        details = exc.errors(include_url=False, include_input=False)
        raise RunPodEmbeddingInputError(f"job.input sai schema: {details}") from exc


def handler(job: dict[str, Any]) -> dict[str, Any]:
    request_id = job.get("id", "unknown") if isinstance(job, dict) else "unknown"
    request = _request_from_job(job)
    logger.info("Nhận job embedding id=%s, items=%d", request_id, len(request.input))
    try:
        return encode_request(request, encoder, max_batch_size=MAX_BATCH_SIZE)
    except EmbeddingRequestError as exc:
        raise RunPodEmbeddingInputError(str(exc)) from exc


if __name__ == "__main__":
    import runpod

    # Nạp đúng một lần trước khi worker nhận job. Nếu GPU/model cache sai thì
    # worker fail ở pha khởi động, thay vì nhận job rồi treo trong cold start.
    encoder.load()
    runpod.serverless.start({"handler": handler})
