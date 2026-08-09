"""Adapter RunPod Serverless: biến giao thức hàng đợi thành giao thức Jina.

`ImageEmbeddingService` gửi một request rồi đọc `{"data": [...]}` ngay trong
response — đó là giao thức của `api.jina.ai` và của `embedding_server/server.py`.
RunPod Serverless không trả kết quả ngay: `/run` chỉ nhận job và trả về `id`,
kết quả nằm trong `output` của `/status/{id}` một lúc sau.

Adapter đặt ở tầng **vận chuyển** (`http_post=`) chứ không phải một client mới,
vì phần thân của client — chia lô theo dung lượng, thu nhỏ ảnh về 512px, đòi
đúng số vector, phân biệt lỗi tạm thời với lỗi vĩnh viễn — không liên quan gì
tới cách kết quả được chuyển về. Viết client thứ hai là nhân đôi đúng những chỗ
đó, và hai bản sẽ lệch nhau ở lần sửa thứ nhất.

Adapter cũng **không giữ khoá nào của riêng nó**: `Authorization` do client
truyền xuống được chuyển tiếp nguyên vẹn, nên chỉ có một chỗ duy nhất quyết
định khoá nào được dùng.

Không dùng cho truy vấn online khi `active workers = 0`: cold start đo được
khoảng 190 giây, đủ để treo request đầu tiên sau lúc rảnh.
"""

from __future__ import annotations

import json as json_module
import os
import time
from typing import Any, Callable

from src.configuration import AppConfig
from src.log.logger import logger

from .embedding_service import (
    ImageEmbeddingConfigurationError,
    ImageEmbeddingService,
)

DEFAULT_API_BASE_URL = "https://api.runpod.ai/v2"
# `/runsync` giữ kết nối tới ~90 giây rồi vẫn trả job chưa xong; `/run` + hỏi
# trạng thái cho hành vi giống nhau ở mọi độ dài job, và không có timeout HTTP
# nào phải đoán trước. Việc dùng chính là index hàng loạt, nơi một vòng hỏi
# thêm vài giây không đáng kể so với cold start.
RUN_PATH = "run"
STATUS_PATH = "status"
# Job đầu tiên gánh cold start (~190s đo được); giới hạn thực thi của endpoint
# đang đặt 600s. Hạn chờ mặc định phải bao được cả hai cộng thời gian xếp hàng.
DEFAULT_JOB_TIMEOUT_SECONDS = 900.0
# Mỗi lời gọi HTTP đơn lẻ chỉ là nhận job / hỏi trạng thái: nhanh và nhỏ. Đây
# KHÔNG phải hạn chờ của job.
DEFAULT_HTTP_TIMEOUT_SECONDS = 30.0
POLL_INITIAL_SECONDS = 0.5
POLL_MAX_SECONDS = 5.0
PENDING_JOB_STATUSES = frozenset({"IN_QUEUE", "IN_PROGRESS"})

# RunPod báo kết cục bằng chữ; client bên trên phân loại lỗi bằng mã HTTP. Bảng
# này là chỗ duy nhất dịch giữa hai cách nói, và mỗi dòng chọn mã theo đúng câu
# hỏi "thử lại có khá hơn không".
#   FAILED    -> 422: RunPod đã tự chạy lại job khi worker chết, nên cái còn
#                lại hầu hết là sai hợp đồng. Thử lại chỉ đốt thêm giây GPU.
#   TIMED_OUT -> 504: lần chạy đầu gánh cold start, lượt sau thì không; đây là
#                trường hợp hiếm hoi mà thử lại có cơ hội thật.
#   CANCELLED -> 409: có người chủ động huỷ. Tự động gửi lại là đi ngược lại
#                đúng ý định đó.
JOB_STATUS_TO_HTTP_CODE = {
    "FAILED": 422,
    "TIMED_OUT": 504,
    "CANCELLED": 409,
}
# Hết hạn chờ *ở phía mình* trong khi job có thể vẫn đang chạy. Không được thử
# lại: gửi lại là xếp thêm một job trùng và trả tiền hai lần cho cùng một lô.
LOCAL_DEADLINE_HTTP_CODE = 408


class RunPodTransportError(RuntimeError):
    """Lỗi ở tầng job, mang theo `response` để client phân loại như lỗi HTTP."""

    def __init__(self, message: str, response: "RunPodJobResponse") -> None:
        super().__init__(message)
        self.response = response


class RunPodJobResponse:
    """Đủ giống `httpx.Response` để client không phải biết RunPod tồn tại."""

    def __init__(self, status_code: int, payload: Any, *, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload

    @property
    def text(self) -> str:
        return json_module.dumps(self._payload, ensure_ascii=False, default=str)

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RunPodTransportError(f"RunPod job lỗi: HTTP {self.status_code} {self.text}", self)


def job_urls(url: str) -> tuple[str, str]:
    """Tách URL người dùng cấu hình thành (url nhận job, tiền tố hỏi trạng thái).

    Nhận cả `.../v2/<endpoint>`, `.../v2/<endpoint>/run` và `.../v2/<endpoint>/runsync`
    để một cấu hình chép từ trang RunPod dùng được ngay, không phải nhớ hậu tố nào.
    """
    trimmed = (url or "").strip().rstrip("/")
    if not trimmed:
        raise ImageEmbeddingConfigurationError("Thiếu URL endpoint RunPod.")
    if trimmed.endswith("/v1/embeddings"):
        # Đây là giao thức Jina trực tiếp của một Pod/FastAPI. Bọc nó vào
        # `{"input": ...}` sẽ ăn 422 rất khó đoán, nên chặn ngay tại cấu hình.
        raise ImageEmbeddingConfigurationError(
            f"URL '{url}' là endpoint Jina trực tiếp, không phải hàng đợi RunPod. "
            "Dùng thẳng ImageEmbeddingService, không cần adapter này."
        )
    base = trimmed
    for suffix in (f"/{RUN_PATH}", "/runsync"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return f"{base}/{RUN_PATH}", f"{base}/{STATUS_PATH}"


class RunPodEmbeddingTransport:
    """Callable dùng làm `http_post=` cho `ImageEmbeddingService`."""

    def __init__(
        self,
        *,
        job_timeout: float = DEFAULT_JOB_TIMEOUT_SECONDS,
        http_timeout: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
        http_post: Callable[..., Any] | None = None,
        http_get: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self.job_timeout = float(job_timeout)
        self.http_timeout = float(http_timeout)
        self._http_post = http_post
        self._http_get = http_get
        self._sleep = sleep or time.sleep
        self._monotonic = monotonic or time.monotonic

    def __call__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json: Any = None,
        timeout: float | None = None,
        **_: Any,
    ) -> Any:
        run_url, status_prefix = job_urls(url)
        # `timeout` của client là hạn chờ MỘT request Jina — mặc định 60 giây,
        # nhỏ hơn cả cold start. Ở hàng đợi, đại lượng tương ứng là hạn chờ cả
        # job, và nó là việc của adapter này, nên tham số kia bị bỏ qua có chủ ý.
        del timeout
        deadline = self._monotonic() + self.job_timeout

        submitted = self._post(run_url, headers=headers, json={"input": json}, timeout=self.http_timeout)
        # Lỗi HTTP lúc nhận job (401 sai khoá, 429, 5xx) trả nguyên vẹn cho
        # client: nó đã biết cách phân loại đúng những mã đó.
        if getattr(submitted, "status_code", 200) >= 400:
            return submitted

        body = submitted.json()
        if not isinstance(body, dict):
            raise RunPodTransportError(
                f"RunPod trả body không phải object khi nhận job: {body!r}",
                RunPodJobResponse(JOB_STATUS_TO_HTTP_CODE["FAILED"], body),
            )
        job_id = str(body.get("id") or "")
        state = self._terminal_state(body, job_id)
        if state is not None:
            return state
        if not job_id:
            raise RunPodTransportError(
                f"RunPod nhận job nhưng không trả 'id': {body!r}",
                RunPodJobResponse(JOB_STATUS_TO_HTTP_CODE["FAILED"], body),
            )

        return self._wait_for_job(f"{status_prefix}/{job_id}", job_id, headers=headers, deadline=deadline)

    def _wait_for_job(
        self, status_url: str, job_id: str, *, headers: dict[str, str] | None, deadline: float
    ) -> RunPodJobResponse:
        delay = POLL_INITIAL_SECONDS
        while True:
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                raise RunPodTransportError(
                    f"Quá hạn chờ job RunPod '{job_id}' sau {self.job_timeout:.0f}s. Job có thể "
                    "vẫn đang chạy — kiểm tra trên bảng điều khiển trước khi gửi lại, gửi lại "
                    "lúc này là xếp thêm một job trùng.",
                    RunPodJobResponse(LOCAL_DEADLINE_HTTP_CODE, {"id": job_id, "status": "PENDING"}),
                )
            self._sleep(min(delay, remaining))
            delay = min(delay * 2, POLL_MAX_SECONDS)

            response = self._get(status_url, headers=headers, timeout=self.http_timeout)
            status_code = getattr(response, "status_code", 200)
            if status_code >= 400:
                # Hỏi trạng thái hỏng không có nghĩa là job hỏng. 5xx/429 thường
                # là nhất thời, cứ hỏi lại cho tới khi hết hạn chờ.
                logger.warning(f"RunPod /status trả HTTP {status_code} cho job {job_id}; hỏi lại")
                continue
            body = response.json()
            if not isinstance(body, dict):
                logger.warning(f"RunPod /status trả body lạ cho job {job_id}: {body!r}; hỏi lại")
                continue
            state = self._terminal_state(body, job_id)
            if state is not None:
                return state

    def _terminal_state(self, body: dict[str, Any], job_id: str) -> RunPodJobResponse | None:
        """Trả response khi job đã ngã ngũ, `None` khi còn phải chờ."""
        status = str(body.get("status") or "").upper()
        if status in PENDING_JOB_STATUSES or not status:
            return None

        if status == "COMPLETED":
            output = body.get("output")
            if not isinstance(output, dict):
                raise RunPodTransportError(
                    f"Job RunPod '{job_id}' báo COMPLETED nhưng 'output' không phải object: {output!r}",
                    RunPodJobResponse(JOB_STATUS_TO_HTTP_CODE["FAILED"], body),
                )
            return RunPodJobResponse(200, output)

        code = JOB_STATUS_TO_HTTP_CODE.get(status, JOB_STATUS_TO_HTTP_CODE["FAILED"])
        raise RunPodTransportError(
            f"Job RunPod '{job_id}' kết thúc {status}: {self._failure_detail(body)}",
            RunPodJobResponse(code, body),
        )

    @staticmethod
    def _failure_detail(body: dict[str, Any]) -> str:
        for key in ("error", "output"):
            value = body.get(key)
            if isinstance(value, dict):
                value = value.get("error") or value.get("message") or value
            if value:
                return str(value)
        return "(không có mô tả lỗi)"

    def _post(self, *args: Any, **kwargs: Any) -> Any:
        if self._http_post is not None:
            return self._http_post(*args, **kwargs)
        return _httpx().post(*args, **kwargs)

    def _get(self, *args: Any, **kwargs: Any) -> Any:
        if self._http_get is not None:
            return self._http_get(*args, **kwargs)
        return _httpx().get(*args, **kwargs)


def _httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - phụ thuộc luôn có trong môi trường chạy
        raise ImageEmbeddingConfigurationError("Thiếu 'httpx' để gọi RunPod Serverless.") from exc
    return httpx


def runpod_endpoint_url(
    endpoint_id: str | None = None,
    *,
    api_base_url: str | None = None,
) -> str:
    endpoint = (endpoint_id or os.getenv("JINA_RUNPOD_ENDPOINT_ID") or "").strip()
    if not endpoint:
        raise ImageEmbeddingConfigurationError(
            "Thiếu JINA_RUNPOD_ENDPOINT_ID — endpoint Serverless của worker nhúng. "
            "RUNPOD_ENDPOINT_ID là worker video, không dùng được ở đây."
        )
    base = (api_base_url or os.getenv("RUNPOD_API_BASE_URL") or DEFAULT_API_BASE_URL).strip().rstrip("/")
    return f"{base}/{endpoint}/{RUN_PATH}"


def build_runpod_embedding_service(
    *,
    config: AppConfig | None = None,
    endpoint_id: str | None = None,
    api_key: str | None = None,
    api_base_url: str | None = None,
    job_timeout: float = DEFAULT_JOB_TIMEOUT_SECONDS,
    transport: RunPodEmbeddingTransport | None = None,
    **service_kwargs: Any,
) -> ImageEmbeddingService:
    """Dựng `ImageEmbeddingService` nói chuyện với endpoint Serverless.

    Khoá dùng ở đây là `RUNPOD_API_KEY` chứ không phải `JINA_API_KEY`: hai biến
    đó thuộc hai giao thức khác nhau và không được chép qua lại.
    """
    key = (api_key or os.getenv("RUNPOD_API_KEY") or "").strip()
    if not key:
        raise ImageEmbeddingConfigurationError("Thiếu RUNPOD_API_KEY để gọi endpoint Serverless.")

    service_kwargs.setdefault("base_url", runpod_endpoint_url(endpoint_id, api_base_url=api_base_url))
    service_kwargs.setdefault("api_key", key)
    # Tự host không có hạn mức token/phút của api.jina.ai; giữ nhịp ở đây chỉ
    # làm batch chậm lại trong khi vẫn trả tiền GPU theo giây.
    service_kwargs.setdefault("tokens_per_minute", 0)
    # Worker chặn job quá `EMBED_MAX_BATCH_SIZE` (mặc định 64) bằng lỗi hợp
    # đồng. Ràng mặc định của client vào đúng con số đó.
    service_kwargs.setdefault("text_batch_size", int(os.getenv("EMBED_MAX_BATCH_SIZE", "64")))
    return ImageEmbeddingService(
        config=config,
        http_post=transport or RunPodEmbeddingTransport(job_timeout=job_timeout),
        **service_kwargs,
    )
