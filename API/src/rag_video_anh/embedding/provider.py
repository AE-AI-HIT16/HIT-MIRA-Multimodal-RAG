"""Chọn đường nhúng: giao thức Jina trực tiếp hay hàng đợi RunPod Serverless.

Hai cách triển khai cùng một model, khác nhau ở chỗ kết quả về bằng lối nào:

* **jina** — một endpoint HTTP luôn sống, trả vector ngay trong response. Là
  `api.jina.ai`, hoặc `embedding_server/server.py` chạy trên Pod.
* **runpod** — endpoint Serverless `active workers = 0`. Rẻ khi rảnh, nhưng
  lần gọi đầu sau lúc rảnh gánh cold start đo được khoảng 190 giây.

Gom việc chọn về một chỗ vì trước đó có ba chỗ tự dựng client, và mỗi chỗ thêm
một dòng `if` là ba cơ hội để chúng lệch nhau.
"""

from __future__ import annotations

import os
from typing import Any

from src.configuration import AppConfig
from src.log.logger import logger

from .embedding_service import ImageEmbeddingConfigurationError, ImageEmbeddingService
from .runpod_transport import build_runpod_embedding_service

PROVIDER_JINA = "jina"
PROVIDER_RUNPOD = "runpod"
SUPPORTED_PROVIDERS = (PROVIDER_JINA, PROVIDER_RUNPOD)

# Hạn chờ job khi truy vấn online lỡ phải đi qua hàng đợi. Đủ bao cold start
# (~190s) cộng chút dư, chứ không phải 900 giây của batch: một người đang chờ
# khung chat mà treo 15 phút thì thà báo lỗi sớm.
ONLINE_JOB_TIMEOUT_SECONDS = 240.0


def _da_cau_hinh(value: Any) -> bool:
    """Dùng đúng luật mà client dùng khi validate, để dò và validate không lệch.

    `${...}` chưa thay và `your_..._here` đều tính là chưa cấu hình — nếu không,
    một placeholder sót lại trong YAML sẽ được coi là "đã có endpoint trực tiếp".
    """
    return not ImageEmbeddingService._is_missing(value)


def resolve_media_embedding_provider(config: AppConfig | None = None) -> str:
    """Trả về `'jina'` hoặc `'runpod'` theo cấu hình hiện có.

    Tự dò thay vì bắt khai tường minh: `.env` thực tế chỉ điền đúng một trong
    hai nhóm biến, và một biến bắt buộc nữa là một chỗ nữa để quên. `MEDIA_EMBEDDING_PROVIDER`
    vẫn thắng khi cần ép — ví dụ đã có Pod trực tiếp nhưng muốn đẩy mẻ index
    sang GPU.
    """
    explicit = (os.getenv("MEDIA_EMBEDDING_PROVIDER") or "").strip().lower()
    if explicit:
        if explicit not in SUPPORTED_PROVIDERS:
            raise ImageEmbeddingConfigurationError(
                f"MEDIA_EMBEDDING_PROVIDER='{explicit}' không hợp lệ; chọn một trong {SUPPORTED_PROVIDERS}."
            )
        return explicit

    app_config = config or AppConfig()
    # Đúng thứ tự `ImageEmbeddingService.__init__` dùng: biến môi trường trước,
    # YAML sau. `AppConfig` khai giá trị này làm default của dataclass nên nó bị
    # chốt ngay lúc import module — đọc mỗi YAML thì một `.env` nạp muộn sẽ được
    # dò một đằng còn client dựng một nẻo.
    base_url = os.getenv("MEDIA_IMAGE_EMBEDDING_BASE_URL") or getattr(
        getattr(app_config, "media_models", None), "clip_api_base_url", None
    )
    # Endpoint trực tiếp thắng khi khai cả hai: nó không có cold start, và đó
    # cũng là hành vi trước khi có adapter — môi trường nào đang chạy được thì
    # vẫn chạy y như cũ.
    if _da_cau_hinh(base_url):
        return PROVIDER_JINA
    if _da_cau_hinh(os.getenv("JINA_RUNPOD_ENDPOINT_ID")):
        return PROVIDER_RUNPOD
    # Không khai gì cả: vẫn trả về nhánh trực tiếp để người vận hành nhận đúng
    # thông báo cũ ("Set JINA_API_KEY" / "Set MEDIA_IMAGE_EMBEDDING_BASE_URL")
    # thay vì một lỗi mới nói về RunPod mà họ chưa hề định dùng.
    return PROVIDER_JINA


def build_media_embedder(
    config: AppConfig | None = None,
    *,
    for_online_queries: bool = False,
    **kwargs: Any,
) -> ImageEmbeddingService:
    """Dựng client nhúng đã chọn sẵn đường đi.

    `for_online_queries=True` cho đường truy vấn của người dùng. Nó không đổi
    lựa chọn — chặn ở đây thì demo không còn gì để chạy — nhưng rút ngắn hạn chờ
    và **kêu to** khi câu hỏi phải đi qua hàng đợi, vì đó đúng là ranh giới
    online/offline mà NFR cấm vượt.
    """
    app_config = config or AppConfig()
    provider = resolve_media_embedding_provider(app_config)

    if provider == PROVIDER_RUNPOD:
        if for_online_queries:
            logger.warning(
                "Truy vấn online đang đi qua RunPod Serverless. Endpoint để "
                "active workers = 0 thì câu hỏi đầu tiên sau lúc rảnh phải chờ "
                "cold start (~190s). Chỉ chấp nhận tạm; đường online nên trỏ "
                "vào một endpoint luôn sống."
            )
            kwargs.setdefault("job_timeout", ONLINE_JOB_TIMEOUT_SECONDS)
        logger.info("Nhúng media qua RunPod Serverless (JINA_RUNPOD_ENDPOINT_ID)")
        return build_runpod_embedding_service(config=app_config, **kwargs)

    logger.info("Nhúng media qua giao thức Jina trực tiếp (MEDIA_IMAGE_EMBEDDING_BASE_URL)")
    return ImageEmbeddingService(config=app_config, **kwargs)
