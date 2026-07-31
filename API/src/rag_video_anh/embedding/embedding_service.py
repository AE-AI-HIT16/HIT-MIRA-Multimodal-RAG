"""Media embedding services for video retrieval indexing."""

from __future__ import annotations

import base64
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

from src.configuration import AppConfig
from src.log.logger import logger

DEFAULT_IMAGE_BATCH_SIZE = 8
DEFAULT_TEXT_BATCH_SIZE = 64
# ~5MB base64 mỗi request, chừa chỗ cho phần bao JSON.
DEFAULT_MAX_REQUEST_BYTES = 5 * 1024 * 1024
# Hạn mức token của Jina tính theo phút -> chờ hết cửa sổ rồi thử lại.
RATE_LIMIT_COOLDOWN_SECONDS = 60.0
# Jina-CLIP v2 xử lý ảnh ở 512px; gửi to hơn chỉ tốn token chứ không lợi gì.
DEFAULT_MAX_IMAGE_SIDE = 512
DEFAULT_IMAGE_JPEG_QUALITY = 90
# Jina đếm token theo cửa sổ trượt 60 giây. Giữ nhịp dưới ngưỡng thay vì đâm
# vào 429 rồi ngủ bù: cách phản ứng đó đo được ~10 ảnh/phút, chủ động thì ~25.
DEFAULT_TOKENS_PER_MINUTE = 100_000
RATE_WINDOW_SECONDS = 60.0
# Mỗi ảnh 512px Jina tính tròn 4.000 token (đo trực tiếp từ trường usage).
TOKENS_PER_IMAGE = 4_000
# Không có số token thật cho text trước khi gửi; ~4 ký tự một token là đủ dùng.
CHARS_PER_TEXT_TOKEN = 4


class ImageEmbeddingConfigurationError(ValueError):
    """Raised when image embedding configuration is missing or invalid."""


class ImageEmbeddingServiceError(RuntimeError):
    """Raised when the image embedding provider returns invalid vectors."""


class ImageEmbeddingProviderFatalError(RuntimeError):
    """Nhà cung cấp từ chối theo cách mà thử lại không bao giờ khá hơn.

    Hết số dư, sai khoá, hết quota — mọi lời gọi sau đó cũng hỏng y hệt. Tách
    khỏi `ImageEmbeddingServiceError` có chủ đích: lỗi từng mục thì được nuốt
    để một keyframe hỏng không giết cả video, còn lỗi này phải nổi lên tận
    ngoài cùng. Không tách thì một tài khoản hết tiền trông giống hệt "ảnh này
    hỏng", và mẻ index chạy tiếp hàng chục video, nhúng được số không, rồi báo
    thành công.
    """


# 402/403: hết số dư hoặc bị chặn quyền. 401: khoá sai. Không cái nào tự khỏi.
FATAL_PROVIDER_STATUS_CODES = frozenset({401, 402, 403})


class ImageEmbeddingService:
    """Jina Embeddings API adapter for keyframe image and transcript text vectors."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
        timeout: float | None = None,
        config: AppConfig | None = None,
        http_post: Callable[..., Any] | None = None,
        image_batch_size: int | None = None,
        text_batch_size: int | None = None,
        max_request_bytes: int | None = None,
        max_image_side: int | None = None,
        image_jpeg_quality: int | None = None,
        tokens_per_minute: int | None = None,
    ) -> None:
        model_config = (config or AppConfig()).media_models
        self.model_name = (
            model_name
            or os.getenv("MEDIA_IMAGE_EMBEDDING_MODEL")
            or getattr(model_config, "clip_model_name", None)
            or "jina-clip-v2"
        )
        self.api_key = api_key if api_key is not None else getattr(model_config, "clip_api_key", None)
        self.base_url = (
            base_url
            or os.getenv("MEDIA_IMAGE_EMBEDDING_BASE_URL")
            or getattr(model_config, "clip_api_base_url", None)
            or "https://api.jina.ai/v1/embeddings"
        )
        self.dimensions = dimensions if dimensions is not None else getattr(model_config, "clip_embedding_dimensions", 1024)
        self.timeout = timeout if timeout is not None else getattr(model_config, "clip_api_timeout", 60.0)
        self.image_batch_size = self._positive_int(
            image_batch_size, os.getenv("MEDIA_IMAGE_EMBEDDING_BATCH_SIZE"), DEFAULT_IMAGE_BATCH_SIZE
        )
        self.text_batch_size = self._positive_int(
            text_batch_size, os.getenv("MEDIA_TEXT_EMBEDDING_BATCH_SIZE"), DEFAULT_TEXT_BATCH_SIZE
        )
        self.max_request_bytes = self._positive_int(
            max_request_bytes, os.getenv("MEDIA_EMBEDDING_MAX_REQUEST_BYTES"), DEFAULT_MAX_REQUEST_BYTES
        )
        # max_image_side <= 0 nghĩa là tắt thu nhỏ, nên không dùng _positive_int.
        self.max_image_side = self._int_or_default(
            max_image_side, os.getenv("MEDIA_IMAGE_EMBEDDING_MAX_SIDE"), DEFAULT_MAX_IMAGE_SIDE
        )
        self.image_jpeg_quality = self._positive_int(
            image_jpeg_quality, os.getenv("MEDIA_IMAGE_EMBEDDING_JPEG_QUALITY"), DEFAULT_IMAGE_JPEG_QUALITY
        )
        # <= 0 nghĩa là tắt hẳn việc giữ nhịp (tests dùng fake không cần chờ).
        self.tokens_per_minute = self._int_or_default(
            tokens_per_minute, os.getenv("MEDIA_EMBEDDING_TOKENS_PER_MINUTE"), DEFAULT_TOKENS_PER_MINUTE
        )
        self._token_window: deque[tuple[float, int]] = deque()
        self._http_post = http_post

    def _reserve_tokens(self, tokens: int) -> None:
        """Chờ vừa đủ để request sắp gửi không vượt hạn mức token mỗi phút.

        Rẻ hơn hẳn cách để Jina trả 429 rồi ngủ 60 giây: lúc đó cả cửa sổ bị
        bỏ phí, còn ở đây chỉ chờ đúng tới khi lô cũ nhất rơi khỏi cửa sổ.
        """
        if self.tokens_per_minute <= 0 or tokens <= 0:
            return

        while True:
            now = time.monotonic()
            cutoff = now - RATE_WINDOW_SECONDS
            while self._token_window and self._token_window[0][0] <= cutoff:
                self._token_window.popleft()

            used = sum(amount for _, amount in self._token_window)
            # Cửa sổ rỗng mà vẫn quá hạn mức thì có chờ cũng vô ích: cứ gửi và
            # để nhánh thử lại 429 lo, chờ mãi ở đây là treo cứng.
            if used + tokens <= self.tokens_per_minute or not self._token_window:
                self._token_window.append((now, tokens))
                return

            wait_for = self._token_window[0][0] + RATE_WINDOW_SECONDS - now
            logger.info(f"Chờ {wait_for:.1f}s để giữ dưới hạn mức {self.tokens_per_minute} token/phút")
            time.sleep(max(0.1, wait_for))

    @staticmethod
    def _int_or_default(explicit: int | None, from_env: str | None, fallback: int) -> int:
        for candidate in (explicit, from_env):
            if candidate is None or candidate == "":
                continue
            try:
                return int(candidate)
            except (TypeError, ValueError):
                continue
        return fallback

    @staticmethod
    def _positive_int(explicit: int | None, from_env: str | None, fallback: int) -> int:
        for candidate in (explicit, from_env):
            if candidate is None or candidate == "":
                continue
            try:
                value = int(candidate)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return fallback

    def embed_images(self, image_paths: list[str | Path]) -> list[list[float]]:
        paths = self._validate_image_paths(image_paths)
        logger.info(f"Embedding {len(paths)} keyframe image(s) with Jina model '{self.model_name}'")
        encoded = [self._image_as_base64(path) for path in paths]
        vectors: list[list[float]] = []
        # Gửi theo lô: một video có thể có hơn trăm keyframe, gộp hết vào một
        # request thì Jina trả 413 Payload Too Large.
        for batch in self._batch_images(encoded):
            self._reserve_tokens(len(batch) * TOKENS_PER_IMAGE)
            payload = {
                "model": self.model_name,
                "input": [{"image": item} for item in batch],
                "embedding_type": "float",
                "dimensions": self.dimensions,
                "normalized": True,
            }
            vectors.extend(self._embed_payload(payload, expected_count=len(batch), label="Image"))
        self._validate_embeddings(vectors, expected_count=len(paths))
        return vectors

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        clean_texts = self._validate_texts(texts)
        logger.info(f"Embedding {len(clean_texts)} transcript text chunk(s) with Jina model '{self.model_name}'")
        vectors: list[list[float]] = []
        for start in range(0, len(clean_texts), self.text_batch_size):
            batch = clean_texts[start : start + self.text_batch_size]
            self._reserve_tokens(sum(len(text) for text in batch) // CHARS_PER_TEXT_TOKEN + len(batch))
            payload = {
                "model": self.model_name,
                "input": [{"text": text} for text in batch],
                # jina-clip-v2 CHỈ nhận 'retrieval.query'; gửi 'retrieval.passage'
                # là 422. Nhúng bất đối xứng passage/query là của
                # jina-embeddings-v3, không phải model này.
                "task": "retrieval.query",
                "embedding_type": "float",
                "dimensions": self.dimensions,
                "normalized": True,
            }
            vectors.extend(self._embed_payload(payload, expected_count=len(batch), label="Text"))
        self._validate_embeddings(vectors, expected_count=len(clean_texts))
        return vectors

    def _batch_images(self, encoded_images: list[str]) -> list[list[str]]:
        """Chia ảnh thành lô theo cả số lượng lẫn tổng dung lượng base64.

        Chia theo số lượng thôi là chưa đủ: keyframe nặng nhẹ rất khác nhau, vài
        ảnh lớn cũng đủ vượt giới hạn kích thước request của nhà cung cấp.
        """
        batches: list[list[str]] = []
        current: list[str] = []
        current_bytes = 0
        for item in encoded_images:
            item_bytes = len(item)
            if current and (
                len(current) >= self.image_batch_size
                or current_bytes + item_bytes > self.max_request_bytes
            ):
                batches.append(current)
                current, current_bytes = [], 0
            # Ảnh đơn lẻ vượt hạn mức vẫn phải gửi một mình: không thể chia nhỏ hơn.
            if not current and item_bytes > self.max_request_bytes:
                logger.warning(
                    f"Keyframe base64 size {item_bytes} exceeds the request budget "
                    f"{self.max_request_bytes}; sending it alone"
                )
            current.append(item)
            current_bytes += item_bytes
        if current:
            batches.append(current)
        return batches

    def _embed_payload(self, payload: dict[str, Any], *, expected_count: int, label: str) -> list[list[float]]:
        self._validate_configuration()
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = self._post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                embeddings = self._response_embeddings(response.json())
            except ImageEmbeddingServiceError:
                raise
            except Exception as exc:
                fatal = self._fatal_provider_reason(exc)
                if fatal is not None:
                    raise ImageEmbeddingProviderFatalError(
                        f"{label} embedding provider refused the request and will keep "
                        f"refusing it: {fatal}"
                    ) from exc
                if attempt < max_attempts - 1 and self._is_retryable_provider_error(exc):
                    delay = self._retry_delay_seconds(exc, attempt)
                    logger.warning(
                        f"{label} embedding provider returned a retryable error; "
                        f"retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue
                raise ImageEmbeddingServiceError(
                    f"{label} embedding provider request failed: {exc.__class__.__name__}."
                ) from exc
            self._validate_embeddings(embeddings, expected_count=expected_count)
            return embeddings
        raise ImageEmbeddingServiceError(f"{label} embedding provider request failed after retries.")

    def _validate_configuration(self) -> None:
        if self._is_missing(self.api_key):
            raise ImageEmbeddingConfigurationError("Missing image embedding configuration. Set JINA_API_KEY.")
        if self._is_missing(self.base_url):
            raise ImageEmbeddingConfigurationError("Missing image embedding configuration. Set MEDIA_IMAGE_EMBEDDING_BASE_URL.")
        if self._is_missing(self.model_name):
            raise ImageEmbeddingConfigurationError("Missing image embedding configuration. Set MEDIA_IMAGE_EMBEDDING_MODEL.")
        if not isinstance(self.dimensions, int) or self.dimensions <= 0:
            raise ImageEmbeddingConfigurationError("Image embedding dimensions must be a positive integer.")

    def _post(self, *args: Any, **kwargs: Any) -> Any:
        if self._http_post is not None:
            return self._http_post(*args, **kwargs)
        try:
            import httpx
        except ImportError as exc:
            raise ImageEmbeddingConfigurationError("Missing dependency 'httpx' for Jina image embedding API.") from exc
        return httpx.post(*args, **kwargs)

    @staticmethod
    def _fatal_provider_reason(exc: Exception) -> str | None:
        """Trả về mô tả lỗi nếu nhà cung cấp từ chối vĩnh viễn, ngược lại None."""
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code not in FATAL_PROVIDER_STATUS_CODES:
            return None
        detail = ""
        try:
            body = response.json()
            detail = str(body.get("detail") or body.get("message") or body)
        except Exception:
            detail = str(getattr(response, "text", "") or "")
        return f"HTTP {status_code} {detail}".strip()

    @staticmethod
    def _is_retryable_provider_error(exc: Exception) -> bool:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        return status_code in {429, 500, 502, 503, 504}

    @staticmethod
    def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", {}) or {}
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), RATE_LIMIT_COOLDOWN_SECONDS)
            except ValueError:
                pass
        # Hạn mức của Jina tính theo token/PHÚT. Backoff 1s, 2s không giải quyết
        # được gì — phải chờ hết cửa sổ phút thì hạn mức mới nạp lại.
        if getattr(response, "status_code", None) == 429:
            return RATE_LIMIT_COOLDOWN_SECONDS
        return float(2 ** attempt)

    def _image_as_base64(self, path: Path) -> str:
        return base64.b64encode(self._downscaled_image_bytes(path)).decode("ascii")

    def _downscaled_image_bytes(self, path: Path) -> bytes:
        """Thu nhỏ keyframe trước khi gửi đi nhúng.

        Jina-CLIP v2 vốn xử lý ảnh ở 512px; gửi keyframe 1280x720 nguyên bản
        tốn 24.000 token/ảnh (hạn mức chỉ 100.000 token/phút) trong khi bản
        512px chỉ tốn 4.000 token mà vector gần như không đổi (cosine ~0.994).
        """
        raw = path.read_bytes()
        if self.max_image_side <= 0:
            return raw
        try:
            import io

            from PIL import Image
        except ImportError:
            # Thiếu Pillow thì vẫn nhúng được, chỉ tốn token hơn.
            logger.warning("Pillow is unavailable; embedding keyframes at full resolution")
            return raw

        try:
            with Image.open(io.BytesIO(raw)) as image:
                if max(image.size) <= self.max_image_side:
                    return raw
                resized = image.convert("RGB")
                resized.thumbnail((self.max_image_side, self.max_image_side), Image.LANCZOS)
                buffer = io.BytesIO()
                resized.save(buffer, format="JPEG", quality=self.image_jpeg_quality)
                return buffer.getvalue()
        except Exception as exc:
            # Ảnh hỏng/định dạng lạ: gửi nguyên bản, để nhà cung cấp quyết định.
            logger.warning(
                f"Could not downscale keyframe '{path.name}' ({exc.__class__.__name__}); "
                "sending it at full resolution"
            )
            return raw

    @classmethod
    def _response_embeddings(cls, response_data: Any) -> list[list[float]]:
        if not isinstance(response_data, dict) or not isinstance(response_data.get("data"), list):
            raise ImageEmbeddingServiceError("Image embedding API response does not contain a data list.")
        rows = response_data["data"]
        try:
            rows = sorted(rows, key=lambda row: int(row.get("index", 0)))
            return [cls._as_vector(row.get("embedding")) for row in rows]
        except (AttributeError, TypeError, ValueError) as exc:
            raise ImageEmbeddingServiceError("Image embedding API response contains invalid vectors.") from exc

    @staticmethod
    def _validate_image_paths(image_paths: list[str | Path]) -> list[Path]:
        if not isinstance(image_paths, list):
            raise TypeError("image_paths must be a list")
        if not image_paths:
            raise ValueError("image_paths must not be empty")
        paths = [Path(path) for path in image_paths]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"image file does not exist: {missing[0]}")
        return paths

    @staticmethod
    def _validate_texts(texts: list[str]) -> list[str]:
        if not isinstance(texts, list):
            raise TypeError("texts must be a list")
        if not texts:
            raise ValueError("texts must not be empty")
        clean_texts = [str(text).strip() for text in texts]
        if any(not text for text in clean_texts):
            raise ValueError("texts must not contain empty values")
        return clean_texts

    @staticmethod
    def _as_vector(embedding: Any) -> list[float]:
        if not isinstance(embedding, list):
            raise TypeError("embedding must be a list")
        return [float(value) for value in embedding]

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        if not isinstance(value, str):
            return False
        stripped = value.strip()
        return (
            not stripped
            or (stripped.startswith("${") and stripped.endswith("}"))
            or (stripped.startswith("your_") and stripped.endswith("_here"))
        )

    def _validate_embeddings(self, embeddings: list[list[float]], expected_count: int) -> None:
        if len(embeddings) != expected_count:
            raise ImageEmbeddingServiceError(
                f"Image embedding count mismatch: expected {expected_count}, got {len(embeddings)}."
            )
        dimensions = {len(embedding) for embedding in embeddings if embedding}
        if len(dimensions) != 1:
            raise ImageEmbeddingServiceError("Image embedding dimensions are not consistent.")
        observed_dimension = dimensions.pop()
        if observed_dimension != self.dimensions:
            raise ImageEmbeddingServiceError(
                f"Image embedding dimension mismatch: expected {self.dimensions}, got {observed_dimension}."
            )
