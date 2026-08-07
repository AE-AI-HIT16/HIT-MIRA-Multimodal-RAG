"""Unified Qwen vision service for keyframe OCR and captioning."""

from __future__ import annotations

import base64
import io
import json
import mimetypes
from pathlib import Path
from typing import Any

from src.configuration import AppConfig
from src.log.logger import logger
from src.rag_video_anh.schemas import CaptionResult, CaptionResultSet, KeyFrameSet, OCRResult, OCRResultSet, StageStatus


class QwenVisionService:
    """Generate OCR text and a caption for each keyframe with one API call."""

    backend = "openai-sdk-chat-completions"

    def __init__(self, config: AppConfig | None = None, client: Any | None = None) -> None:
        self.config = config or AppConfig()
        self.model_config = self.config.media_models
        self.prompt_config = self.config.media_prompts
        self.client = client
        self._runtime_loaded = client is not None
        self._load_error: str | None = None

    def analyze(
        self,
        *,
        media_id: str,
        keyframes: KeyFrameSet,
        ocr_policy: dict[str, Any] | None = None,
        caption_policy: dict[str, Any] | None = None,
    ) -> tuple[OCRResultSet, CaptionResultSet]:
        """Analyze every keyframe once and split the response into OCR/caption result sets."""

        if not keyframes.frames:
            return (
                OCRResultSet(media_id=media_id, status=StageStatus.SKIPPED, reason="no keyframes available for OCR"),
                CaptionResultSet(media_id=media_id, status=StageStatus.SKIPPED, reason="no keyframes available for captioning"),
            )

        runtime = self._load_runtime()
        if runtime is None:
            reason = self._load_error or "missing dependency 'openai'"
            return (
                OCRResultSet(media_id=media_id, status=StageStatus.SKIPPED, reason=reason),
                CaptionResultSet(media_id=media_id, status=StageStatus.SKIPPED, reason=reason),
            )

        ocr_results: list[OCRResult] = []
        caption_results: list[CaptionResult] = []
        prompt_version = self._prompt_version(caption_policy or {})
        for frame in keyframes.frames:
            try:
                image = frame.image_payload if frame.image_payload is not None else frame.image_path
                output = self._analyze_frame(image, ocr_policy or {}, caption_policy or {})
                ocr_text = self._clean_text(output.get("ocr_text"))
                caption_text = self._clean_text(output.get("caption_text"))
                meta = {
                    "model": self.model_name,
                    "backend": self.backend,
                    "prompt_version": prompt_version,
                    "max_tokens": self.max_tokens,
                    "ocr_blocks": output.get("ocr_blocks") or [],
                    "scene": output.get("scene") or "",
                    "objects": output.get("objects") or [],
                    "activities": output.get("activities") or [],
                    "keywords": output.get("keywords") or [],
                }
                ocr_results.append(
                    OCRResult(
                        frame_id=frame.frame_id,
                        full_text=ocr_text,
                        status=StageStatus.DONE if ocr_text else StageStatus.NOT_FOUND,
                        reason=None if ocr_text else "no visible text found",
                    )
                )
                caption_results.append(
                    CaptionResult(
                        frame_id=frame.frame_id,
                        caption_text=caption_text,
                        generation_meta=meta,
                        prompt_version=prompt_version,
                        status=StageStatus.DONE if caption_text else StageStatus.NOT_FOUND,
                        reason=None if caption_text else "empty caption",
                    )
                )
            except Exception as exc:
                reason = f"Qwen vision analysis failed: {self._exception_message(exc)}"
                ocr_results.append(OCRResult(frame_id=frame.frame_id, status=StageStatus.ERROR, reason=reason))
                caption_results.append(
                    CaptionResult(
                        frame_id=frame.frame_id,
                        generation_meta={"model": self.model_name, "backend": self.backend},
                        prompt_version=prompt_version,
                        status=StageStatus.ERROR,
                        reason=reason,
                    )
                )

        ocr_status = StageStatus.DONE if all(item.status != StageStatus.ERROR for item in ocr_results) else StageStatus.ERROR
        caption_status = StageStatus.DONE if all(item.status != StageStatus.ERROR for item in caption_results) else StageStatus.ERROR
        logger.info(f"Qwen vision processed {len(keyframes.frames)} frame(s) for media_id '{media_id}'")
        return (
            OCRResultSet(media_id=media_id, results=ocr_results, status=ocr_status),
            CaptionResultSet(media_id=media_id, results=caption_results, status=caption_status),
        )


    @staticmethod
    def _exception_message(exc: Exception) -> str:
        response = getattr(exc, "response", None)
        if response is not None:
            body = str(getattr(response, "text", "") or "").strip()
            if len(body) > 500:
                body = body[:500] + "..."
            if body:
                return f"{exc.__class__.__name__}: {exc}; response_body={body}"
        return f"{exc.__class__.__name__}: {exc}"

    def _load_runtime(self) -> Any | None:
        if self._runtime_loaded:
            return self.client
        api_key = self._clean_optional_config(getattr(self.model_config, "vision_api_key", None))
        base_url = self._clean_optional_config(getattr(self.model_config, "vision_api_base_url", None))
        model_name = self._clean_optional_config(getattr(self.model_config, "vision_model_name", None))
        if not api_key:
            self._runtime_loaded = True
            self._load_error = "missing vision API key. Set MEDIA_VISION_API_KEY"
            return None
        if not base_url:
            self._runtime_loaded = True
            self._load_error = "missing vision API base URL. Set MEDIA_VISION_API_BASE_URL"
            return None
        # Không còn tên model mặc định để rơi về, nên thiếu là phải báo thiếu ở
        # đây; để nó đi tiếp thì endpoint trả 404 model không tồn tại, khó lần
        # hơn hẳn so với một câu nói thẳng biến nào chưa khai.
        if not model_name:
            self._runtime_loaded = True
            self._load_error = "missing vision model name. Set MEDIA_VISION_MODEL_NAME"
            return None
        try:
            from openai import OpenAI
        except ImportError:
            self._runtime_loaded = True
            self._load_error = "missing dependency 'openai'"
            return None

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=float(getattr(self.model_config, "vision_api_timeout", 180.0) or 180.0),
            # Không để mặc định 2 của SDK: một ảnh hết giờ sẽ nhân ba cả thời
            # gian lẫn token rồi vẫn hỏng, trong khi mẻ caption đã tự chạy lại
            # được ở lượt sau.
            max_retries=self._so_lan_thu_lai(),
        )
        self._runtime_loaded = True
        return self.client

    def _analyze_frame(self, image: Any, ocr_policy: dict[str, Any], caption_policy: dict[str, Any]) -> dict[str, Any]:
        if image is None:
            raise ValueError("keyframe image is not available")
        tham_so_mo_rong = self._tham_so_mo_rong()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self._messages(image, ocr_policy, caption_policy),
            temperature=float(getattr(self.model_config, "vision_api_temperature", 0.0) or 0.0),
            max_tokens=self.max_tokens,
            **({"extra_body": tham_so_mo_rong} if tham_so_mo_rong else {}),
        )
        return self._parse_response(self._completion_text(response))

    def _tham_so_mo_rong(self) -> dict[str, Any]:
        """Tham số ngoài chuẩn OpenAI — chỉ gửi khi người vận hành khai rõ.

        `repetition_penalty` là phần mở rộng của vLLM. Thiếu nó thì ảnh nhiều
        chữ làm model lặp tới cụt token và JSON hỏng; nhưng gửi nó tới endpoint
        không hiểu thì ăn 400 và mất cả tầng caption. Nên trả dict rỗng khi
        chưa khai, và người bật phải biết backend của mình là vLLM.
        """

        penalty = getattr(self.model_config, "vision_repetition_penalty", None)
        try:
            gia_tri = float(penalty)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return {}
        return {"repetition_penalty": gia_tri} if gia_tri > 0 else {}

    def _messages(self, image: Any, ocr_policy: dict[str, Any], caption_policy: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._user_prompt(ocr_policy, caption_policy)},
                    {"type": "image_url", "image_url": {"url": self._image_to_data_url(image)}},
                ],
            },
        ]

    def _system_prompt(self) -> str:
        prompt = self._clean_optional_config(getattr(self.prompt_config, "vision_system_prompt", None))
        return prompt or "You analyze video keyframes. Return only valid minified JSON and no markdown."

    def _user_prompt(self, ocr_policy: dict[str, Any], caption_policy: dict[str, Any]) -> str:
        policy_prompt = self._clean_optional_config(caption_policy.get("prompt") or caption_policy.get("vision_prompt"))
        config_prompt = self._clean_optional_config(getattr(self.prompt_config, "vision_prompt", None))
        if policy_prompt or config_prompt:
            return policy_prompt or config_prompt or ""

        caption_instruction = self._clean_optional_config(getattr(self.prompt_config, "vision_caption_instruction", None)) or (
            "Write one factual Vietnamese caption describing the visible scene, people, objects, and context."
        )
        ocr_instruction = self._clean_optional_config(ocr_policy.get("prompt") or ocr_policy.get("vision_prompt")) or (
            "Extract all readable visible text exactly as it appears. Preserve Vietnamese accents when visible. "
            "Order text from top to bottom and left to right. Return an empty string if no text is visible."
        )
        return (
            "Analyze the attached keyframe once and return a JSON object with exactly these keys: "
            '"ocr_text", "ocr_blocks", "caption_text", "scene", "objects", "activities", and "keywords". '
            f"For ocr_text and ocr_blocks: {ocr_instruction} "
            f"For caption_text: {caption_instruction}"
        )

    @staticmethod
    def _completion_text(response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(getattr(item, "text", item)))
            return "".join(parts).strip()
        return str(content).strip()

    def _parse_response(self, content: str) -> dict[str, Any]:
        content = content.strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = self._loads_json_fragment(content)
        if not isinstance(parsed, dict):
            # Model định trả JSON nhưng hỏng giữa chừng — thường là lặp tới cụt
            # token. Nuốt nó thành caption thì hàng caption_results ghi DONE với
            # một khối JSON 9.000 ký tự: payload Qdrant nhiễm rác, mà lượt chạy
            # sau lại bỏ qua vì "đã có caption". Ném lên để thành FAILED, rồi lượt
            # sau tự thử lại — lỗi này ngẫu nhiên nên thử lại gần như luôn được.
            if content.startswith("{"):
                raise ValueError(f"JSON hỏng hoặc bị cắt ({len(content)} ký tự), có thể do model lặp tới hết token")
            # Không giống JSON thì là văn xuôi: có model trả thẳng câu mô tả,
            # nhánh này giữ nguyên cho chúng.
            return {"ocr_text": "", "ocr_blocks": [], "caption_text": self._clean_text(content)}

        ocr_payload = parsed.get("ocr") if isinstance(parsed.get("ocr"), dict) else {}
        return {
            "ocr_text": parsed.get("ocr_text") or ocr_payload.get("full_text") or parsed.get("ocr") or "",
            "ocr_blocks": self._string_list(parsed.get("ocr_blocks") or parsed.get("text_blocks") or ocr_payload.get("text_blocks")),
            "caption_text": parsed.get("caption_text") or parsed.get("caption") or "",
            "scene": self._clean_text(parsed.get("scene")),
            "objects": self._string_list(parsed.get("objects"))[:10],
            "activities": self._string_list(parsed.get("activities"))[:5],
            "keywords": self._string_list(parsed.get("keywords"))[:10],
        }

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()] if str(value).strip() else []

    @staticmethod
    def _loads_json_fragment(content: str) -> Any:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return None

    def _image_to_data_url(self, image: Any) -> str:
        if isinstance(image, (str, Path)):
            path = Path(image)
            mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime_type};base64,{encoded}"

        if isinstance(image, bytes):
            encoded = base64.b64encode(image).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"

        normalized = self._normalize_image(image)
        buffer = io.BytesIO()
        if hasattr(normalized, "save"):
            normalized.save(buffer, format="JPEG", quality=95)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"
        raise TypeError(f"unsupported image payload for Qwen vision API: {type(image).__name__}")

    @staticmethod
    def _normalize_image(image: Any) -> Any:
        if isinstance(image, str):
            try:
                from PIL import Image
            except ImportError:
                return image
            return Image.open(image).convert("RGB")

        try:
            import cv2
            import numpy as np
            from PIL import Image
        except ImportError:
            return image

        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                return Image.fromarray(image).convert("RGB")
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return image

    def _prompt_version(self, caption_policy: dict[str, Any]) -> str:
        version = caption_policy.get("prompt_version") or getattr(self.prompt_config, "vision_prompt_version", "qwen-vision-v1")
        return str(version).strip() or "qwen-vision-v1"

    @property
    def model_name(self) -> str:
        return str(getattr(self.model_config, "vision_model_name", "") or "")

    def _so_lan_thu_lai(self) -> int:
        """Số lần thử lại của SDK; 0 là hợp lệ nên không dùng `or`."""
        value = getattr(self.model_config, "vision_max_retries", 1)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 1

    @property
    def max_tokens(self) -> int:
        return int(getattr(self.model_config, "vision_max_tokens", 1024) or 1024)

    @staticmethod
    def _clean_text(value: Any) -> str:
        return " ".join(str(value or "").replace("<pad>", "").split()).strip()

    @staticmethod
    def _clean_optional_config(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text or (text.startswith("${") and text.endswith("}")):
            return None
        return text
