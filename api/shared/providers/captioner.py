"""Interface caption ảnh/frame (HỢP ĐỒNG) + impl Gemini Vision. [BR-203 · T-22]

Không tốn GPU, ra tiếng Việt, dùng chung provider LLM. Import nặng lazy trong __init__.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

_PROMPT = (
    "Mô tả ngắn gọn bằng tiếng Việt nội dung chính trong ảnh này trong 1 câu, "
    "tập trung vào người/hoạt động/bối cảnh sự kiện. Không suy đoán thông tin không thấy."
)


class Captioner(ABC):
    @abstractmethod
    def caption(self, image_path: str) -> str:
        """Ảnh → câu mô tả tiếng Việt."""
        raise NotImplementedError


class GeminiVisionCaptioner(Captioner):
    """Gemini 2.5 Flash Vision → caption tiếng Việt. [T-22]"""

    def __init__(self, api_key: str, *, model: str = "gemini-2.5-flash",
                 client: object | None = None) -> None:
        if not api_key:
            raise RuntimeError("Thiếu LLM_API_KEY (đặt trong .env) để dùng Gemini Vision.")
        self.model_name = model
        if client is not None:  # inject fake khi test
            self._model = client
            return
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    def caption(self, image_path: str) -> str:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        resp = self._model.generate_content([_PROMPT, image])
        text = getattr(resp, "text", None)
        if not text:
            raise RuntimeError("Gemini Vision trả rỗng (bị chặn an toàn / hết quota).")
        return text.strip()
