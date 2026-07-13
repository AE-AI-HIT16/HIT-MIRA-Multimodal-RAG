"""Interface LLM sinh câu trả lời (HỢP ĐỒNG cho sẵn) + impl Gemini. [BR-401/407 · T-23]

Đặt sau abstraction để đổi Gemini ↔ GPT ↔ Claude không sửa domain answer.
Provider cụ thể (GeminiClient) chỉ phụ thuộc google-generativeai; wiring cấu hình
(api_key/model) nằm ở app.deps để `shared/` không phụ thuộc ngược vào `app/`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, temperature: float = 0.2) -> str:
        """prompt (đã chèn ngữ cảnh truy xuất) → câu trả lời text."""
        raise NotImplementedError


class GeminiClient(LLMClient):
    """Gemini 2.5 Flash / Flash-Lite qua google-generativeai (free-tier). [T-23]

    Import nặng đặt trong __init__ (lazy) để `import` module không đòi package.
    """

    def __init__(self, api_key: str, *, model: str = "gemini-2.5-flash",
                 client: object | None = None) -> None:
        if not api_key:
            raise RuntimeError("Thiếu LLM_API_KEY (đặt trong .env) để dùng Gemini.")
        self.model_name = model
        if client is not None:  # cho phép inject fake khi test
            self._model = client
            return
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    def generate(self, prompt: str, *, temperature: float = 0.2) -> str:
        resp = self._model.generate_content(
            prompt,
            generation_config={"temperature": temperature},
        )
        text = getattr(resp, "text", None)
        if not text:
            raise RuntimeError("Gemini trả rỗng (có thể bị chặn an toàn / hết quota).")
        return text.strip()
