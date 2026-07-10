"""Sinh caption cho ảnh/frame thiếu mô tả (OFFLINE). [BR-203 · US-203.1 · T-12]

  Input : image_path (+ captioner, mặc định lấy từ app.deps)
  Output: caption tiếng Việt (str); lỗi/độ tin cậy thấp → "" để gắn cờ review, KHÔNG chặn
  Gợi ý : providers.captioner.Captioner.caption(...)
  Pass  : tests/test_pipeline.py::test_caption_generated
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def generate_caption(image_path: str, *, captioner: object | None = None) -> str:
    if captioner is None:
        from app.deps import get_captioner

        captioner = get_captioner()

    try:
        return captioner.caption(image_path)
    except Exception as exc:  # noqa: BLE001 — caption lỗi không được chặn pipeline
        log.warning("Bỏ caption cho %s (gắn cờ review): %s", image_path, exc)
        return ""
