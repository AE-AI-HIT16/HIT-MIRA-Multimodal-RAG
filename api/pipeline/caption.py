"""Sinh caption cho ảnh/frame thiếu mô tả (OFFLINE). [BR-203 · US-203.1]

Hợp đồng — sinh viên implement:
  Input : image_path
  Output: caption tiếng Việt (str); độ tin cậy thấp → gắn cờ để review
  Gợi ý : providers.captioner.Captioner.caption(...)
  Pass  : tests/test_pipeline.py::test_caption_generated
"""
from __future__ import annotations


def generate_caption(image_path: str) -> str:
    raise NotImplementedError("US-203.1: sinh viên sinh caption ảnh/frame")
