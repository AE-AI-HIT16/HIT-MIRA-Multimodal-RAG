"""Phục vụ media + cắt clip (LÕI HỌC). [BR-105/403/404]

Hợp đồng — sinh viên implement:
  get_media  : id → file + metadata; id lạ → 404
  cut_clip   : (video, timestamp) → clip 3s trước + 3s sau (ffmpeg); sát biên → co lại
  stream_range: stream/seek video tới timestamp (HTTP range)
  Pass : tests/test_media.py

Đọc bytes qua storage provider (KHÔNG mở đường dẫn đĩa trực tiếp — để filesystem/minio
đổi được): inject `store: StorageProvider = Depends(get_storage)`, đọc theo
`media_assets.storage_key` → `store.open(key)`; `store.exists(key)` False mà bản ghi còn
tồn tại → file hỏng (500 + needs_review=True), khác với id không có (404).
"""
from __future__ import annotations

from typing import Any


def get_media(media_id: int) -> dict[str, Any]:
    raise NotImplementedError("US-105.1: trả file + metadata theo id")


def cut_clip(media_id: int, timestamp_sec: float) -> str:
    raise NotImplementedError("US-403.1: cắt clip ~6s quanh frame bằng ffmpeg")


def stream_range(media_id: int, timestamp_sec: float) -> Any:
    raise NotImplementedError("US-404.1: stream/seek tới timestamp")
