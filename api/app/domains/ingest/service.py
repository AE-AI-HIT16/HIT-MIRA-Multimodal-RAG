"""Nạp dữ liệu nguồn (LÕI HỌC). [BR-102/104/105/107]

Hợp đồng — sinh viên implement:
  save_upload      : lưu file + metadata → posts/media_assets; thiếu trường bắt buộc → chặn
  load_regulations : nạp văn bản nội quy → tách điều/khoản (rule_chunks)
  Ràng buộc: mọi ingestion phải qua cổng require_consent (BR-101) — đã có ở deps.
  Pass : tests/test_ingest.py

Ghi file gốc qua storage provider (inject `store: StorageProvider = Depends(get_storage)`):
`store.put(key, data)` với key logic (vd 'media/<id>/<filename>'), rồi lưu key trả về
vào `media_assets.storage_key`. get_media đọc lại đúng key đó.
"""
from __future__ import annotations

from typing import Any


def save_upload(file_path: str, metadata: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError("US-102.1: lưu media + metadata, validate trường bắt buộc")


def load_regulations(file_path: str, meta: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError("US-107.1: tách nội quy theo điều/khoản")
