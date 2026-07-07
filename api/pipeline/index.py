"""Nạp embedding vào Qdrant (OFFLINE). [BR-205/206 · US-205.1/206.1]

Hợp đồng — sinh viên implement:
  Input : embedding + payload (target_type, target_id, ...) + tên collection
  Output: dữ liệu vào đúng 1 trong 3 collection (media/transcript/nội quy)
  Gợi ý : shared.vectorstore.qdrant.QdrantStore (ensure_collection → upsert)
  Edge  : sai chiều vector → từ chối + báo; upsert trùng id → ghi đè
  Pass  : tests/test_pipeline.py::test_index_upsert
"""
from __future__ import annotations

from typing import Any


def build_index(collection: str, dim: int, ids: list[int | str],
                vectors: list[list[float]], payloads: list[dict[str, Any]]) -> None:
    raise NotImplementedError("US-205.1: sinh viên nạp embedding vào Qdrant")
