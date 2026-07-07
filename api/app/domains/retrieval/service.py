"""Truy xuất multimodal (LÕI HỌC). [BR-301/302/303/306/307/308]

Hợp đồng — sinh viên implement:
  retrieve_media       : text/ảnh → top-k media (embed query → search Qdrant media)
  retrieve_regulations : text → top-k rule_chunk (search collection nội quy)
  retrieve_by_transcript: text → đoạn video khớp lời nói, gộp keyframe theo video_id
  rank                 : xếp hạng giảm dần; mọi điểm < ngưỡng → coi như "không tìm thấy"
  Gợi ý : providers.embeddings + shared.vectorstore.qdrant.QdrantStore
  Pass  : tests/test_retrieval.py
"""
from __future__ import annotations

from typing import Any


def retrieve_media(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    raise NotImplementedError("US-301.1: embed query → search Qdrant media")


def retrieve_regulations(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    raise NotImplementedError("US-307.1: search collection nội quy")


def retrieve_by_transcript(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    raise NotImplementedError("US-308.1: khớp transcript + gộp keyframe theo video_id")


def rank(candidates: list[dict[str, Any]], threshold: float = 0.0) -> list[dict[str, Any]]:
    raise NotImplementedError("US-306.1: xếp hạng + ngưỡng 'không tìm thấy'")
