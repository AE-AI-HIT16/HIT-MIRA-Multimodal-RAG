"""Sinh embedding cho ảnh/frame/text (OFFLINE). [BR-202/207/208]

Hợp đồng — sinh viên implement:
  Input : danh sách ảnh HOẶC text + provider tương ứng
  Output: list vector cùng dim/model; ảnh decode fail → bỏ qua + log
  Gợi ý : providers.embeddings.ImageEmbedder / TextEmbedder
  Pass  : tests/test_pipeline.py::test_embed_same_dim
"""
from __future__ import annotations


def embed_images(image_paths: list[str]) -> list[list[float]]:
    raise NotImplementedError("US-202.1: sinh viên nhúng ảnh (CLIP)")


def embed_texts(texts: list[str]) -> list[list[float]]:
    raise NotImplementedError("US-202.1/207.1/208.1: sinh viên nhúng text")
