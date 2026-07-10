"""Sinh embedding cho ảnh/frame/text (OFFLINE). [BR-202/207/208 · T-13]

Input : danh sách ảnh HOẶC text (+ provider tương ứng, mặc định lấy từ app.deps)
Output: list vector cùng dim/model; ảnh không đọc được → bỏ qua + log (không chặn)
Gợi ý : providers.embeddings.ImageEmbedder / TextEmbedder
Pass  : tests/test_pipeline.py::test_embed_same_dim
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def embed_images(image_paths: list[str], *, embedder: object | None = None) -> list[list[float]]:
    """Nhúng ảnh/frame (CLIP). Bỏ qua file không tồn tại; vector trả về theo thứ tự file hợp lệ."""
    if embedder is None:
        from app.deps import get_image_embedder

        embedder = get_image_embedder()

    valid = [p for p in image_paths if os.path.exists(p)]
    for p in image_paths:
        if p not in valid:
            log.warning("Bỏ qua ảnh không đọc được: %s", p)
    if not valid:
        return []
    return embedder.embed(valid)


def embed_texts(texts: list[str], *, embedder: object | None = None) -> list[list[float]]:
    """Nhúng text (transcript/nội quy) bằng Vietnamese_Embedding."""
    if embedder is None:
        from app.deps import get_text_embedder

        embedder = get_text_embedder()

    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return []
    return embedder.embed(texts)
