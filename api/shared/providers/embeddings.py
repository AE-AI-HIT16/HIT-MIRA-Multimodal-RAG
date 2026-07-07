"""Interface embedder (HỢP ĐỒNG cho sẵn — impl là bài của sinh viên).

Đổi model không đụng logic retrieval: chỉ cần class con của interface này.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class TextEmbedder(ABC):
    """Nhúng text tiếng Việt (dùng cho transcript & nội quy). [BR-202/207/208]"""

    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """list text → list vector (cùng `dim`)."""
        raise NotImplementedError


class ImageEmbedder(ABC):
    """Nhúng ảnh vào CÙNG không gian với text (multilingual CLIP). [BR-202]"""

    dim: int

    @abstractmethod
    def embed(self, image_paths: list[str]) -> list[list[float]]:
        """list đường dẫn ảnh → list vector."""
        raise NotImplementedError


# TODO(sinh viên): impl bằng open-clip-torch (ảnh) + sentence-transformers (text VN).
