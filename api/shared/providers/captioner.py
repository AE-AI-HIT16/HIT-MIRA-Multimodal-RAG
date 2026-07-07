"""Interface caption ảnh/frame (HỢP ĐỒNG cho sẵn). [BR-203]"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Captioner(ABC):
    @abstractmethod
    def caption(self, image_path: str) -> str:
        """Ảnh → câu mô tả tiếng Việt."""
        raise NotImplementedError


# TODO(sinh viên): impl bằng BLIP-2 / VLM.
