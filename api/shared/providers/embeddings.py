"""Interface embedder (HỢP ĐỒNG) + impl model chốt 2026. [BR-202/207/208 · T-20]

Đổi model không đụng logic retrieval: chỉ cần class con của interface này.
  - Text (transcript & nội quy): AITeamVN/Vietnamese_Embedding (nền bge-m3).
  - Ảnh + query text CHUNG không gian: Jina-CLIP v2 (multilingual 89 thứ tiếng).
Import model nặng đặt trong __init__ (lazy) → `import` module không đòi torch.
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

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """1 câu hỏi text → 1 vector CÙNG không gian với ảnh (text→ảnh retrieval)."""
        raise NotImplementedError


class VietnameseTextEmbedder(TextEmbedder):
    """AITeamVN/Vietnamese_Embedding — retrieval-tuned từ bge-m3 (300k triplet). [T-20]"""

    def __init__(self, model_name: str = "AITeamVN/Vietnamese_Embedding",
                 device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True,
        )
        return vecs.tolist()


class JinaClipEmbedder(ImageEmbedder):
    """Jina-CLIP v2 — ảnh & query text CHUNG không gian, đa ngôn ngữ. [T-20]"""

    def __init__(self, model_name: str = "jinaai/jina-clip-v2",
                 device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, image_paths: list[str]) -> list[list[float]]:
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in image_paths]
        vecs = self._model.encode(images, normalize_embeddings=True, convert_to_numpy=True)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self._model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
        return vec[0].tolist()
