"""Cửa tương thích: adapter thật đã chuyển sang `src.common_utils.text_embedding`.

Nhánh media cũng dùng chung adapter này cho caption/OCR/transcript, nên để nó
nằm trong `rag_noiquy` sẽ buộc media import ngược sang một package không liên
quan. Module này giữ nguyên đường import cũ để 5 chỗ đang dùng không phải sửa.
"""

from src.common_utils.text_embedding import (
    EmbeddingConfigurationError,
    EmbeddingService,
    EmbeddingServiceError,
    TextEmbeddingService,
)

__all__ = [
    "EmbeddingConfigurationError",
    "EmbeddingService",
    "EmbeddingServiceError",
    "TextEmbeddingService",
]
