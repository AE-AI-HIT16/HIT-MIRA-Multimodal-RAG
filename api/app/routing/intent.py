"""Router RAG — phân loại ý định câu hỏi → chọn nguồn/tool  [BR-507].

Chiến lược (khớp NFR "Router (định tuyến)"):
  1. LUẬT trước (rẻ, deterministic): có ảnh → media; từ khóa nội quy → regulation.
  2. CLASSIFIER fallback khi tín hiệu luật không rõ.
  3. Cho phép ÉP chế độ thủ công (override).
KHÔNG phân rã/multi-hop ở v1 — đó là Agentic RAG (v2).
"""
from __future__ import annotations

from enum import Enum


class Source(str, Enum):
    MEDIA = "media"
    REGULATION = "regulation"
    BOTH = "both"


REGULATION_KEYWORDS = (
    "nội quy", "quy chế", "điều", "khoản", "quy định", "được phép", "cấm", "vi phạm",
)


def route(text: str | None, has_image: bool, override: Source | None = None) -> Source:
    """Trả về nguồn cần định tuyến. Deterministic & log được."""
    if override is not None:
        return override
    if has_image:
        return Source.MEDIA
    t = (text or "").lower()
    if any(kw in t for kw in REGULATION_KEYWORDS):
        return Source.REGULATION
    # TODO(v1): classifier fallback khi luật không bắt được (BR-507 / đo ở BR-606).
    return Source.MEDIA
