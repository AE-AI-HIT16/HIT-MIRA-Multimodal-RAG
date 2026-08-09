"""Phân biệt lỗi nhất thời với lỗi nhà cung cấp từ chối vĩnh viễn.

Ngày 31/07/2026 tài khoản Jina hết số dư giữa mẻ index 59 video. Mọi lời gọi
nhúng trả 403 AUTHZ_INSUFFICIENT_BALANCE, nhưng `IndexingService` bắt hết
`Exception` để một keyframe hỏng không giết cả video — nên nó nuốt luôn lỗi
này. Kết quả: 23 video liên tiếp nhúng được 0 điểm mà script vẫn in "OK", và
chuyện đó chỉ lộ ra khi có người tình cờ hỏi chatbot.

Ràng buộc trong CLAUDE.md: mẻ chạy dài phải "fail loudly". Test này chốt đúng
ranh giới đó.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from src.rag_video_anh.embedding.embedding_service import (  # noqa: E402
    ImageEmbeddingProviderFatalError,
    ImageEmbeddingService,
    ImageEmbeddingServiceError,
)


class FakeResponse:
    """Đủ giống httpx.Response cho phần mã đang xét."""

    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.headers: dict[str, str] = {}
        self.text = str(self._payload)

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise FakeHTTPStatusError(response=self)


class FakeHTTPStatusError(Exception):
    def __init__(self, response: FakeResponse) -> None:
        super().__init__(f"HTTP {response.status_code}")
        self.response = response


def build_service(responses: list[FakeResponse]) -> tuple[ImageEmbeddingService, list[int]]:
    """Dịch vụ dùng http_post giả; trả kèm bộ đếm số lần gọi."""
    calls: list[int] = []

    def fake_post(*_args, **_kwargs) -> FakeResponse:
        calls.append(1)
        return responses[min(len(calls) - 1, len(responses) - 1)]

    service = ImageEmbeddingService(
        api_key="khoa-gia",
        base_url="https://api.jina.ai/v1/embeddings",
        model_name="jina-clip-v2",
        dimensions=1024,
        http_post=fake_post,
        tokens_per_minute=0,  # tắt bộ giữ nhịp để test không phải chờ thật
    )
    return service, calls


def test_het_so_du_nem_loi_chi_mang_va_khong_thu_lai() -> None:
    """403 hết số dư: nổi lên ngay, không phí ba lượt thử."""
    service, calls = build_service(
        [FakeResponse(403, {"detail": "Insufficient account balance.", "code": "AUTHZ_INSUFFICIENT_BALANCE"})]
    )
    with pytest.raises(ImageEmbeddingProviderFatalError) as caught:
        service.embed_texts(["một câu hỏi"])

    assert "Insufficient account balance" in str(caught.value)
    assert len(calls) == 1, "lỗi vĩnh viễn thì không được thử lại"


@pytest.mark.parametrize("status_code", [401, 402, 403])
def test_cac_ma_tu_choi_vinh_vien(status_code: int) -> None:
    """Khoá sai, hết tiền, bị chặn quyền — không cái nào tự khỏi."""
    service, _ = build_service([FakeResponse(status_code, {"detail": "nope"})])
    with pytest.raises(ImageEmbeddingProviderFatalError):
        service.embed_texts(["câu bất kỳ"])


def test_loi_nhat_thoi_van_thu_lai_va_nem_loi_thuong() -> None:
    """500 là lỗi nhất thời: vẫn thử lại, và không phải lỗi chí mạng."""
    service, calls = build_service([FakeResponse(500, {"detail": "server sập tạm"})])
    with pytest.raises(ImageEmbeddingServiceError) as caught:
        service.embed_texts(["một câu hỏi"])

    assert not isinstance(caught.value, ImageEmbeddingProviderFatalError)
    assert len(calls) == 3, "lỗi nhất thời phải được thử lại đủ số lượt"
