"""Cấu hình chung cho test của `API/`.

`src/config/config.py` gọi `load_dotenv()` ngay lúc import, nên **file `.env`
thật của máy dev chảy thẳng vào tiến trình pytest**. `ImageEmbeddingService`
đọc mặc định của nó bằng `os.getenv`, nên một thay đổi vận hành hoàn toàn
chính đáng — trỏ `MEDIA_IMAGE_EMBEDDING_BASE_URL` sang embedding server tự host
— làm đổ ba test khẳng định hành vi mặc định, dù không dòng code nào đổi.

Test phải nói về code, không nói về `.env` của người đang chạy nó. Fixture dưới
đây gỡ đúng nhóm biến mà `ImageEmbeddingService` đọc, để "mặc định" trong test
nghĩa là mặc định ghi trong code.
"""

from __future__ import annotations

import pytest

# Đúng danh sách `embedding_service.py` đọc. Cố ý liệt kê tường minh thay vì xoá
# theo tiền tố `MEDIA_`: nhánh vision (`MEDIA_VISION_*`) có test riêng cần giữ.
BIEN_MOI_TRUONG_NHUNG = (
    "MEDIA_IMAGE_EMBEDDING_MODEL",
    "MEDIA_IMAGE_EMBEDDING_BASE_URL",
    "MEDIA_IMAGE_EMBEDDING_BATCH_SIZE",
    "MEDIA_IMAGE_EMBEDDING_MAX_SIDE",
    "MEDIA_IMAGE_EMBEDDING_JPEG_QUALITY",
    "MEDIA_TEXT_EMBEDDING_BATCH_SIZE",
    "MEDIA_EMBEDDING_MAX_REQUEST_BYTES",
    "MEDIA_EMBEDDING_TOKENS_PER_MINUTE",
)


@pytest.fixture(autouse=True)
def moi_truong_nhung_sach(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gỡ cấu hình nhúng của máy thật khỏi mọi test."""
    for ten in BIEN_MOI_TRUONG_NHUNG:
        monkeypatch.delenv(ten, raising=False)
