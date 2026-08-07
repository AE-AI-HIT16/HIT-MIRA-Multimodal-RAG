"""Chuẩn hoá tên tiếng Việt thành khoá tra cứu ổn định.

Dùng chung bởi hai tầng: `repository/events.py` sinh ra `event_series.slug` khi
ghi DB, còn `vector_store.py` phải sinh ra ĐÚNG chuỗi đó khi lọc theo sự kiện.
Hai bản cài đặt riêng sẽ trôi khỏi nhau, và triệu chứng là bộ lọc trả về rỗng
cho một sự kiện có thật — không có lỗi nào được ném ra.
"""

from __future__ import annotations

import re
import unicodedata

MAX_SLUG_LENGTH = 255


def bo_dau(value: str) -> str:
    """Bỏ dấu tiếng Việt, giữ lại chữ cái.

    `đ`/`Đ` (U+0111/U+0110) là chữ cái riêng chứ không phải `d` + dấu, nên NFKD
    KHÔNG tách nó ra — `encode("ascii", "ignore")` sẽ XOÁ HẲN nó. Không đổi trước
    thì "đại hội" thành "ai hoi", và "Đại hội" với "Dai hoi" ra hai khoá khác
    nhau. Đúng cái bẫy đã làm hỏng bộ lọc boilerplate của ASR (xem
    `asr_service._normalize_transcript_text`).
    """
    khong_gach = value.replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFKD", khong_gach).encode("ascii", "ignore").decode("ascii")


def normalize_alias(value: str) -> str:
    """Dạng chuẩn để so hai cách gọi có phải cùng một sự kiện không.

    "HIT Open Day", "hit open day", "HIT  OPEN DAY" → cùng một chuỗi. Đây là thứ
    được ghi vào `event_aliases.normalized_alias` (UNIQUE), nên nó chính là cơ
    chế chống việc trích xuất tạo ra ba series cho một sự kiện.
    """
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", bo_dau(value).lower()).strip()


def slugify(value: str) -> str:
    """Khoá `event_key` xuất hiện trong payload Qdrant, nên phải là ASCII an toàn."""
    if not isinstance(value, str):
        return ""
    thuong = re.sub(r"[^a-z0-9]+", "-", bo_dau(value).lower())
    return thuong.strip("-")[:MAX_SLUG_LENGTH]
