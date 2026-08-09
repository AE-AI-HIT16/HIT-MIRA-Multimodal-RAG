"""Băm mật khẩu và cấp/kiểm token — T-51 / US-505.1.

**Băm bằng `hashlib.scrypt` của thư viện chuẩn.** argon2 hay bcrypt tốt hơn về
lý thuyết, nhưng thêm một phụ thuộc gốc C vào một dự án đồ án là thêm một thứ
có thể vỡ khi cài trên máy khác; scrypt có sẵn trong Python và với tham số dưới
đây thì đủ mạnh cho phạm vi này.

**Không bao giờ lưu mật khẩu thô, và cũng không lưu hash trần.** Mỗi mật khẩu
một salt ngẫu nhiên 16 byte, ghi kèm trong chuỗi kết quả theo dạng
`scrypt$n$r$p$<salt_b64>$<hash_b64>` — có tham số trong chuỗi thì sau này tăng
chi phí băm mà vẫn kiểm được mật khẩu cũ.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

# n=2^15 nặng khoảng vài chục ms mỗi lần kiểm — đủ chậm để dò mật khẩu hàng loạt
# là vô vọng, đủ nhanh để đăng nhập không thấy trễ.
SCRYPT_N = 1 << 15
SCRYPT_R = 8
SCRYPT_P = 1
DO_DAI_KHOA = 32
# scrypt cần 128 * n * r byte = 33,5 MB với tham số trên, vượt hạn mức mặc định
# 32 MB của OpenSSL → `ValueError: memory limit exceeded`. Phải khai `maxmem`
# tường minh; hạ n xuống cho vừa 32 MB cũng được nhưng là tự làm yếu hàm băm vì
# một con số mặc định của thư viện.
SCRYPT_MAXMEM = 128 * SCRYPT_N * SCRYPT_R * 2
THUAT_TOAN_JWT = "HS256"
HAN_TOKEN_GIO_MAC_DINH = 12


class AuthConfigurationError(RuntimeError):
    """Thiếu cấu hình bắt buộc — phải nổ lúc khởi động, không âm thầm bỏ qua."""


def bam_mat_khau(mat_khau: str) -> str:
    salt = secrets.token_bytes(16)
    khoa = hashlib.scrypt(
        mat_khau.encode(),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        maxmem=SCRYPT_MAXMEM,
        dklen=DO_DAI_KHOA,
    )
    return "$".join(
        [
            "scrypt",
            str(SCRYPT_N),
            str(SCRYPT_R),
            str(SCRYPT_P),
            base64.b64encode(salt).decode(),
            base64.b64encode(khoa).decode(),
        ]
    )


def kiem_mat_khau(mat_khau: str, chuoi_bam: str) -> bool:
    try:
        thuat_toan, n, r, p, salt_b64, khoa_b64 = chuoi_bam.split("$")
        if thuat_toan != "scrypt":
            return False
        khoa = hashlib.scrypt(
            mat_khau.encode(),
            salt=base64.b64decode(salt_b64),
            n=int(n),
            r=int(r),
            p=int(p),
            # Tham số đọc từ chính chuỗi đã lưu, nên maxmem phải tính theo đó —
            # hằng số cứng sẽ chặn mất mật khẩu băm bằng tham số nặng hơn.
            maxmem=128 * int(n) * int(r) * 2,
            dklen=len(base64.b64decode(khoa_b64)),
        )
    except (ValueError, TypeError):
        # Chuỗi hỏng thì coi như sai mật khẩu, không ném lỗi ra ngoài — thông
        # báo lỗi khác nhau giữa "sai định dạng" và "sai mật khẩu" là rò rỉ.
        return False
    # So sánh hằng thời gian: so bằng `==` để lộ độ dài tiền tố khớp qua thời gian.
    return hmac.compare_digest(khoa, base64.b64decode(khoa_b64))


def _khoa_ky() -> str:
    khoa = os.getenv("AUTH_SECRET_KEY") or ""
    if not khoa or len(khoa) < 16:
        # Sinh khoá ngẫu nhiên khi thiếu là cái bẫy tệ nhất: chạy được ở máy dev,
        # rồi mọi token chết mỗi lần khởi động lại, và không ai hiểu vì sao.
        raise AuthConfigurationError(
            "Thiếu AUTH_SECRET_KEY (tối thiểu 16 ký tự). Sinh một khoá: "
            "python -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )
    return khoa


def tao_token(subject: str, role: str, gio_song: int | None = None) -> tuple[str, int]:
    """Cấp JWT. Trả về (token, số giây còn sống)."""
    import jwt

    gio = gio_song or int(os.getenv("AUTH_TOKEN_TTL_HOURS") or HAN_TOKEN_GIO_MAC_DINH)
    het_han = datetime.now(timezone.utc) + timedelta(hours=gio)
    payload = {"sub": subject, "role": role, "exp": het_han}
    return jwt.encode(payload, _khoa_ky(), algorithm=THUAT_TOAN_JWT), gio * 3600


def doc_token(token: str) -> dict[str, Any] | None:
    """Giải mã và kiểm hạn. Trả None nếu token hỏng, sai chữ ký hoặc hết hạn."""
    import jwt

    try:
        # `algorithms` phải khai tường minh: để thư viện tự chọn theo header là
        # mở đường cho token ký bằng thuật toán "none".
        return jwt.decode(token, _khoa_ky(), algorithms=[THUAT_TOAN_JWT])
    except Exception:
        return None
