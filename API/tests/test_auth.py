"""Xác thực và phân quyền — TC-505 / US-505.1.

Chạy offline hoàn toàn: băm mật khẩu và JWT không cần mạng, không cần DB.
Phần chạm PostgreSQL (`/auth/register`) đã kiểm bằng tay trên hệ thống thật;
ở đây chốt những thứ **sai là rò rỉ**, tức là phần đáng có test tự động nhất.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from src.auth import security  # noqa: E402

KHOA_THU = "khoa-ky-du-dai-cho-test-1234567890"


@pytest.fixture(autouse=True)
def khoa_ky(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_SECRET_KEY", KHOA_THU)


# ------------------------------------------------------------ băm mật khẩu


def test_bam_khong_bao_gio_chua_mat_khau_tho() -> None:
    chuoi = security.bam_mat_khau("MatKhauSieuBiMat")
    assert "MatKhauSieuBiMat" not in chuoi


def test_kiem_dung_va_sai_mat_khau() -> None:
    chuoi = security.bam_mat_khau("matkhau12345")
    assert security.kiem_mat_khau("matkhau12345", chuoi) is True
    assert security.kiem_mat_khau("matkhau1234", chuoi) is False


def test_cung_mat_khau_ra_hai_chuoi_khac_nhau() -> None:
    """Mỗi lần một salt mới — nếu không, hai người cùng mật khẩu lộ ra nhau."""
    a = security.bam_mat_khau("matkhau12345")
    b = security.bam_mat_khau("matkhau12345")
    assert a != b
    assert security.kiem_mat_khau("matkhau12345", a)
    assert security.kiem_mat_khau("matkhau12345", b)


def test_chuoi_bam_hong_tra_false_chu_khong_nem_loi() -> None:
    """Dòng DB hỏng không được biến thành HTTP 500 — 500 và 401 khác nhau là rò rỉ."""
    for hong in ("", "rác", "scrypt$x$y$z$w$v", "bcrypt$1$2$3$4$5"):
        assert security.kiem_mat_khau("bất kỳ", hong) is False


def test_tham_so_scrypt_nam_trong_chuoi_de_sau_nay_nang_cap_duoc() -> None:
    """Có tham số trong chuỗi thì tăng chi phí băm mà vẫn kiểm được mật khẩu cũ."""
    chuoi = security.bam_mat_khau("matkhau12345")
    thuat_toan, n, r, p, _salt, _hash = chuoi.split("$")
    assert thuat_toan == "scrypt"
    assert (int(n), int(r), int(p)) == (security.SCRYPT_N, security.SCRYPT_R, security.SCRYPT_P)


def test_maxmem_du_cho_tham_so_dang_dung() -> None:
    """scrypt cần 128*n*r byte; vượt hạn mức OpenSSL là ValueError lúc chạy.

    Từng vỡ thật: n=2^15, r=8 cần 33,5 MB > mặc định 32 MB, và `/auth/register`
    trả 500 cho mọi lần đăng ký.
    """
    assert security.SCRYPT_MAXMEM >= 128 * security.SCRYPT_N * security.SCRYPT_R


# -------------------------------------------------------------------- JWT


def test_token_giai_ma_lai_dung_chu_the_va_quyen() -> None:
    token, song_giay = security.tao_token("a@b.com", "admin")
    payload = security.doc_token(token)
    assert payload is not None
    assert payload["sub"] == "a@b.com"
    assert payload["role"] == "admin"
    assert song_giay > 0


def test_token_ky_bang_khoa_khac_thi_bi_tu_choi(monkeypatch: pytest.MonkeyPatch) -> None:
    token, _ = security.tao_token("a@b.com", "admin")
    monkeypatch.setenv("AUTH_SECRET_KEY", "mot-khoa-hoan-toan-khac-1234567890")
    assert security.doc_token(token) is None


def test_token_het_han_bi_tu_choi() -> None:
    token, _ = security.tao_token("a@b.com", "user", gio_song=-1)
    assert security.doc_token(token) is None


def test_token_rac_tra_none_chu_khong_nem_loi() -> None:
    for rac in ("", "rác", "a.b.c", "Bearer x"):
        assert security.doc_token(rac) is None


def test_thieu_khoa_ky_thi_nem_loi_ro_rang(monkeypatch: pytest.MonkeyPatch) -> None:
    """Thiếu cấu hình phải nổ, không được âm thầm sinh khoá ngẫu nhiên.

    Sinh khoá mỗi lần khởi động thì mọi token chết sau mỗi lần restart, và
    triệu chứng ("đăng nhập xong vẫn bị đá ra") không hề trỏ về nguyên nhân.
    """
    monkeypatch.delenv("AUTH_SECRET_KEY", raising=False)
    with pytest.raises(security.AuthConfigurationError):
        security.tao_token("a@b.com", "user")

    monkeypatch.setenv("AUTH_SECRET_KEY", "ngan")
    with pytest.raises(security.AuthConfigurationError):
        security.tao_token("a@b.com", "user")
