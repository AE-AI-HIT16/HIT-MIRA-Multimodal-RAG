"""Xác thực người dùng (LÕI HỌC). [BR-505 · NFR bảo mật]

Hợp đồng — sinh viên implement:
  authenticate : email+password → user (verify hash bcrypt/argon2)
  create_token : user → JWT (kèm role, hạn dùng)
  Ràng buộc: KHÔNG lưu mật khẩu thô.
  Pass : tests/test_auth.py
"""
from __future__ import annotations

from typing import Any


def authenticate(email: str, password: str) -> dict[str, Any] | None:
    raise NotImplementedError("US-505.1: verify hash mật khẩu")


def create_token(user: dict[str, Any]) -> str:
    raise NotImplementedError("US-505.1: phát JWT kèm role")
