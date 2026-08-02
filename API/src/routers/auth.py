"""Đăng ký / đăng nhập / xem tài khoản — T-51 / US-505.1 / TC-505.

Phạm vi cố ý hẹp: đủ để phân biệt `admin` với người dùng thường và khoá màn
admin lại, không hơn. Không đổi mật khẩu, không refresh token, không quên mật
khẩu — những thứ đó chỉ có nghĩa khi hệ thống có người dùng thật.

**Tài khoản đầu tiên đăng ký sẽ là `admin`.** Không có seed thì không ai vào
được màn admin, mà thêm một script seed riêng lại là thêm một chỗ giấu mật khẩu
mặc định. Từ tài khoản thứ hai trở đi đều là `user`.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from src.auth.security import bam_mat_khau, doc_token, kiem_mat_khau, tao_token
from src.rag_video_anh.repository.database import get_session_manager

router = APIRouter(prefix="/auth", tags=["auth"])

DO_DAI_MAT_KHAU_TOI_THIEU = 8

# Kiểm email bằng regex thay vì `pydantic.EmailStr`: EmailStr kéo theo gói
# `email-validator`, thêm một phụ thuộc chỉ để chặn chuỗi thiếu dấu @ thì không
# đáng. Đây là kiểm hình thức, không phải xác minh hộp thư có thật.
MAU_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")


class _CoEmail(BaseModel):
    email: str

    @field_validator("email")
    @classmethod
    def _kiem_email(cls, gia_tri: str) -> str:
        gia_tri = gia_tri.strip()
        if not MAU_EMAIL.match(gia_tri):
            raise ValueError("email không hợp lệ")
        return gia_tri


class DangKyRequest(_CoEmail):
    password: str = Field(..., min_length=DO_DAI_MAT_KHAU_TOI_THIEU)
    name: str | None = None


class DangNhapRequest(_CoEmail):
    password: str = Field(..., min_length=1)


def _nguoi_dung_theo_email(email: str) -> dict[str, Any] | None:
    with get_session_manager().session() as session:
        dong = session.execute(
            sa.text(
                "select user_id, email, name, role, password_hash from users "
                "where lower(email) = lower(:email)"
            ),
            {"email": email},
        ).first()
    if dong is None:
        return None
    return {
        "user_id": str(dong[0]),
        "email": dong[1],
        "name": dong[2],
        "role": dong[3],
        "password_hash": dong[4],
    }


def _cong_khai(nguoi_dung: dict[str, Any]) -> dict[str, Any]:
    """Bỏ `password_hash` trước khi trả ra ngoài — kể cả hash cũng không lộ."""
    return {k: v for k, v in nguoi_dung.items() if k != "password_hash"}


async def nguoi_dung_hien_tai(
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Đọc Bearer token, trả về tài khoản. 401 nếu thiếu/hỏng/hết hạn."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Thiếu token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = doc_token(authorization.split(" ", 1)[1].strip())
    if payload is None or not payload.get("sub"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token không hợp lệ hoặc đã hết hạn",
            headers={"WWW-Authenticate": "Bearer"},
        )
    nguoi_dung = await run_in_threadpool(_nguoi_dung_theo_email, str(payload["sub"]))
    if nguoi_dung is None:
        # Token còn hạn nhưng tài khoản đã bị xoá — vẫn phải chặn.
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Tài khoản không tồn tại")
    return nguoi_dung


async def yeu_cau_admin(
    nguoi_dung: dict[str, Any] = Depends(nguoi_dung_hien_tai),
) -> dict[str, Any]:
    """TC-505: không phải admin thì 403, không phải 401 — đã biết anh là ai rồi."""
    if nguoi_dung.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cần quyền admin")
    return nguoi_dung


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def dang_ky(request: DangKyRequest) -> dict[str, Any]:
    def _tao() -> dict[str, Any]:
        with get_session_manager().session() as session:
            trung = session.execute(
                sa.text("select 1 from users where lower(email) = lower(:email)"),
                {"email": request.email},
            ).first()
            if trung is not None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT, detail="Email đã được đăng ký"
                )
            chua_co_ai = session.execute(sa.text("select count(*) from users")).scalar() == 0
            user_id = uuid.uuid4()
            session.execute(
                sa.text(
                    "insert into users (user_id, email, password_hash, name, role) "
                    "values (:uid, :email, :ph, :name, :role)"
                ),
                {
                    "uid": str(user_id),
                    "email": request.email,
                    "ph": bam_mat_khau(request.password),
                    "name": request.name,
                    "role": "admin" if chua_co_ai else "user",
                },
            )
            session.commit()
            return {
                "user_id": str(user_id),
                "email": request.email,
                "name": request.name,
                "role": "admin" if chua_co_ai else "user",
            }

    return await run_in_threadpool(_tao)


@router.post("/login")
async def dang_nhap(request: DangNhapRequest) -> dict[str, Any]:
    nguoi_dung = await run_in_threadpool(_nguoi_dung_theo_email, request.email)
    # Một thông báo duy nhất cho cả "không có email" lẫn "sai mật khẩu": tách ra
    # là cho phép dò xem email nào đã đăng ký.
    if nguoi_dung is None or not kiem_mat_khau(request.password, nguoi_dung["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Email hoặc mật khẩu không đúng"
        )
    token, song_giay = tao_token(nguoi_dung["email"], nguoi_dung["role"])
    return {"access_token": token, "token_type": "bearer", "expires_in": song_giay}


@router.get("/me")
async def toi_la_ai(nguoi_dung: dict[str, Any] = Depends(nguoi_dung_hien_tai)) -> dict[str, Any]:
    return _cong_khai(nguoi_dung)
