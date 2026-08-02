-- T-51 / US-505.1 — bảng tài khoản đăng nhập.
--
-- Chạy: psql "$DATABASE_URL" -f migrations/20260802_users.sql
-- An toàn khi chạy lại (IF NOT EXISTS).

BEGIN;

CREATE TABLE IF NOT EXISTS users (
    user_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         VARCHAR(255) NOT NULL UNIQUE,
    -- Chỉ lưu chuỗi đã băm dạng scrypt$n$r$p$salt$hash. Đặt tên có hậu tố
    -- `_hash` để không ai lỡ ghi mật khẩu thô vào đây.
    password_hash TEXT NOT NULL,
    name          VARCHAR(255),
    role          VARCHAR(32) NOT NULL DEFAULT 'user',
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tra cứu lúc đăng nhập luôn theo email; UNIQUE đã tạo index nên không thêm.

COMMIT;
