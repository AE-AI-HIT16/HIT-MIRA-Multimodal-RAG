"""
archive_api/token_manager.py — Quản lý X-Chatbot-Token cho các API cần
xác thực (Search Archives, Get Archive Detail, Staff Archive Metadata,
File Proxy).

Token lấy từ POST /api/v1/chatbot/session-token, cache trong bộ nhớ
(process-level, dùng chung cho mọi request) để không phải xin token
mới cho MỖI lần gọi tool. Nếu 1 request bị 401 (token hết hạn/invalid)
-> tự động xin token mới rồi thử lại đúng 1 lần (xem archive_api/client.py).

LƯU Ý QUAN TRỌNG: request body cụ thể để lấy token (client_id/secret
hay cơ chế khác) và tên field chứa token trong response
(token/accessToken/...) tùy vào hợp đồng THẬT của API
`/api/v1/chatbot/session-token` mà đề bài chưa mô tả chi tiết. Code
dưới đang đoán hợp lý theo quy ước phổ biến — CẦN chỉnh lại đúng theo
tài liệu/response mẫu thật của API này trước khi chạy production.
"""
import asyncio
from typing import Optional

import httpx

from config.configs import config_object
from logger import get_logger

logger = get_logger(__name__)


class SessionTokenManager:
    def __init__(self):
        self._token: Optional[str] = None
        self._lock = asyncio.Lock()

    async def get_token(self, force_refresh: bool = False) -> str:
        if not config_object.AUTH_ENABLED:
            return ""  # API chưa yêu cầu token, tránh gọi endpoint chưa tồn tại

        if self._token and not force_refresh:
            return self._token

        async with self._lock:
            # double-checked locking: 1 request khác có thể đã refresh
            # xong token trong lúc request này chờ lock
            if self._token and not force_refresh:
                return self._token
            self._token = await self._request_new_token()
            return self._token

    async def _request_new_token(self) -> str:
        url = f"{config_object.ARCHIVE_API_BASE_URL}{config_object.CHATBOT_TOKEN_PATH}"

        body = {}
        if config_object.CHATBOT_CLIENT_ID:
            body["clientId"] = config_object.CHATBOT_CLIENT_ID
        if config_object.CHATBOT_CLIENT_SECRET:
            body["clientSecret"] = config_object.CHATBOT_CLIENT_SECRET

        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=body, timeout=config_object.HTTP_TIMEOUT_SECONDS)
            resp.raise_for_status()
            data = resp.json()

        # TODO: đổi đúng field name theo response thật của API
        token = data.get("token") or data.get("accessToken") or (data.get("data") or {}).get("token")
        if not token:
            raise RuntimeError(
                f"Không tìm thấy field token trong response của {url}: {data}. "
                f"Kiểm tra field name thật (token/accessToken/...) rồi sửa "
                f"SessionTokenManager._request_new_token()."
            )
        logger.info("Đã lấy X-Chatbot-Token mới.")
        return token


_token_manager = SessionTokenManager()


def get_token_manager() -> SessionTokenManager:
    return _token_manager
