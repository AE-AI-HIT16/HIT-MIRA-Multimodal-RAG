"""
archive_api/client.py — Client gọi TRỰC TIẾP Public Archive API cho
4 nhóm API mới (search, detail, staff metadata, file proxy).

KHÁC với rag/infrastructure/archive_api_client.py: client đó chỉ phục
vụ ingestion hàng loạt cho pipeline RAG (fetch_page/download_file để
đưa vào Qdrant, không cần auth). Client này phục vụ tool MCP trả lời
LIVE cho người dùng — gọi thẳng theo field lọc chính xác, có auth qua
X-Chatbot-Token, và tự refresh token khi gặp 401.

Cố ý đặt trong mcp/ (không phải rag/) vì đây không phải logic
embedding/vector — mcp/ sở hữu tích hợp trực tiếp này để phục vụ đúng
tool của nó, giữ rag/ chỉ tập trung vào ingestion + semantic retrieval.
"""
from typing import Any, Optional

import httpx

from archive_api.token_manager import get_token_manager
from config.configs import config_object
from logger import get_logger

logger = get_logger(__name__)

# File lớn hơn mức này chỉ trả metadata + gợi ý thay vì base64 inline,
# tránh làm phình payload trả về cho LLM (tốn token, có thể vượt giới
# hạn kích thước phản hồi của tool).
MAX_INLINE_FILE_BYTES = 8 * 1024 * 1024  # 8MB


class ArchiveApiClient:
    def __init__(self):
        self._base_url = config_object.ARCHIVE_API_BASE_URL
        self._timeout = config_object.HTTP_TIMEOUT_SECONDS
        self._token_manager = get_token_manager()

    async def _request(self, method: str, path: str, params: Optional[dict] = None) -> httpx.Response:
        url = f"{self._base_url}{path}"
        token = await self._token_manager.get_token()
        headers = {"X-Chatbot-Token": token} if token else {}

        async with httpx.AsyncClient() as client:
            resp = await client.request(method, url, params=params, headers=headers, timeout=self._timeout)

            if resp.status_code == 401 and config_object.AUTH_ENABLED:
                logger.warning(f"Token hết hạn/không hợp lệ khi gọi {url}, xin token mới rồi thử lại...")
                token = await self._token_manager.get_token(force_refresh=True)
                headers = {"X-Chatbot-Token": token} if token else {}
                resp = await client.request(method, url, params=params, headers=headers, timeout=self._timeout)

            resp.raise_for_status()
            return resp

    async def search_archives(
            self,
            keyword: Optional[str] = None,
            status: Optional[str] = None,
            warehouse_id: Optional[str] = None,
            language: Optional[str] = None,
            maintenance: Optional[str] = None,
            created_from: Optional[str] = None,
            created_to: Optional[str] = None,
            updated_from: Optional[str] = None,
            updated_to: Optional[str] = None,
            page: int = 0,
            size: int = 20,
    ) -> dict[str, Any]:
        """
        Luôn trả về ĐÚNG 1 TRANG (mặc định page=0, size=20) — KHÔNG tự
        động gộp nhiều trang, để tránh dồn quá nhiều record vào 1 lần
        trả lời LLM (tốn token, dễ vượt context). Response có đủ
        totalPages/totalElements/last để bên gọi (feature_manager) biết
        còn trang tiếp theo hay không và tự quyết định có cần gọi lại
        tool với page+1 hay dừng lại vì đã tìm thấy.
        """
        params = {
            "keyword": keyword,
            "status": status,
            "warehouseId": warehouse_id,
            "language": language,
            "maintenance": maintenance,
            "createdFrom": created_from,
            "createdTo": created_to,
            "updatedFrom": updated_from,
            "updatedTo": updated_to,
            "page": page,
            "size": size,
        }
        params = {k: v for k, v in params.items() if v is not None}  # bỏ filter rỗng, không gửi lên API

        resp = await self._request("GET", config_object.ARCHIVE_SEARCH_PATH, params=params)
        return resp.json()

    async def get_archive_detail(self, archive_id: str) -> dict[str, Any]:
        path = config_object.ARCHIVE_DETAIL_PATH.format(id=archive_id)
        resp = await self._request("GET", path)
        return resp.json()

    async def get_staff_archive_metadata(self, only_metadata: bool = True) -> dict[str, Any]:
        params = {"only_metadata": str(only_metadata).lower()}
        resp = await self._request("GET", config_object.STAFF_ARCHIVE_PATH, params=params)
        return resp.json()

    async def get_file_proxy(self, key: str, file_name: str) -> tuple[bytes, str]:
        params = {"key": key, "fileName": file_name}
        resp = await self._request("GET", config_object.FILE_PROXY_PATH, params=params)
        content_type = resp.headers.get("Content-Type", "application/octet-stream")
        return resp.content, content_type


_client: Optional[ArchiveApiClient] = None


def get_archive_api_client() -> ArchiveApiClient:
    global _client
    if _client is None:
        _client = ArchiveApiClient()
    return _client
