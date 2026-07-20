import httpx
from src.config import configs
from src.clients.auth_client import get_chatbot_token

async def _get(path: str, params: dict | None = None):
    token = await get_chatbot_token()
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{configs.ARCHIVE_API_BASE_URL}{path}",
            params=params,
            headers={"X-Chatbot-Token": token},
        )
        resp.raise_for_status()
        return resp.json()

async def search_archives(
    keyword: str | None = None,
    status: str | None = None,
    warehouseId: str | None = None,
    language: str | None = None,
    maintenance: str | None = None,
    createdFrom: str | None = None,
    createdTo: str | None = None,
    updatedFrom: str | None = None,
    updatedTo: str | None = None,
    page: int = 1,
    size: int = 10,
):
    params = {k: v for k, v in locals().items() if v is not None}
    return await _get("/api/public/archives", params)

async def get_archive_detail(archive_id: str):
    return await _get(f"/api/public/archives/{archive_id}")

async def get_staff_archive_metadata(only_metadata: bool = True):
    return await _get("/api/public/staff-archive", {"only_metadata": str(only_metadata).lower()})

async def get_file_proxy_url(key: str, fileName: str):
    # File proxy thường trả file nhị phân, nên chỉ cần build URL để trả về cho người dùng
    return f"{configs.ARCHIVE_API_BASE_URL}/api/public/files/proxy?key={key}&fileName={fileName}"
