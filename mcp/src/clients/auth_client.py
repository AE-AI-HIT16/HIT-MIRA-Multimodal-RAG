import time
import httpx
from src.config import configs
from src.logger import logger

_token_cache = {"token": None, "expires_at": 0}

async def get_chatbot_token() -> str:
    now = time.time()
    if _token_cache["token"] and _token_cache["expires_at"] > now + 30:
        return _token_cache["token"]

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            configs.CHATBOT_TOKEN_URL,
            json={
                "clientId": configs.CHATBOT_CLIENT_ID,
                "clientSecret": configs.CHATBOT_CLIENT_SECRET,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    _token_cache["token"] = data["token"]
    _token_cache["expires_at"] = now + data.get("expiresIn", 3600)
    logger.info("Đã lấy X-Chatbot-Token mới")
    return _token_cache["token"]