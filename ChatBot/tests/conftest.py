"""Cấu hình chung cho test của ChatBot.

Toàn bộ code trong `src/` import theo dạng tuyệt đối (`from src.graph...`),
nên thư mục `ChatBot/` phải nằm trên `sys.path` thì test mới chạy được dù
pytest được gọi từ đâu.

Test ở đây **chạy hoàn toàn offline**: không cần MCP server, không cần API
key, không gọi LLM. Cái gì cần dịch vụ thật thì dùng đồ giả.
"""

from __future__ import annotations

import sys
from pathlib import Path

CHATBOT_ROOT = Path(__file__).resolve().parents[1]
if str(CHATBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(CHATBOT_ROOT))
