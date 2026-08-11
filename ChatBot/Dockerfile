# LangGraph dev server (cổng 2024) — nơi tổng hợp câu trả lời.
#
# Vẫn là `langgraph dev` chứ không phải `langgraph build`: bản build đóng gói
# cho LangGraph Platform cần LANGSMITH_API_KEY và một Postgres riêng cho
# checkpoint. Đồ án chỉ cần server in-memory, và đây đúng là thứ `make serve`
# đang chạy ở máy thật, nên container không lệch khỏi môi trường dev.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/ChatBot/.venv \
    PATH=/app/ChatBot/.venv/bin:$PATH

RUN pip install --no-cache-dir uv

WORKDIR /app/ChatBot

# Cài dependency trước khi copy code: sửa src/ thì không phải resolve lại lock.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . /app/ChatBot
RUN uv sync --frozen

# `langgraph.json` khai env: "../.env" → /app/.env. File .env thật bị
# .dockerignore loại (bí mật không nướng vào ảnh), giá trị đến từ compose. Nhưng
# CLI vẫn mở đường dẫn đó, nên phải có một file rỗng ở đúng chỗ.
RUN touch /app/.env

EXPOSE 2024
# --host 0.0.0.0: mặc định CLI chỉ bind 127.0.0.1, tức là cổng publish ra ngoài
# container không ai gọi được. --no-browser vì không có trình duyệt trong ảnh.
CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "2024", "--no-browser"]
