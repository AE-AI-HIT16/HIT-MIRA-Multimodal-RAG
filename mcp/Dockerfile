# MCP server (streamable-http, cổng 8091). Chatbot gọi tool qua đây, tool gọi
# ngược lại FastAPI.
#
# Build context là ./mcp nhưng code phải nằm ở /app/mcp: `config/configs.py`
# tính PROJECT_ROOT = parents[3] rồi resolve mọi đường dẫn tương đối từ đó
# (MCP_CONFIG_PATH=./mcp/Resources/dev.yaml, TOOLS_PATH trong chính file YAML).
# Copy thẳng vào /app là mọi đường dẫn lệch một cấp và server chết lúc đọc cấu
# hình chứ không phải lúc chạy.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/mcp/src \
    MCP_CONFIG_PATH=./mcp/Resources/dev.yaml \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8091 \
    MCP_TRANSPORT=streamable-http \
    MCP_SERVER_NAME=internal_rules_rag \
    MCP_TIMEOUT_SECONDS=60

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app/mcp

EXPOSE 8091
CMD ["python", "/app/mcp/src/server.py"]
