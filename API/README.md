# HIT-MIRA API

FastAPI service cho hệ thống HIT-MIRA Multimodal RAG, cung cấp các API xác thực,
ingest, truy hồi nội quy, truy hồi media và quản trị tác vụ nền.

## Chạy cục bộ

Từ thư mục `API/`:

```bash
python -m pip install -e .
python -m uvicorn src.server:app --reload --port 8000
```

Chạy kiểm tra:

```bash
ruff check src tests
pytest -q
```
