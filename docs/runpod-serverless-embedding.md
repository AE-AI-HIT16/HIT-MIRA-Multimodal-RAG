# RunPod Serverless cho jina-clip-v2

Tài liệu này chốt bước hạ tầng đầu tiên: đóng gói model nhúng đa phương thức
thành worker GPU ngoại tuyến. Endpoint đã được deploy và smoke test ngày
2026-08-06; bước này chưa thay đổi PostgreSQL/Qdrant.

## Phạm vi

- Dùng cho benchmark Jina và nhúng ảnh/text hàng loạt khi index.
- Không dùng trực tiếp cho query online khi `active workers = 0` vì cold start.
- Dùng lại `embedding_server/encoder.py`; không có bản tiền xử lý thứ hai.
- Không cần MinIO/PostgreSQL credential. Input là text hoặc ảnh base64, output
  chỉ là vector.

## Contract

Request gửi tới RunPod `/run` hoặc `/runsync`:

```json
{
  "input": {
    "model": "jina-clip-v2",
    "input": [
      {"text": "sự kiện HIT Open Day"}
    ],
    "task": "retrieval.query",
    "embedding_type": "float",
    "dimensions": 1024,
    "normalized": true
  }
}
```

Ảnh thay mỗi phần tử bằng `{"image": "<base64>"}`. Không trộn text và ảnh
trong cùng job. Worker trả giao thức Jina bên trong `output` của RunPod:

```json
{
  "status": "COMPLETED",
  "output": {
    "object": "list",
    "data": [
      {"object": "embedding", "index": 0, "embedding": [0.01]}
    ],
    "usage": {"total_tokens": 0, "prompt_tokens": 0, "items": 1}
  }
}
```

Vector thật có 1.024 phần tử. Worker chặn ngay các trường hợp: request không
hợp lệ, batch trên 64, dimensions khác 1.024, trộn hai modality, model trả
thiếu vector hoặc sai số chiều. Exception không bị nuốt để RunPod đánh dấu job
`FAILED`.

## Build image

```bash
cd /home/ubuntu/HIT-MIRA-Multimodal-RAG/embedding_server

docker build -f Dockerfile.serverless \
  -t <dockerhub-user>/hit-mira-embed-serverless:v0.1.3 .

docker push <dockerhub-user>/hit-mira-embed-serverless:v0.1.3
```

Image dùng CUDA 12.9 và PyTorch cu129 để không lỗi kiến trúc `sm_120` nếu
RunPod cấp GPU Blackwell. `transformers` được ghim đúng `4.46.2` theo config
của model: remote code còn import `clip_loss` đã bị gỡ trong transformers 5,
còn bản 4.57.x cảnh báo tokenizer regex sai và có nguy cơ làm lệch vector.

`jina-clip-v2` còn dùng `auto_map` để gọi code từ
`jinaai/jina-clip-implementation`; text tower tiếp tục đọc config/tokenizer
`jina-embeddings-v3` và code `xlm-roberta-flash-implementation`. RunPod chỉ
cache được một model cho mỗi endpoint, nên Docker image bake ba dependency
code/config/tokenizer nhỏ này theo commit cố định và bật Hugging Face offline mode.
Weights lớn của `jina-clip-v2` vẫn nằm ngoài image, trong Cached model.

## Cấu hình endpoint sau khi image đã được push

| Trường RunPod | Giá trị |
|---|---|
| Source | Docker image vừa push |
| Compute type | GPU |
| GPU configuration | 24 GB |
| Max workers | 1 |
| Active workers | 0 |
| GPU count | 1 |
| Idle timeout | 5 giây ở benchmark; có thể tăng khi index liên tiếp |
| Execution timeout | bật, 600 giây |
| Cached model | `jinaai/jina-clip-v2` |
| Network volume | không cần |
| Auto scaling | Queue delay |

RunPod hiện khuyến nghị Cached models cho model Hugging Face. Cache của custom
worker nằm tại `/runpod-volume/huggingface-cache/hub/`; encoder tự tìm snapshot
theo `refs/main`, sau đó mới fallback sang snapshot có sẵn đầu tiên. Container
không tìm thấy cache sẽ fail trước khi nhận job.

Nguồn tham khảo: [RunPod handler functions](https://docs.runpod.io/serverless/workers/handler-functions),
[Cached models](https://docs.runpod.io/serverless/endpoints/model-caching).

## Cổng kiểm chứng trước bước deploy

```bash
cd /home/ubuntu/HIT-MIRA-Multimodal-RAG/API
PY=/home/ubuntu/miniconda3/envs/nhhoang/bin/python

$PY -m pytest -q \
  tests/test_embedding_server_contract.py \
  tests/test_embedding_serverless_handler.py

$PY -m ruff check \
  tests/test_embedding_server_contract.py \
  tests/test_embedding_serverless_handler.py \
  ../embedding_server/encoder.py \
  ../embedding_server/server.py \
  ../embedding_server/runpod_handler.py
```

Sau khi deploy, cổng tiếp theo là một job text nhỏ, kiểm `status=COMPLETED`,
`len(output.data)=1`, `len(embedding)=1024`, norm xấp xỉ 1; rồi mới chạy
benchmark Jina đầy đủ.

## Deployment đã kiểm chứng

- Endpoint: `hit-mira-jina-clip-v2` (`z1zyigyd4vl2fy`).
- Template: `kzlz8sjukm`.
- Image: `hoangtechcs/hit-mira-embed-serverless:v0.1.3`.
- Digest: `sha256:0535c247e4f9a6d6119bdf15888db54877329cb02e0ef0ff11d3e87dc574f9ed`.
- Smoke test text tiếng Việt: `COMPLETED`, 1 vector, 1.024 chiều,
  norm L2 = 1,0.
- Cold start đầu tiên: khoảng 190 giây; thời gian thực thi job: 1,281 giây.
- Sau smoke test: 0 worker unhealthy. Endpoint giữ `workersMin=0`,
  `workersMax=1` để batch offline có thể scale về 0.

Không điền endpoint queue này trực tiếp vào
`MEDIA_IMAGE_EMBEDDING_BASE_URL`: client hiện tại chờ giao thức Jina trực tiếp
tại `/v1/embeddings`, còn RunPod bọc kết quả trong job `/run`/`/runsync`.

## Adapter hàng đợi

`API/src/rag_video_anh/embedding/runpod_transport.py` nối hai giao thức đó.
Nó nằm ở tầng vận chuyển (`http_post=` của `ImageEmbeddingService`) chứ không
phải một client thứ hai: chia lô, thu nhỏ ảnh 512px, đòi đúng số vector, phân
biệt lỗi tạm thời với lỗi vĩnh viễn — không việc nào trong đó liên quan tới cách
kết quả được chuyển về, nên không việc nào bị viết lại lần hai.

```python
from src.rag_video_anh.embedding.runpod_transport import build_runpod_embedding_service

service = build_runpod_embedding_service()   # đọc RUNPOD_API_KEY + JINA_RUNPOD_ENDPOINT_ID
vectors = service.embed_texts(["câu hỏi tiếng Việt"])
```

Ba điểm dễ sai:

- **Khoá là `RUNPOD_API_KEY`, không phải `JINA_API_KEY`.** Hai biến thuộc hai
  giao thức khác nhau; `JINA_API_KEY` dành cho Pod nói `/v1/embeddings` trực tiếp.
  Adapter không giữ khoá riêng — nó chuyển tiếp header do client truyền xuống.
- **Endpoint là `JINA_RUNPOD_ENDPOINT_ID`.** `RUNPOD_ENDPOINT_ID` là worker
  video; nhầm thì job vào đúng hàng đợi sai. Cấu hình thiếu bị chặn ngay với
  thông báo nói rõ biến nào.
- **Job `FAILED` không được thử lại**, `TIMED_OUT` thì có. RunPod đã tự chạy lại
  job khi worker chết, nên `FAILED` còn lại hầu hết là sai hợp đồng và thử lại
  chỉ đốt giây GPU; còn `TIMED_OUT` thường do lần chạy đầu gánh cold start, lượt
  sau thì không. Quá hạn chờ ở phía client cũng không gửi lại: job có thể vẫn
  đang chạy, gửi lại là xếp thêm một job trùng.

## Chọn đường nhúng

`API/src/rag_video_anh/embedding/provider.py` là chỗ duy nhất quyết định đi
đường nào; ba điểm dựng client — [indexing_service.py](../API/src/rag_video_anh/retrieval/indexing_service.py),
[retriever.py](../API/src/rag_video_anh/retrieval/retriever.py),
[retrieval_service.py](../API/src/rag_video_anh/retrieval/retrieval_service.py)
— đều gọi `build_media_embedder()`. Trước đó mỗi chỗ tự dựng, tức ba cơ hội để
chúng lệch nhau.

| Cấu hình | Đường đi |
|---|---|
| Có `MEDIA_IMAGE_EMBEDDING_BASE_URL` | Giao thức Jina trực tiếp |
| Không có, có `JINA_RUNPOD_ENDPOINT_ID` | Hàng đợi RunPod Serverless |
| Có cả hai | Trực tiếp — không cold start, và giống hệt hành vi trước khi có adapter |
| Không có gì | Trực tiếp, để lỗi vẫn là thông báo cũ (`Set JINA_API_KEY`) chứ không phải một lỗi RunPod mà người vận hành chưa định dùng |
| `MEDIA_EMBEDDING_PROVIDER=runpod\|jina` | Ép, thắng mọi luật trên |

Luật dò đọc biến môi trường **trước** YAML, đúng thứ tự `ImageEmbeddingService`
dùng. `AppConfig` khai giá trị này làm default của dataclass nên nó bị chốt ngay
lúc import; đọc mỗi YAML thì có lúc dò một đằng mà client dựng một nẻo.

Đường truy vấn online (`for_online_queries=True`) không bị chặn khi phải đi qua
hàng đợi — chặn thì demo không còn gì để chạy — nhưng nó **ghi WARNING** và rút
hạn chờ job từ 900s xuống 240s: người đang chờ khung chat không nên bị treo bằng
hạn của batch.

Khi index liên tiếp, **tăng idle timeout** của endpoint: adapter gửi job tuần tự
và khoảng nghỉ giữa hai job chỉ dưới một giây, nhưng idle timeout 5 giây không
còn dư địa nếu một lô ảnh mất nhiều thời gian chuẩn bị ở phía client.

Kiểm chứng: `API/tests/test_runpod_embedding_transport.py` (21 test, HTTP và
đồng hồ đều là fake, không job nào được gửi đi trong lúc chạy test).

## Đã dùng thật

Benchmark chữ đầu tiên chạy qua adapter này ngày 06/08/2026 —
[text-embedding-jina-vs-azure.md](text-embedding-jina-vs-azure.md). 67 job, 505
document, không job nào lỗi. Jina-CLIP v2 thắng Azure `text-embedding-3-small`
trên cùng tập nhãn ở cả Recall@5, MRR@10 và nDCG@10.
