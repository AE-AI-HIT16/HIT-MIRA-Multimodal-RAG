# embedding_server — tự host jina-clip-v2

Thay `api.jina.ai` bằng model chạy trên máy mình, **giữ nguyên model và nguyên
giao thức**. Sinh ra sau khi tài khoản Jina hết số dư và kéo sập cả ba đường:
`/api/media/search`, `/api/retrieval/search`, và mọi script index.

## Vì sao vẫn là jina-clip-v2, không đổi model khác

Đổi model là **vứt toàn bộ 2.409 điểm** đang có trong Qdrant. Hai model là hai
không gian vector; trộn vào một collection thì điểm xếp hạng mất hết ý nghĩa
(xem `CLAUDE.md`, mục "One embedding model"). Giữ nguyên model thì index cũ còn
nguyên giá trị, GPU chỉ phải làm nốt 38 video chưa index.

Ràng buộc ép chọn model này cũng không đổi: **chỉ model đa phương thức mới nhúng
nổi ảnh**, mà cả hệ thống dùng chung một vector cho ảnh lẫn text. Có GPU riêng
không làm điều đó khác đi.

## Đã đo: vector tự host trùng vector của Jina

`scripts/check_embedding_parity.py` lấy điểm thật trong Qdrant, tải đúng ảnh đó
từ MinIO, tiền xử lý bằng chính `ImageEmbeddingService._image_as_base64`, nhúng
lại rồi so cosine. Đo ngày 02/08/2026:

| Nơi chạy | Collection | n | Thấp nhất | Trung bình |
|---|---|---|---|---|
| **GPU RTX A5000, fp16** | `media_clip` | 6 | **0.999801** | 0.999898 |
| **GPU RTX A5000, fp16** | `video_transcript` | 6 | **0.999225** | 0.999790 |
| CPU, fp32 | `media_clip` | 4 | 0.999841 | 0.999904 |
| CPU, fp32 | `video_transcript` | 4 | 0.998703 | 0.999558 |

**Kết luận: cùng không gian vector, index cũ dùng tiếp được.** Hai model khác
nhau sẽ cho cosine quanh 0 ở không gian 1024 chiều, chứ không phải 0.99.

Đáng chú ý: bản **GPU fp16 sát hơn** bản CPU fp32, dù fp16 kém chính xác hơn.
Nhiều khả năng vì api.jina.ai cũng chạy fp16 trên GPU, nên giống nhau cả ở chỗ
làm tròn.

Chạy lại bất cứ lúc nào, kể cả khi Jina đang chết (nó chỉ đọc Qdrant + MinIO):

```bash
python scripts/check_embedding_parity.py                       # model trong tiến trình
python scripts/check_embedding_parity.py --url https://…/v1/embeddings   # endpoint đã deploy
```

## Ba tham số không được sai

Sai một trong ba thì vector rơi ra ngoài không gian cũ, mà **không có lỗi nào
được ném ra** — chỉ có kết quả tìm kiếm ngày càng vô lý:

| Tham số | Giá trị | Vì sao |
|---|---|---|
| `task` | `retrieval.query` | Chọn LoRA adapter. jina-clip-v2 CHỈ nhận giá trị này; `retrieval.passage` là của jina-embeddings-v3. |
| `truncate_dim` | `1024` | Model này là Matryoshka, cắt được xuống tận 64 chiều. |
| chuẩn hoá | L2 | Client cũ luôn gửi `"normalized": true`. |

Ảnh được thu nhỏ về 512px **phía client** trước khi base64, nên server không
đụng vào kích thước.

## Một endpoint, hai định dạng client

Cùng một pod phục vụ cả ba đường đang chết, nhưng hai nhánh **không gửi cùng
định dạng** — server nhận cả hai:

| Nhánh | Client | `input` |
|---|---|---|
| media (ảnh, keyframe, lời thoại, câu hỏi) | `ImageEmbeddingService` | kiểu Jina: `[{"image": b64}]` / `[{"text": s}]` |
| nội quy | LangChain `OpenAIEmbeddings` | kiểu OpenAI: `["chuỗi", "chuỗi"]` |

Chỉ nhận một dạng thì nhánh kia ăn 422, mà lỗi hiện ra tận trong LangChain nên
rất khó lần ngược về đây.

## Deploy lên RunPod

Cách nhanh — một lệnh, không cần container registry:

```bash
python scripts/deploy_embedding_pod.py            # xem kế hoạch + giá
python scripts/deploy_embedding_pod.py --apply    # tạo pod, in sẵn cấu hình cần đặt
python scripts/deploy_embedding_pod.py --status
python scripts/deploy_embedding_pod.py --terminate <POD_ID>
```

Script dùng ảnh PyTorch công khai và đẩy code qua MinIO presigned URL. Cho môi
trường thật thì `Dockerfile` ở đây sạch hơn (nướng sẵn trọng số, không phụ
thuộc MinIO lúc khởi động), nhưng cần credential registry:

```bash
docker build -t <registry>/hit-mira-embed:1.0 embedding_server/
docker push  <registry>/hit-mira-embed:1.0
```

Dựng **pod thường trực**, không phải serverless: serverless cold start 30–60
giây, mà `/api/media/search` cần nhúng câu hỏi ngay lúc người dùng gõ.

Rồi trỏ client sang, **không phải sửa code**:

```bash
# nhánh media
MEDIA_IMAGE_EMBEDDING_BASE_URL=https://<pod>-8100.proxy.runpod.net/v1/embeddings
JINA_API_KEY=<đúng EMBED_SERVER_API_KEY của pod>   # đây là Bearer token client gửi
# nhánh nội quy — LangChain tự nối "/embeddings" vào sau
EMBEDDING_BASE_URL=https://<pod>-8100.proxy.runpod.net/v1
EMBEDDING_API_KEY=<đúng EMBED_SERVER_API_KEY của pod>
```

(hoặc sửa `MEDIA_MODELS.CLIP_API_BASE_URL` trong `API/Resources/model.yaml`)

**Tiền:** pod tính theo giờ kể cả lúc không dùng. RTX A5000 đo thật là
**$0,27/giờ** (không phải $0,16 như bảng giá cộng đồng) → **~$195/tháng** nếu
để thường trực. Index một lần rồi tắt thì chỉ vài xu, và vector nằm lại trong
Qdrant vĩnh viễn. Nhớ `--terminate`.

Khi đã tự host thì **bỏ luôn bộ giữ nhịp token** — nó sinh ra để né hạn mức của
Jina, giờ chỉ còn làm chậm:

```bash
MEDIA_EMBEDDING_TOKENS_PER_MINUTE=0   # <= 0 nghĩa là tắt hẳn
```

## Chạy trên máy CPU — CẢNH BÁO

Máy API hiện tại (7,6GB RAM, 2 nhân) **không đủ chỗ**. Số đo thật:

- nạp xong còn chiếm **3,68GB thường trú**, đỉnh **5,05GB**
- lúc đó chỉ còn 1,68GB rảnh, và forward pass đầu tiên bị OOM giết (exit 137)
- `bfloat16` không cứu được: đỉnh vẫn 4,89GB, vì bản fp32 mmap và bản đã ép kiểu
  cùng nằm trong bộ nhớ lúc chuyển đổi
- phép đo parity ở trên chỉ chạy xong nhờ **tạm thêm 8GB swap**, đã gỡ sau khi đo

Nên nhúng câu hỏi cũng phải đi qua pod RunPod. Muốn chạy CPU cục bộ thì phải
thêm RAM hoặc thêm swap thường trực, và chấp nhận vài giây mỗi câu hỏi.

## Vì sao có venv riêng

`transformers` 5.x làm vỡ remote code của jina-clip-v2 theo hai nấc:

1. `configuration_clip.py` gọi `hasattr(torch, torch_dtype)`, mà transformers 5
   đã đổi `torch_dtype` thành đối tượng `torch.dtype`
   → `TypeError: attribute name must be string`.
2. `modeling_clip.py` import `clip_loss`, thứ đã bị gỡ khỏi
   `transformers.models.clip.modeling_clip`
   → `ImportError`.

Nấc 1 lách được bằng cách truyền tên kiểu dưới dạng **chuỗi** (đã làm trong
`encoder.py`). Nấc 2 thì không. Vì vậy `requirements.txt` ghim `transformers<5`,
và trên máy dev thì dùng venv riêng để không phải hạ cấp cả môi trường chính:

```bash
python -m venv --system-site-packages /home/ubuntu/.venvs/embedding_server
/home/ubuntu/.venvs/embedding_server/bin/pip install "transformers>=4.45,<5" accelerate timm
```

Gỡ ghim được khi Jina cập nhật remote code cho transformers 5.

## Tệp

| Tệp | Việc |
|---|---|
| `encoder.py` | Nạp model, sinh vector. Dùng chung cho CPU và GPU để hai nơi không thể lệch tiền xử lý. |
| `server.py` | FastAPI, `POST /v1/embeddings` đúng giao thức Jina + `GET /health`. |
| `Dockerfile` | Ảnh GPU, nướng sẵn trọng số vào trong. |
