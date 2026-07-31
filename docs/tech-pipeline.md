# Đề xuất Công nghệ & Pipeline — HIT-MIRA (rút từ 2 paper AI Challenge)

> Nguồn tham chiếu:
> - **MERVIN 2025** (HCMUS, 79/88 điểm) — `public/MERVIN.pdf`. Kiến trúc gần dự án ta nhất:
>   keyframe + transcript + summary, 3 index tách biệt, Gemini làm sạch ASR.
> - **SOICT 2023** (UIT, top 1 AI Challenge 2023) — `public/3628797.3629022.pdf`.
>   Multi-model multi-stage: BEiT-3+Faiss, OCR+ASR+ES, object detection, temporal.
>
> **Khác biệt cốt lõi:** 2 paper là hệ *retrieval thuần* để thi KIS (không sinh câu trả lời).
> HIT-MIRA thêm **RAG sinh câu trả lời + router đa nguồn + hỏi-đáp nội quy**. Nên ta *mượn
> pipeline dữ liệu & truy xuất* của họ, rồi *chồng thêm tầng RAG/LLM* của mình.

## 0. Sau khi code xong: đề xuất nào đã đổi, vì sao

Phần còn lại của tài liệu này là **bản đề xuất viết trước khi code**, giữ lại vì
phần đối chiếu với 2 paper vẫn còn giá trị. Nhưng có 4 quyết định đã đổi khi chạm
dữ liệu thật. Bảng này là phần đúng; chỗ nào bên dưới mâu thuẫn thì tin bảng này
và tin code.

| Stage | Đề xuất ban đầu | **Thực tế v1** | Vì sao đổi |
|---|---|---|---|
| Nhúng text (transcript + nội quy) | AITeamVN/Vietnamese_Embedding (nền bge-m3) | **jina-clip-v2** — dùng chung với ảnh | Chỉ model đa phương thức mới nhúng được ảnh. Chọn model text mạnh nhất nghĩa là phải nuôi *hai* không gian vector và nhúng câu hỏi *hai* lần mỗi request. v1 chọn một model: một khoá, một trần rate limit, một số chiều. Đánh đổi và mốc phải xem lại: **CLAUDE.md § "One embedding model"** |
| Caption ảnh/frame | Gemini 2.5 Flash Vision | **`MEDIA_VISION_MODEL_NAME`** (rơi về `OPENROUTER_MODEL_NAME`) — một lời gọi trả **cả caption lẫn OCR** | Gộp 2 việc vào 1 lời gọi giảm nửa số request và nửa quota; đổi provider được bằng biến môi trường, không phải sửa code |
| Làm sạch transcript bằng LLM | Gemini 2.5 Flash clean + summarize | **chưa làm** | Chưa đo được là nó có đáng không. Transcript thô của Zipformer đang được nhúng thẳng — viết hoa toàn bộ, không dấu câu. Prompt của ChatBot chịu trách nhiệm viết lại cho dễ đọc lúc trích dẫn |
| Sinh câu trả lời | Gemini 2.5 Flash | **`LLM_PROVIDER:LLM_MODEL`**, gọi từ `ChatBot/` (LangGraph) qua MCP | Tầng trả lời tách hẳn khỏi `API/`, xem `docs/structure.md` |

Hai thứ **có** trong code mà bảng §1 xếp vào v2:

- **OCR**: đã trích chữ trong hình và lưu ở payload `ocr_text`. Cái *chưa* có là
  tìm kiếm từ khoá/hybrid trên nó — vẫn hoàn toàn là vector.
- **Object detection**: đã chạy, lưu ở `detected_objects` / `object_counts`. Cái
  *chưa* có là lọc kết quả theo object.

Một thứ không hề có trong kế hoạch mà thực tế phải dựng: **`runpod_worker/`** —
worker GPU chạy theo kiểu artifact-only (nhận URL presigned, trả về một file ZIP,
không hề thấy thông tin đăng nhập PostgreSQL/MinIO). Đây là câu trả lời cho ràng
buộc "không có GPU" ở §4, thay cho phương án "đẩy hết lên API free-tier".

> Lưu ý cấu hình: biến `ASR_MODEL` từng có trong `.env.example` **không được đọc ở
> đâu cả** và ghi sai model (`whisper`). Giá trị thật nằm ở
> `API/Resources/model.yaml` → `MEDIA_MODELS.ASR_MODEL_NAME`.

## 1. Bảng công nghệ theo stage

| Stage | Paper dùng gì | **Đề xuất v1** | Lý do | Map scaffold |
|---|---|---|---|---|
| Keyframe video | MERVIN: **TransNetV2**, 3 frame/shot @0.15/0.5/0.85 | **TransNetV2** (fallback: ffmpeg theo bước thời gian) | Cắt theo shot → frame đại diện tốt hơn lấy mẫu cố định | `pipeline/frames.py` |
| Nhúng ảnh/frame | MERVIN: PE-Core-bigG-14-448 · UIT: BEiT-3 | **Jina-CLIP v2** (đa ngôn ngữ 89 thứ tiếng + Matryoshka cắt dim) | Query **tiếng Việt → ảnh trực tiếp**, nhẹ hơn bigG. *SigLIP 2 mạnh về ảnh nhưng **không đa ngôn ngữ** → phải dịch query VN→EN, nên bỏ khỏi lựa chọn chính; giữ làm fallback.* | `providers/embeddings.py::ImageEmbedder` |
| ASR (audio→text) | ASR tiếng Việt | **hynt/Zipformer-30M-RNNT-6000h** chạy qua backend sherpa-onnx | ZipFormer RNNT tiếng Việt 30M tham số, nhẹ và nhanh trên CPU; transcript hiện lưu 1 segment phủ toàn audio vì backend hiện chunk timestamp thô theo cửa sổ audio | `providers/asr.py` + `pipeline/asr.py` |
| Làm sạch transcript | MERVIN: **Gemini 1.5 Flash** (clean ~8k tok + summarize 3–4k) | **Gemini 2.5 Flash** (free-tier) làm sạch + tóm tắt | Khử nhiễu ASR tiếng Việt (đúng rủi ro đã ghi trong BRD). *(Gemini 2.0 Flash đã bị khai tử 3/3/2026 — không dùng.)* | `pipeline/asr.py` → `providers/llm.py` |
| Nhúng text (transcript + nội quy) | MERVIN: dangvantuan/vietnamese-embedding (STS, không phải retrieval) | **AITeamVN/Vietnamese_Embedding** (fine-tune **từ bge-m3** trên 300k triplet query–pos–neg) | Tuned cho *retrieval* (không phải STS như dangvantuan); vì nền là bge-m3 nên **cùng lúc** retrieval-tuned + context dài + hỗ trợ hybrid → **một model dùng cho cả transcript lẫn nội quy** (bỏ được lựa chọn kép vi-embed/bge-m3) | `providers/embeddings.py::TextEmbedder` |
| Caption ảnh/frame | *(2 paper không caption — họ dùng OCR + object)* | **Gemini 2.5 Flash Vision** → caption tiếng Việt | Không tốn GPU, ra tiếng Việt, cùng provider LLM | `pipeline/caption.py` + `providers/captioner.py` |
| Vector DB | MERVIN: **Milvus** · UIT: **Faiss** | **Qdrant** (đã chọn) — 3 collection: media / transcript / nội quy | Filter+payload dễ, nhẹ, vận hành đơn giản cho capstone | `shared/vectorstore/qdrant.py` |
| Sinh câu trả lời (RAG) | *(không có — retrieval thuần)* | **Gemini 2.5 Flash** synthesis + trích dẫn + disclaimer; **Flash-Lite** cho truy vấn đơn giản | Đóng góp riêng của HIT-MIRA so với 2 paper | `domains/answer/service.py` |
| LLM provider | MERVIN: Gemini 1.5 Flash | **Gemini 2.5 Flash / Flash-Lite** (free-tier) | Miễn phí, tiếng Việt tốt, dùng lại cho clean/caption/answer. Ràng buộc quota ở §4 | `providers/llm.py` |

### Đẩy sang v2 (mượn từ paper nhưng chưa cần cho v1)
| Stage | Paper | Ghi chú |
|---|---|---|
| OCR chữ trong ảnh | UIT: DeepSolo (detect) + **PARSeq** (train tiếng Việt) + Elasticsearch fuzzy | Dùng **PARSeq/VietOCR** hoặc PaddleOCR; BR-204 |
| Hybrid keyword search | UIT: **Elasticsearch fuzzy** cho OCR/ASR | Bắt lỗi nhận dạng tiếng Việt mà vector bỏ sót → BM25 + vector |
| Object filtering | UIT: **Co-DETR + Swin-L** + lọc theo lưới MxN | PRD không yêu cầu lọc object → bỏ v1 |
| Temporal đa sự kiện | MERVIN: `S = 10·S_pair + 5·(S̄1+S̄2)` · UIT multi-stage | Áp cho tìm chuỗi 2 sự kiện; BR-308 v1 chỉ gộp video_id+timestamp |

## 2. Pipeline OFFLINE (worker — không nằm trên request, đúng NFR)

```
video ──► TransNetV2 ──► keyframe+timestamp ──► Jina-CLIP v2 ──► [Qdrant: media_clip]
  │                                    └─► vision model: caption + OCR (1 lời gọi) ──► payload
  │                                    └─► detection ──► payload (chưa dùng để lọc)
  └─► ffmpeg tách audio ──► Zipformer-30M-RNNT-6000h ──► transcript (thô, chưa clean)
                                   └─► chunk ──► Jina-CLIP v2 ──► [Qdrant: video_transcript]

nội quy ──► parse → clean → chunk ──► Jina-CLIP v2 ──► [Qdrant: rag_documents]
```
> Sơ đồ này là bản đã cập nhật theo code; các mục còn nhắc Gemini/Vietnamese_Embedding
> bên dưới là dấu vết của bản đề xuất — xem §0.
Điều phối ở `pipeline/run.py`; lỗi 1 mục → log, không chặn batch.
**Chunking (cần chốt sớm):** transcript cắt theo cửa sổ ~200–300 token, overlap ~15%, giữ timestamp đầu/cuối; nội quy cắt theo điều/khoản (1 khoản = 1 chunk, kèm số điều làm neo trích dẫn).

## 3. Luồng ONLINE (Router RAG — tầng của mình, 2 paper không có)

```
câu hỏi (tiếng Việt)
  └─► supervisor (LangGraph) tự chọn tool   # "router" nằm trong prompt, không phải bộ luật
        ├─ search_media       → nhúng câu hỏi MỘT lần (Jina-CLIP v2)
        │                       → tìm song song media_clip + video_transcript (cùng không gian)
        └─ search_regulations → rag_documents (cùng model, cùng không gian)
  └─► rank (cosine) + gộp theo video_id/timestamp + dựng chuỗi trích dẫn [1] [2] [3]
  └─► LLM tổng hợp + trích dẫn nguồn + disclaimer (chỉ với câu trả lời nội quy)
```
> Khác bản đề xuất ở hai chỗ: **không có** `routing/intent.py` (LLM chọn tool), và
> **chỉ nhúng câu hỏi một lần** cho cả hai nhánh media vì dùng chung một model.
> Gộp keyframe↔transcript theo `video_id`+timestamp chính là cách MERVIN kết hợp 2 nhánh —
> validate cho thiết kế BR-308 của ta.

## 4. Điều chỉnh cho ràng buộc capstone (GPU/ngân sách hạn chế)
- **Đẩy phần nặng lên API free-tier**: caption + làm sạch transcript + sinh câu trả lời → **Gemini 2.5 Flash** (không cần GPU). MERVIN chạy trên RTX 3060 12GB; ta né bằng cách này.
- **Chạy trên GPU server**: nhúng ảnh (Jina-CLIP v2 ~ViT-L), **hynt/Zipformer-30M-RNNT-6000h** (backend sherpa-onnx). Batch offline, không nằm trên request.
- **Không chọn PE-Core-bigG/BEiT-3** cho v1: mạnh nhưng nặng & English-centric; **multilingual CLIP (Jina-CLIP v2)** cho query tiếng Việt trực tiếp, đơn giản hơn (khỏi dịch query).
  - *Nếu chất lượng text→ảnh chưa đạt Recall@5 ≥ 0.80*: fallback = **SigLIP 2 / PE-Core** cho ảnh + **dịch query VN→EN** rồi mới nhúng.
- **Trần quota Gemini free-tier (2026)**: 2.5 Flash ~**15 req/phút · 1.500 req/ngày · 250k token/phút**. Đường answer online gọi Gemini mỗi request → dễ đụng trần lúc demo đông. Giảm thiểu:
  - Cache câu trả lời + summary/caption **tính sẵn offline**; dùng **Flash-Lite** cho truy vấn đơn giản; chuẩn bị API key dự phòng.
  - Đối chiếu NFR "trả lời trung bình ≤5s" — đo latency thực (T-72) sớm, không để tới sprint cuối mới biết vỡ.
- **Cảnh báo dữ liệu**: free-tier Google AI Studio **dùng prompt/response để train**. Nội quy CLB không nhạy cảm nên chấp nhận được; nếu về sau xử lý dữ liệu thành viên → chuyển tier trả phí hoặc self-host LLM.

## 5. Rủi ro & phương án thay thế
| Rủi ro | Giảm thiểu |
|---|---|
| ASR tiếng Việt nhiễu (tạp âm sự kiện) | **hynt/Zipformer-30M-RNNT-6000h** + Gemini clean (MERVIN); transcript chỉ **bổ trợ** cạnh keyframe, không phải nguồn duy nhất |
| Multilingual CLIP yếu với tiếng Việt | Fallback SigLIP 2/PE-Core + dịch query; hoặc thêm caption (Gemini) để search bằng text VN |
| Rerank tiếng Việt kém (MERVIN đo VN reranker chỉ 0.15) | Dùng điểm cosine, **chưa cần** cross-encoder reranker ở v1 |
| Free-tier Gemini giới hạn quota (15 rpm/1.5k ngày) | Cache caption/summary offline; batch; Flash-Lite cho truy vấn đơn giản; key dự phòng |
| Chunk transcript/nội quy chưa định nghĩa → hụt Recall | Chốt chiến lược chunking ở §2; **Vietnamese_Embedding (nền bge-m3)** đã lo context dài |
| Cold-start: chưa có dữ liệu thật | Chốt sớm nguồn video/ảnh/nội quy từ fanpage + người chuẩn hóa; không có input thì pipeline chạy rỗng |

## 6. Chốt cho `/write-hld`
- Model v1 **thực tế đang chạy**: **TransNetV2 · Jina-CLIP v2 (ảnh + transcript + nội quy + câu hỏi) ·
  Zipformer-30M-RNNT-6000h (backend sherpa-onnx) · vision model qua `MEDIA_VISION_MODEL_NAME` (caption+OCR) ·
  `LLM_PROVIDER:LLM_MODEL` (sinh câu trả lời) · Qdrant**. Dòng gạch bỏ của bản đề xuất: Gemini 2.5 Flash,
  AITeamVN/Vietnamese_Embedding — xem §0.
- **Thứ tự bắt buộc:** dựng **bộ eval có nhãn (T-70) ~30–50 truy vấn *trước*** rồi mới benchmark — không có ground-truth thì không đo Recall@5 được. (T-70 đã kéo lên sprint 1–2, xem `tasks.md`.)
- Benchmark chốt số: **Jina-CLIP v2** (mặc định) vs fallback SigLIP 2+dịch cho ảnh; Vietnamese_Embedding cho text (đơn model, không còn phải chọn kép).
