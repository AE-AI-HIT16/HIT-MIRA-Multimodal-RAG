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
video ──► TransNetV2 ──► keyframe(3/shot)+timestamp ──► Jina-CLIP v2 ──► [Qdrant: media]
  │                                                          └─► Gemini 2.5 Flash caption ──► (payload/caption)
  └─► ffmpeg tách audio ──► Zipformer-30M-RNNT-6000h ──► transcript
                                   └─► Gemini 2.5 Flash: clean + summarize ──► chunk ──► Vietnamese_Embedding ──► [Qdrant: transcript]

nội quy ──► tách điều/khoản ──► chunk ──► Vietnamese_Embedding (nền bge-m3) ──► [Qdrant: regulation]
```
Điều phối ở `pipeline/run.py`; lỗi 1 mục → log, không chặn batch.
**Chunking (cần chốt sớm):** transcript cắt theo cửa sổ ~200–300 token, overlap ~15%, giữ timestamp đầu/cuối; nội quy cắt theo điều/khoản (1 khoản = 1 chunk, kèm số điều làm neo trích dẫn).

## 3. Luồng ONLINE (Router RAG — tầng của mình, 2 paper không có)

```
câu hỏi (text ± ảnh)
  └─► routing.intent.route()               # luật: có ảnh→media · từ khóa nội quy→regulation
        ├─ media tool      → retrieve_media (Jina-CLIP v2)        ┐
        │                    + retrieve_by_transcript (Vietnamese_Embedding) ├─ gộp theo video_id+timestamp
        └─ regulation tool → retrieve_regulations (Vietnamese_Embedding)     ┘
  └─► rank (cosine; ngưỡng "không tìm thấy")
  └─► answer: Gemini 2.5 Flash synthesis + trích dẫn nguồn / neo điều-khoản + disclaimer
             (Flash-Lite cho truy vấn đơn giản; cache câu trả lời để né quota — xem §4)
```
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
- Model v1: **TransNetV2 · Jina-CLIP v2 · Zipformer-30M-RNNT-6000h (backend sherpa-onnx) · Gemini 2.5 Flash / Flash-Lite (clean/caption/answer) · AITeamVN/Vietnamese_Embedding · Qdrant**.
- **Thứ tự bắt buộc:** dựng **bộ eval có nhãn (T-70) ~30–50 truy vấn *trước*** rồi mới benchmark — không có ground-truth thì không đo Recall@5 được. (T-70 đã kéo lên sprint 1–2, xem `tasks.md`.)
- Benchmark chốt số: **Jina-CLIP v2** (mặc định) vs fallback SigLIP 2+dịch cho ảnh; Vietnamese_Embedding cho text (đơn model, không còn phải chọn kép).
