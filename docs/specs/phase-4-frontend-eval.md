# Spec P4 — Frontend · Eval · Demo

> Phase 4 · Epic E7 + E8 · Sprint 5 (riêng P4-4 eval set: sprint 1–2) · Owner: AIE-2/shared (web) + AIE-1 (eval)
> Mục tiêu: giao diện chat đa phương thức + số đo chất lượng khách quan + demo 4 luồng lõi.
> Exit gate: web gửi/nhận + render ảnh/clip/link inline · báo cáo Recall@k/MRR/latency/routing · `TC-605` 4 luồng pass.

## Phạm vi

| ID | Chức năng | Task | Sprint | Ghi chú |
|---|---|---|---|---|
| P4-1 | Web chat (Next.js) | T-60 | 5 | 🔴 đường găng |
| P4-2 | Render đa phương thức | T-61 | 5 | |
| P4-3 | Màn admin | T-62 | 5 | |
| P4-4 | Bộ eval có nhãn | T-70 | **1–2** | làm sớm — cần cho benchmark model |
| P4-5 | Recall@k + MRR | T-71 | 5 | |
| P4-6 | Latency avg/p95 | T-72 | 5 | |
| P4-7 | Eval nội quy + routing | T-73 | 5 | |
| P4-8 | Demo checklist | T-74 | 5 | 🔴 đích dự án |

---

## P4-1 · Web chat `[T-60]` 🔴 đường găng

**Thư mục:** `web/` (Next.js App Router + TypeScript + Tailwind) · Phụ thuộc: P3-4

**Màn hình v1 (tối thiểu 2):**
1. `/` — chat: khung hội thoại, input text + nút đính kèm ảnh, chọn override nguồn (auto/media/nội quy), danh sách bong bóng.
2. `/login` — form đăng nhập (JWT lưu localStorage; guest vẫn chat được).

**Gọi API:**
```ts
POST {API}/chat/message  {text?, image_path?, override?, conversation_id?}
// upload ảnh: POST /ingest/query-image (multipart) → path tạm → gắn vào message
// (endpoint upload ảnh truy vấn thuộc P3-4 mở rộng — KHÔNG đi qua consent gate vì không nạp kho)
```

**Trạng thái UI:**
- Đang chờ → indicator "đang tìm…"; timeout/lỗi mạng → bong bóng lỗi + nút **Gửi lại**, giữ nguyên nội dung đã gõ (US-501.1 AC-2).
- Tin rỗng → disable gửi; file >10MB hoặc sai định dạng → chặn client-side + thông báo (US-502.1 AC-2).
- Responsive: mobile 1 cột, desktop max-width; chạy Chrome/Firefox/Safari (US-505.1).

**DoD / Test `TC-501/502/505`:** gửi text nhận trả lời trong luồng; đính ảnh → kết quả truy vấn ảnh; mở mobile viewport thao tác được.

---

## P4-2 · Render kết quả đa phương thức `[T-61]`

**Thư mục:** `web/` · Phụ thuộc: P4-1, P3-6

**Card kết quả (map từ `items` của ChatReply):**

| Loại item | Nhận diện | Render |
|---|---|---|
| Ảnh | `frame_path`/`media_type=image` | `<img src=/media/{id}>` + caption; không caption → nhãn "chưa có mô tả" (US-402.1 AC-2) |
| Video khớp frame | có `timestamp` | player clip `GET /media/{id}/clip?ts=` + nút "xem trong video" → `/stream#t=` (US-403/404) |
| Video khớp transcript | có `moments` | đoạn text khớp + chips "nhảy tới 02:13" cho từng moment |
| Nội quy | có `article/dieu` | blockquote "Điều X Khoản Y" + text; disclaimer hiển thị cuối câu trả lời |
| Nguồn | `source_url` | link bài gốc; rỗng → nhãn "nguồn nội bộ" KHÔNG link gãy (US-405.1 AC-2) |

**Edge:** media tải lỗi → placeholder + nút thử lại, không vỡ layout (US-503.1 AC-2); >6 kết quả → cuộn ngang/xem thêm.

**DoD / Test `TC-503`:** kết quả gồm ảnh + clip → cả hai render và phát inline.

---

## P4-3 · Màn admin `[T-62]`

**Thư mục:** `web/app/admin/` · Phụ thuộc: P1-3, P4-5 · Guard: role admin (P3-5)

**Chức năng:** ① form upload media + metadata → `/ingest/upload` (hiện created/skipped + lý do) · ② upload nội quy → `/ingest/regulations` · ③ nút chạy pipeline cho asset chưa index + xem summary `{n_frames, n_indexed, n_captioned, n_transcript}` · ④ bảng kết quả eval các lần chạy (`eval_runs`).

**DoD:** thao tác nạp/chạy pipeline/xem eval qua UI được.

---

## P4-4 · Bộ eval có nhãn `[T-70]` — LÀM SỚM (sprint 1–2)

**Vị trí:** `data/eval/eval_queries.jsonl` + bảng `eval_queries` · Lý do sớm: cần cho benchmark chọn/xác nhận model (BR-601/602) trước khi index toàn kho.

**Format mỗi dòng:**
```json
{"id": 1, "category": "text2media|image2event|regulation|transcript|out_of_domain",
 "query_text": "...", "query_image": null,
 "expected_media_ids": [3, 7], "expected_rule_chunks": [{"article": 5, "clause": 2}],
 "expected_source": "media|regulation|both"}
```

**Cơ cấu (~40 truy vấn):** 15 text→media · 5 ảnh→sự kiện · 10 nội quy (tra cứu + tình huống) · 5 transcript ("ai nói gì trong video…") · 5 ngoài miền (đo không-bịa).

**Quy trình gán nhãn:** 2 người gán độc lập trên kho mẫu đã nạp → bất đồng → thảo luận chốt; nhãn là **tập media/điều-khoản đúng**, không phải thứ hạng.

**DoD / Test `TC-601`:** file load được, ≥30 truy vấn đủ trường, mỗi category ≥5.

---

## P4-5 · Recall@k + MRR `[T-71]`

**File:** `app/domains/eval/service.py` · Phụ thuộc: P2-1, P4-4

**Chữ ký:**
```python
def recall_at_k(retrieved_ids: list, expected_ids: list, k: int) -> float
    # |top-k ∩ expected| / |expected|; expected rỗng → skip query (không chia 0)
def mrr(list_of_runs: list[tuple[retrieved_ids, expected_ids]]) -> float
    # mean(1/rank vị trí ĐẦU TIÊN xuất hiện expected; không có → 0)
def run_eval(queries, *, retriever, k=5) -> EvalReport   # chạy cả bộ, ghi eval_runs
```

**DoD / Test `TC-602` `test_recall_and_mrr_match_by_hand`:** bộ nhỏ 3–5 query tính tay khớp từng số (đây là skip cuối trong `test_learning_todo.py`).

---

## P4-6 · Latency `[T-72]`

**File:** cùng service · Phụ thuộc: P3-4

**Đo:** wrap `handle_message` với bộ eval → thu `latency_ms` từng lượt → `avg`, `p95` (numpy percentile hoặc sort thủ công). Tách 2 chế độ: có LLM thật / fake-LLM (đo riêng phần retrieval).

**DoD / Test `TC-603`:** report có avg + p95; mục tiêu avg ≤5s (đo với LLM thật ghi vào `eval_runs.notes`).

---

## P4-7 · Eval nội quy & routing `[T-73]`

**File:** cùng service · Phụ thuộc: P2-7, P3-2

**3 chỉ số (BR-606):**
1. **Điều/khoản đúng:** % câu nội quy có expected chunk trong top-k (`recall_at_k` tái dùng, id = (article, clause)).
2. **Groundedness:** % câu trả lời mà mọi trích dẫn `[Điều X]` đều có trong context truy được (parse regex từ answer — proxy tự động; đánh giá tay bổ sung).
3. **Routing accuracy:** % query mà `route()` trả đúng `expected_source` (chạy trên cả 40 query, gồm câu mơ hồ để đo P2-7).

**DoD / Test `TC-606`:** report đủ 3 số trên bộ eval.

---

## P4-8 · Demo checklist `[T-74]` 🔴 đích dự án

**File:** `docs/demo-checklist.md` · Phụ thuộc: mọi Must P0→P4

**4 luồng lõi (`TC-605`) — mỗi luồng ghi bước bấm + kết quả kỳ vọng:**

| # | Luồng | Kịch bản | Pass khi |
|---|---|---|---|
| 1 | Text → media | "cho xem ảnh lễ kết nạp thành viên 2024" | ảnh đúng sự kiện + caption + link nguồn |
| 2 | Ảnh → sự kiện | upload ảnh sự kiện đã nạp | "đây là sự kiện X" + ảnh cùng sự kiện |
| 3 | Nội quy | "vắng 3 buổi sinh hoạt có bị khai trừ không?" | trả lời neo Điều/Khoản + disclaimer |
| 4 | Transcript | "video nào nói về kết nạp thành viên?" | đúng video + clip/nhảy tới đoạn nói |
| + | Không bịa | câu ngoài miền ("thời tiết mai thế nào") | "chưa tìm thấy dữ liệu phù hợp", không bịa |

**Điều kiện demo:** kho mẫu ≥2 sự kiện, ≥2 video có lời nói, 1 văn bản nội quy; docker compose full-stack; chạy trơn 2 lần liên tiếp.

**DoD:** cả 5 dòng pass, quay video demo dự phòng.
