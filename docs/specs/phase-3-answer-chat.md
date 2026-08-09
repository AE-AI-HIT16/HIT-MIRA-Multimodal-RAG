# Spec P3 — Sinh câu trả lời & Chat

> Phase 3 · Epic E5 + E6 · Sprint 4 · Owner chính: AIE-2
> Mục tiêu: ghép lõi RAG thành **một luồng `POST /chat/message`**; câu trả lời bám nguồn, có trích dẫn, KHÔNG bịa; kèm auth + media serving.
> Exit gate: chat text ± ảnh → định tuyến đúng → trả lời có citation (media) / điều-khoản + disclaimer (nội quy) / "không tìm thấy" khi rỗng; non-admin bị chặn endpoint admin; clip/seek chạy.

## Phạm vi

| ID | Chức năng | Task | Ghi chú |
|---|---|---|---|
| P3-1 | `synthesize_answer` | T-41 | 🔴 đường găng |
| P3-2 | `answer_regulation` | T-42 | |
| P3-3 | `handle_not_found` | T-43 | |
| P3-4 | `handle_message` + `/chat/message` | T-50 | 🔴 đường găng — mắt xích ráp toàn hệ |
| P3-5 | Auth JWT | T-51 | |
| P3-6 | Media serving: clip + stream/seek | T-52 | |
| P3-7 | Lưu conversations/messages | T-53 | |

## Nguyên tắc chống bịa (áp toàn Phase)

1. Context rỗng → `handle_not_found`, **không gọi LLM** (US-406.1 AC-1).
2. LLM lỗi/timeout → fallback liệt kê nguồn thô + báo lỗi nhẹ, **không sập** (US-401.1 AC-2).
3. Nội quy: chỉ trả lời từ điều/khoản truy được, **luôn** kèm disclaimer, không suy diễn vượt văn bản (US-407.1).

---

## P3-1 · synthesize_answer `[T-41]`

**File:** `app/domains/answer/service.py`

**Chữ ký:** `synthesize_answer(query: str, hits: list[dict], *, llm=None) -> dict {"text", "sources"}`

**Hành vi:**
1. `hits` rỗng → return `handle_not_found(query)`.
2. Dựng prompt grounded: mỗi hit thành dòng `[n] <label>: <snippet>` — `_source_label` ("Video 3 @ 00:42" / "Ảnh 5"), `_snippet` ưu tiên `caption → transcript/text → summary`, thiếu hết → "(không có mô tả text)".
3. Prompt ràng: *chỉ dùng thông tin trong nguồn, trích `[n]`, không thêm sự kiện ngoài ngữ cảnh, trả lời tiếng Việt*.
4. `llm.generate(prompt)`; Exception → `_raw_fallback(hits)`: text "Hệ thống sinh câu trả lời đang lỗi, đây là kết quả truy xuất:" + list nguồn.
5. `sources` = hits gốc (đủ trường cho web render ảnh/clip/link).

**DoD:** bám nguồn (đánh giá tay); rỗng → không gọi LLM; LLM lỗi → fallback. Test: `test_answer.py`.

---

## P3-2 · answer_regulation `[T-42]`

**Chữ ký:** `answer_regulation(query: str, rule_hits: list[dict], *, llm=None) -> dict`

**Hành vi:**
1. Rỗng → `handle_not_found` biến thể "nội quy không quy định nội dung này" (AC-2).
2. Prompt neo `_rule_label` = `[Điều X Khoản Y]` (đọc `article/dieu`, `clause/khoan`); ràng: *căn cứ đúng điều khoản, tình huống cần nhiều điều → liệt kê, không suy diễn*.
3. **Mọi nhánh ra** (thành công + `_raw_regulation_fallback`) đều append `_DISCLAIMER`: "Thông tin chỉ mang tính tham khảo, quyết định cuối cùng thuộc Ban Chủ nhiệm."

**DoD:** `TC-407` citation + disclaimer; test cả nhánh lỗi vẫn có disclaimer.

---

## P3-3 · handle_not_found `[T-43]`

**Chữ ký:** `handle_not_found(query: str) -> dict {"text", "sources": []}`

**Hành vi:** message cố định "chưa tìm thấy dữ liệu phù hợp" + gợi ý chỉnh truy vấn (thêm tên sự kiện/thời gian). Không LLM. `TC-406`.

---

## P3-4 · handle_message — orchestrator `[T-50]` 🔴 mắt xích nối lõi RAG

**File:** `app/domains/chat/service.py` (đang `NotImplementedError`) + `router.py`

**API:**
```
POST /chat/message
  body: {text?: str, image_path?: str, override?: "media"|"regulation"|"both",
         top_k?: int = 5, threshold?: float = 0.0, conversation_id?: int}
  → 200 ChatReply {
      answer: str,                # câu trả lời cuối
      source: str,                # nhãn nguồn đã dùng: media|regulation|both
      items: list[dict],          # hits đa phương thức cho web render
    }
  422 nếu text và image_path đều rỗng
```

**Chữ ký service:**
```python
def handle_message(text: str | None, image_path: str | None = None, *,
                   override: Source | None = None, top_k: int = 5,
                   threshold: float = 0.0) -> dict
```

**Luồng (3 bước ráp P2 + P3):**
```
1. source = route(text, has_image=image_path is not None, override)
2. chạy tool:
   MEDIA      → registry.get("media_search").run(query, top_k=…, threshold=…)
   REGULATION → registry.get("regulation_search").run(…)
   BOTH       → chạy cả hai, giữ 2 ToolResult riêng
   (query = text; text rỗng + có ảnh → nhánh ảnh: embed ảnh → search — v1 tối thiểu
    có thể tạm dùng caption/text bắt buộc, ghi rõ giới hạn)
3. tổng hợp:
   media      → synthesize_answer(text, items)
   regulation → answer_regulation(text, items)
   both       → answer từng nguồn rồi ghép 2 đoạn, nhãn nguồn "both"
   items rỗng cả 2 nguồn → handle_not_found
```

**Edge:**
- Ảnh + text mâu thuẫn → ưu tiên ảnh (mặc định cấu hình — US-502.1).
- `override` thắng mọi luật routing (US-507.1).
- Tool raise (Qdrant down) → 503 message rõ, không 500 trần.

**DoD / Test `TC-507`:**
- Câu nội quy → source="regulation", answer chứa "Điều" + disclaimer.
- Câu media/kèm ảnh → source="media", items có `frame_path`.
- `override="regulation"` với câu media → đi regulation.
- Cả hai rỗng → text "chưa tìm thấy". End-to-end fake toàn bộ (llm/embedder/store).

---

## P3-5 · Auth JWT `[T-51]`

**File:** `app/domains/auth/*` + `app/deps.py::require_admin` + `app/core/security.py`

**API:**
```
POST /auth/register  {email, password}         → 201 (v1: role=member; admin seed bằng script)
POST /auth/login     {email, password}         → 200 {access_token, token_type: "bearer"}
```

**Service:**
```python
def login(session, email, password) -> str            # sai → 401 "Sai email hoặc mật khẩu"
def create_token(user) -> str                          # JWT HS256, sub=user.id, role, exp=24h, key=settings.jwt_secret
def require_admin(token = Depends(oauth2_scheme))      # decode → role != admin → 403
```

**Hash:** bcrypt (passlib). Endpoint gắn guard: `/ingest/*`, `/eval/*` → `require_admin`; `/chat/*` → v1 cho phép guest (đăng nhập optional), lưu message gắn user nếu có.

**DoD / Test `TC-505`:** login đúng → token decode được role; non-admin gọi `/ingest/upload` → 403; token hỏng → 401.

---

## P3-6 · Media serving — clip + stream/seek `[T-52]`

**File:** `app/domains/media/service.py` + router · Phụ thuộc: P1-2

**API:**
```
GET /media/{id}/clip?ts=42.0        → mp4 ~6s (3s trước + 3s sau ts)
GET /media/{id}/stream              → video, hỗ trợ Range request (seek phía client bằng #t=)
```

**Service:**
```python
def cut_clip(video_path: str, ts: float, *, pre=3.0, post=3.0,
             out_dir: str) -> str   # ffmpeg -ss max(0,ts-pre) -t (pre+post), co theo biên
```

**Hành vi:**
- Clip cache theo `(media_id, round(ts))` trong `data/clips/` — cắt 1 lần.
- Frame sát đầu/cuối → clip ngắn hơn 6s, không lỗi (US-403.1 AC-2).
- Cắt lỗi (codec) → 302 sang `/stream#t=<ts>` (fallback preview — US-403 edge).
- Stream: `StarletteFileResponse` hỗ trợ Range sẵn → trình duyệt seek `#t=` (US-404: mở đúng ±1s).

**DoD / Test `TC-403`/`TC-404`:** clip dài ≤6s chứa ts (ffprobe duration); ts=1.0 trên video 3s → clip ~4s; stream trả 206 khi có Range header.

---

## P3-7 · Lưu hội thoại `[T-53]`

**File:** `app/domains/chat/*` · Phụ thuộc: P0-3

**Hành vi:** mỗi lượt `/chat/message`:
1. `conversation_id` rỗng → tạo `conversations` mới (user nếu đăng nhập, guest → null).
2. Lưu message user (`role="user"`, text, image_path) + message assistant (`role="assistant"`, answer, `source_label`, `payload_json`=items rút gọn).
3. Response trả kèm `conversation_id` để client giữ luồng.

**Edge:** lưu DB lỗi → vẫn trả câu trả lời (log warning) — không để persistence chặn UX.

**DoD / Test `TC-501`:** gửi 1 câu → bảng `messages` có 2 dòng đúng role + conversation reuse khi truyền id.

---

## Thứ tự thực thi P3

```
P3-4 handle_message  ← làm NGAY (chỉ cần P2, ráp bằng fake được, mở khoá demo API)
P3-5 auth            ← song song (phụ thuộc P0-3 bảng users)
P3-7 messages        ← sau P0-3
P3-6 clip/stream     ← sau P1-2, trước P4-2 (web cần endpoint này để render clip)
```
