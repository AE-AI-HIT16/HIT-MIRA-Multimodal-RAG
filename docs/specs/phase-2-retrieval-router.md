# Spec P2 — Truy xuất & Router

> Phase 2 · Epic E4 + T-40 · Sprint 3 · Owner chính: AIE-1 (retrieval/tools) + AIE-2 (router)
> Mục tiêu: câu hỏi → chọn nguồn → top-k từ Qdrant → lọc ngưỡng. Mọi hàm inject `embedder=`/`store=` để test fake.
> Exit gate: router phân đúng nguồn (kể cả câu mơ hồ) · 3 nhánh retrieve trả đúng payload · rank cắt ngưỡng · tool trả `ToolResult` đúng nhãn.

## Phạm vi

| ID | Chức năng | Task | Ghi chú |
|---|---|---|---|
| P2-1 | `retrieve_media` | T-30 | 🔴 đường găng |
| P2-2 | `retrieve_regulations` | T-31 | |
| P2-3 | `retrieve_by_transcript` | T-32 | |
| P2-4 | `rank` + ngưỡng | T-33 | |
| P2-5 | Tools + registry | T-34 | contract `BaseTool` cho sẵn trong scaffold |
| P2-6 | Router luật | BR-507 | luật cho sẵn trong scaffold |
| P2-7 | Classifier fallback | T-40 | |

## Item shape chung

Mọi hàm retrieve trả `list[dict]` — hit Qdrant map phẳng:
```python
{"id": <point_id>, "score": <float>, **payload}
# media:      + video_id, timestamp, frame_path, caption
# transcript: + video_id, score(max), moments=[{start_sec, end_sec, text}, ...]
# regulation: + article/dieu, clause/khoan, text
```
Đây là **contract giữa P2 và P3** — `synthesize_answer`/`answer_regulation` đọc các khoá này.

---

## P2-1 · retrieve_media `[T-30]`

**File:** `app/domains/retrieval/service.py`

**Chữ ký:** `retrieve_media(query: str, top_k: int = 5, *, embedder=None, store=None) -> list[dict]`

**Hành vi:** `embedder.embed_query(query)` — text nhúng vào **không gian ảnh Jina-CLIP** (không cần caption trung gian) → `store.search(vector, top_k)` → `_hits_to_items`.

**Edge:** kho rỗng → `[]`; ngoài miền → điểm thấp, để P2-4 cắt (US-301.1 AC-2 xử lý ở tầng trên).

**DoD:** trả đủ trường payload + tôn trọng `top_k`; `TC-301` recall@k đo thật ở P4.

---

## P2-2 · retrieve_regulations `[T-31]`

**Chữ ký:** `retrieve_regulations(query: str, top_k: int = 5, *, embedder=None, store=None) -> list[dict]`

**Hành vi:** `embedder.embed([query])[0]` (Vietnamese_Embedding — KHÔNG dùng CLIP cho text nội quy) → search `regulation_text`.

**Edge:** không chunk nào trên ngưỡng → tầng trên trả "nội quy không quy định" (US-307.1 AC-2 — do P2-4 + P3-2 phối hợp).

**DoD:** `TC-307`: điều/khoản đúng, điểm giảm dần.

---

## P2-3 · retrieve_by_transcript `[T-32]`

**Chữ ký:** `retrieve_by_transcript(query: str, top_k: int = 5, *, embedder=None, store=None) -> list[dict]`

**Hành vi (gộp theo video — US-308.1 AC-2):**
1. Search `video_transcript` lấy dư (k×hệ số) hit mức chunk.
2. Group theo `video_id`: score video = **max** score chunk; `moments` = list `{start_sec, end_sec, text}` các chunk khớp (sắp theo thời gian).
3. Trả top-k **video** (không phải chunk) → 1 video không nhân đôi trong kết quả.

**Nối P3/P4:** `moments` là đầu vào cắt clip (P3-6) và render "nhảy tới phút…" (P4-2).

**DoD:** `TC-308`: đúng video + đoạn; video có nhiều chunk khớp chỉ xuất hiện 1 lần.

---

## P2-4 · rank + ngưỡng `[T-33]`

**Chữ ký:** `rank(candidates: list[dict], threshold: float = 0.0) -> list[dict]`

**Hành vi:** lọc `score >= threshold` → sort giảm dần theo score. **Tất cả dưới ngưỡng → `[]`** — tín hiệu "không tìm thấy" chuẩn cho P3 (US-306.1 AC-2: trả không-tìm-thấy thay vì kết quả yếu).

**Tie-break (nâng cấp sau):** điểm bằng → ưu tiên `posted_at` mới hơn / metadata đầy đủ hơn.

**Chọn threshold:** để 0.0 tới khi có bộ eval (P4-4) — calibrate trên phân bố score thật, tách threshold riêng media/text (không gian khác nhau, thang điểm khác nhau).

**DoD:** `TC-306` `test_ranking_and_threshold`.

---

## P2-5 · Tools + registry `[T-34]`

**File:** `app/tools/{base,registry,media_tool,regulation_tool}.py`

**Contract:**
```python
@dataclass
class ToolResult: source: str; items: list[Any]

class BaseTool(ABC):
    name: str; description: str          # description để classifier/agent v2 hiểu tool
    def run(self, query, **kwargs) -> ToolResult
```

**Impl:** `MediaSearchTool.run(query, *, top_k=5, threshold=0.0)` → `rank(retrieve_media(...), threshold)` → `ToolResult(source="media", items=…)`; tương tự `RegulationSearchTool` (source="regulation"). Import retrieval service lazy trong `run` (giữ import tool nhẹ).

**Registry:** `register(tool)` / `get(name)` — P3-4 resolve tool theo `Source`; đây là đường nâng lên Agentic RAG v2 (LLM tự chọn tool từ cùng registry).

**DoD:** `test_tools.py`: đúng nhãn nguồn + threshold có tác dụng.

---

## P2-6 · Router luật `[BR-507]`

**File:** `app/routing/intent.py`

**Chữ ký:** `route(text: str | None, has_image: bool, override: Source | None = None) -> Source` với `Source = MEDIA | REGULATION | BOTH`

**Thứ tự luật (deterministic, log được):**
1. `override` → dùng luôn (người dùng ép chế độ — US-507.1 edge).
2. `has_image=True` → `MEDIA` (ảnh luôn là truy vấn media).
3. Text chứa từ khóa nội quy (`nội quy, quy chế, điều, khoản, quy định, được phép, cấm, vi phạm`) → `REGULATION`.
4. Mặc định → `MEDIA` (sẽ thay bằng P2-7).

**DoD:** `test_router.py` các case luật.

---

## P2-7 · Classifier fallback `[T-40]`

**File:** `app/routing/intent.py` (mở rộng)

**Vấn đề:** câu không ảnh + không từ khóa ("thành viên mới cần làm gì?" — thực chất nội quy) hiện rơi mặc định MEDIA → sai nguồn.

**Thiết kế (đề xuất — chọn 1, ưu tiên A):**

**Phương án A — LLM zero-shot (khuyến nghị v1):**
```python
def classify(text: str, *, llm=None) -> Source:
    """Gọi 1 lượt LLM: 'Câu sau hỏi về TƯ LIỆU media hay NỘI QUY hay CẢ HAI?
    Trả đúng 1 từ: media|regulation|both'. Parse strict; lỗi/không parse được → MEDIA."""
```
- Ưu: không cần train/eval set trước; đủ tốt với Gemini Flash-Lite; injectable (`llm=`) test bằng fake.
- Nhược: +1 call LLM/câu-mơ-hồ (chỉ câu lọt luật mới gọi → tần suất thấp); cache theo text.

**Phương án B — embedding similarity:** so cosine câu hỏi với 2 tập câu mẫu (media/regulation) bằng Vietnamese_Embedding, ngưỡng chênh nhỏ → BOTH. Ưu: không tốn LLM call; nhược: cần bộ câu mẫu + calibrate.

**Tích hợp vào `route`:**
```python
# bước 4 thay mặc định:
return classify(t, llm=llm)   # llm=None → get_llm() từ deps; lỗi → Source.MEDIA (an toàn)
```

**Edge:**
- Classifier trả BOTH → P3-4 chạy cả 2 tool rồi gộp.
- LLM lỗi/timeout → fallback `MEDIA` + log (không chặn chat).
- Vẫn cho override thủ công thắng tất cả.

**DoD / Test:** `test_ambiguous_uses_classifier` (fake llm trả "regulation" → route ra REGULATION); LLM hỏng → MEDIA không raise. Routing accuracy đo ở P4-7 (BR-606).
