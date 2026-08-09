# web/ — Frontend (Next.js)

Chatbot web app cho HIT-MIRA (BR-500): khung chat, hỏi bằng ảnh, thẻ kết quả
đa phương thức (ảnh + keyframe + lời thoại + nội quy), màn hình admin.

```bash
npm run dev     # cổng 3000; cần LangGraph ở 2024 và API ở 8000
```

> Bản README cũ mô tả một bộ khung dự kiến (`chat/`, `media-viewer/`,
> `lib/api-client`) chưa bao giờ tồn tại. Dưới đây là **cây thật**.

- `app/page.tsx` — màn chat. `app/admin`, `app/login` — hai màn còn lại.
- `components/` — `Composer`, `MessageBubble`, `ResultCard`, `RichText`,
  `ChatSidebar`, `MediaImage`, `TopBar`, `EmptyState`.
- `lib/` — `api.ts` (FastAPI + LangGraph SSE), `toolcall.ts` (đọc **tham số**
  tool), `toolresult.ts` (đọc **kết quả** tool), `results.ts` (dựng thẻ),
  `markdown.ts` (parser markdown tối giản), `smalltalk.ts`, `format.ts`.

## Một câu trả lời được dựng từ đâu

Một lượt hỏi mở đúng **một** stream tới LangGraph (`stream_mode: ["messages"]`)
và lấy mọi thứ từ đó:

| Message trong stream | Dùng để |
| --- | --- |
| `type: "ai"`, có `content` | phần chữ — đi qua `RichText` |
| `type: "ai"`, có `tool_calls` | bộ lọc `years`/`events` agent suy ra (chỉ để dự phòng) |
| `type: "tool"` | **kết quả truy xuất** — nguồn dựng mọi thẻ |

**Thẻ kết quả phải dựng từ ToolMessage, không được tra lại.** ToolMessage mang
nguyên payload của API (đo được 12,5 KB cho một lượt 10 clip), nên chữ và thẻ
chắc chắn cùng một lượt truy xuất. Bản trước tra lại bằng REST và hai lượt đó
lệch nhau thật: cùng câu "cho anh xem ảnh Open Day 2024", agent gọi
`top_k: 10, source: "clip"` còn giao diện gọi `top_k: 6, source: "both"` — câu
trả lời viết "tìm thấy 10 kết quả" trong khi bên dưới hiện 6 thẻ *khác*.

Nhờ chung một lượt mà số `[n]` agent trích trong `context` trỏ đúng vào thẻ thứ
n; đó là ý nghĩa của huy hiệu `[n]` trên thẻ. Thẻ dựng bằng REST **không** có số
này — một con số trỏ sai còn tệ hơn không có số.

`napTheKetQua` (REST) chỉ còn là **đường dự phòng**, đúng hai trường hợp:

1. **Truy vấn có ảnh** — MCP `search_media` chỉ nhận chữ, nên kết quả tool của
   lượt đó là "LLM tả ảnh thành chữ rồi đi tìm". `/api/media/search-image` nhúng
   chính tấm ảnh và so với vector ảnh trong kho (US-303.1), tốt hơn hẳn.
2. **Agent sập hoặc tool lỗi** — "chữ hỏng thì ảnh vẫn phải xem được".

## Những chỗ dễ làm hỏng lại

- **Duyệt mọi message trong chunk, đừng `find(type === "ai")`.** Kết quả truy
  xuất là ToolMessage; lọc theo `"ai"` là bỏ qua sạch, đúng lỗi bản trước.
- **Nhiều keyframe của một video gộp thành một thẻ**, phần còn lại thành chip
  mốc thời gian. Không gộp thì 5 keyframe cùng video chiếm 5 ô lưới bằng 5 trình
  phát giống nhau — và `item.frames` không bao giờ được điền, khiến `TimeChips`
  thành code chết.
- **Số trên đầu khối đếm KẾT QUẢ, không đếm THẺ.** Gộp keyframe làm hai con số
  lệch nhau (10 kết quả → 8 thẻ), mà nó nằm ngay dưới câu "em tìm thấy 10".
- **`notes` của API phải hiện ra.** Đó là chỗ API nói "nhánh lời thoại bị bỏ qua
  vì truy vấn chỉ có ảnh" hay "rỗng vì lọc năm 2019". Giấu đi thì `videos: []`
  bị đọc thành "CLB không nói gì về chuyện này".
- **Đừng thu hồi objectURL của ảnh vừa gửi.** Bong bóng lượt hỏi đang dùng chính
  chuỗi `blob:` đó; `revokeObjectURL` trong cùng handler chạy trước lần vẽ đầu
  tiên, nên ảnh hiện ra vỡ. Quyền sở hữu chuyển sang lượt hỏi (`nhaAnhKhoiONhap`).
- **Điểm của hai nhánh không cùng thang** (media: Jina-CLIP v2; nội quy: Azure
  1536-d). Xếp trong từng nguồn, đừng `sort` cả mảng gộp.

## Còn nợ

- Hỏi bằng ảnh chỉ ra thẻ khi agent có gọi tool. Agent tả thẳng tấm ảnh (đo
  được với ảnh logo) thì không có thẻ nào, dù đó đúng là lúc tìm-ảnh-giống-ảnh
  có ích nhất. Đóng lại cần một tool mang được ảnh — v2, xem CLAUDE.md.
- Nhánh media không có ngưỡng lọc điểm. `docs/eval-report.md` §4 đo được điểm
  trong miền và ngoài miền chồng lấn, nên mọi ngưỡng đặt ở đây đều là số bịa.
