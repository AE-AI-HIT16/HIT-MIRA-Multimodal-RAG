/**
 * Đọc tool call của agent để biết **tra nguồn nào** và **lọc thế nào**.
 *
 * Vì sao phải đọc lại tool call thay vì tự quyết ở giao diện:
 *
 * 1. **API truy xuất không hiểu câu chữ.** `/api/media/search` là vector search
 *    thuần — đo được là gửi `query: "hình ảnh hoạt động năm 2025"` trả về lẫn
 *    2022 và 2024, còn gửi `years: [2025]` mới lọc đúng. Bước suy luận
 *    "năm 2025" → `years: [2025]` chỉ tồn tại đúng một chỗ: lúc agent điền tham
 *    số tool (nó đọc mô tả tham số trong `mcp/Resources/tools.yaml`).
 *
 * 2. **Chọn nguồn cũng là một suy luận, và cũng chỉ agent làm.** Đo trên lượt
 *    hỏi "mượn phòng phải báo trước bao lâu": agent chỉ gọi `search_regulations`
 *    (nội quy 0.623 / 0.508 / 0.387 / 0.302), nhưng giao diện vẫn tự gọi thêm
 *    kho media và nhận về 12 thẻ điểm 0.246–0.257 — dồn cục ở đáy trong dải rộng
 *    0.011, tức là không cái nào khớp. Người dùng thấy video Kahoot dưới câu trả
 *    lời về chìa khóa phòng.
 *
 * Hai nơi suy luận độc lập là hai nơi có thể lệch nhau ngay trong cùng một câu
 * trả lời, nên ở đây chỉ chép lại chứ không đoán thêm lần nào.
 */

import type { LGToolCall } from "./types";

/** Hai nhánh truy xuất của hệ thống. */
export type NguonTraCuu = "media" | "regulation";

/** Bộ lọc agent đã suy ra cho một nguồn. */
export type LocSuyLuan = {
  query?: string;
  years?: number[];
  events?: string[];
};

/**
 * Nguồn nào agent đã tra, kèm bộ lọc của nguồn đó.
 *
 * `null` mang nghĩa **khác hẳn** với `{}`: `null` là "agent chưa nói gì" (nó sập
 * trước khi kịp gọi tool), còn `{}` là "agent đã quyết định không tra nguồn nào".
 * Lẫn hai cái này thì hoặc mất sạch nguồn tham khảo lúc agent lỗi, hoặc lại đổ
 * nhiễu về như cũ.
 */
export type KeHoachTraCuu = Partial<Record<NguonTraCuu, LocSuyLuan>>;

const TOOL_SANG_NGUON: Record<string, NguonTraCuu> = {
  search_media: "media",
  search_regulations: "regulation",
};

/** Kho ảnh/video trải từ 2021; chặn trên nới rộng để khỏi phải sửa mỗi năm. */
const NAM_NHO_NHAT = 2000;
const NAM_LON_NHAT = 2100;

/**
 * `null` nếu không phải tool truy xuất.
 *
 * Đọc phòng thủ: đây là dữ liệu do LLM sinh ra, không phải hợp đồng đã kiểm
 * chứng. Một `years: ["2025"]` (chuỗi thay vì số) lọt xuống API sẽ thành 422 và
 * mất sạch thẻ, nên ép kiểu và vứt phần tử vô nghĩa ngay tại đây.
 */
export function docToolCall(
  tc: LGToolCall,
): { nguon: NguonTraCuu; loc: LocSuyLuan } | null {
  const nguon = tc && TOOL_SANG_NGUON[tc.name];
  if (!nguon) return null;
  const args = tc.args ?? {};
  const years = (Array.isArray(args.years) ? args.years : [])
    .map((n) => Number(n))
    .filter((n) => Number.isInteger(n) && n >= NAM_NHO_NHAT && n <= NAM_LON_NHAT);
  const events = (Array.isArray(args.events) ? args.events : [])
    .filter((e): e is string => typeof e === "string" && e.trim().length > 0)
    .map((e) => e.trim());
  return {
    nguon,
    loc: {
      query: typeof args.query === "string" ? args.query : undefined,
      years,
      events,
    },
  };
}
