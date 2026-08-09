/**
 * Đọc **kết quả** tool trong stream — thứ agent thật sự đã đọc để viết câu trả lời.
 *
 * Trước đây giao diện chỉ đọc *tool call* (tham số agent điền) rồi tự gọi lại
 * `/api/media/search` để dựng thẻ. Hai lượt tra khác nhau thì không có gì buộc
 * chúng khớp nhau, và đo được là chúng không khớp: cùng câu "cho anh xem ảnh
 * Open Day 2024", agent gọi `top_k: 10, source: "clip"` còn giao diện gọi
 * `top_k: 6, source: "both"`. Câu trả lời viết "em tìm thấy 10 kết quả" trong
 * khi bên dưới hiện 6 thẻ, và 6 thẻ đó **không phải** 6 trong 10 cái kia.
 *
 * Kết quả tool đi qua SSE nguyên vẹn (`type: "tool"`, `content` là JSON đầy đủ
 * của API, đo được 12,5 KB cho một lượt 10 clip), nên không cần gọi lại gì cả:
 * đọc thẳng chỗ này thì chữ và thẻ chắc chắn cùng một lượt truy xuất, và số
 * `[n]` trong `context` trỏ đúng vào thẻ thứ n.
 *
 * Vẫn giữ đường tra lại bằng REST cho hai trường hợp tool không phục vụ được:
 * truy vấn bằng ảnh (MCP `search_media` chỉ nhận chữ) và agent sập trước khi
 * tool kịp trả về.
 */

import type {
  LGMessage,
  MediaSearchResponse,
  RegulationSearchResponse,
} from "./types";
import type { NguonTraCuu } from "./toolcall";

const TOOL_SANG_NGUON: Record<string, NguonTraCuu> = {
  search_media: "media",
  search_regulations: "regulation",
};

export type KetQuaTool =
  | { nguon: "media"; payload: MediaSearchResponse; loi?: undefined }
  | { nguon: "regulation"; payload: RegulationSearchResponse; loi?: undefined }
  | { nguon: NguonTraCuu; payload: null; loi: string };

/** Gộp `content` của ToolMessage thành chuỗi — nó là string hoặc mảng part. */
function chuTuContent(content: LGMessage["content"]): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((p) => p?.type === "text")
    .map((p) => p.text ?? "")
    .join("");
}

/**
 * `null` nếu message này không phải kết quả của một tool truy xuất.
 *
 * Đọc phòng thủ như `docToolCall`: MCP `log_exceptions` cố ý biến mọi lỗi thành
 * `{"success": false, "error_type": ..., "message": ...}` chứ không ném ra, nên
 * một payload hợp lệ về mặt JSON vẫn có thể là một lượt tra hỏng. Phân biệt được
 * hai cái mới quyết định đúng: hỏng thì tra lại bằng REST, còn rỗng-thật thì
 * hiện "không tìm thấy" chứ không tra lại để rồi ra một câu trả lời khác.
 */
export function docKetQuaTool(msg: LGMessage): KetQuaTool | null {
  if (!msg || msg.type !== "tool") return null;
  const nguon = TOOL_SANG_NGUON[msg.name ?? ""];
  if (!nguon) return null;

  const chu = chuTuContent(msg.content).trim();
  if (!chu) return { nguon, payload: null, loi: "tool không trả về nội dung" };

  let payload: unknown;
  try {
    payload = JSON.parse(chu);
  } catch {
    // ToolMessage lỗi thường là chuỗi thông báo chứ không phải JSON.
    return { nguon, payload: null, loi: chu.slice(0, 200) };
  }
  if (!payload || typeof payload !== "object") {
    return { nguon, payload: null, loi: "tool trả về dữ liệu không đọc được" };
  }

  const doi_tuong = payload as Record<string, unknown>;
  if (doi_tuong.success === false || msg.status === "error") {
    const thong_bao =
      typeof doi_tuong.message === "string" ? doi_tuong.message : "tool lỗi";
    return { nguon, payload: null, loi: thong_bao };
  }

  // Từ đây coi như đúng hợp đồng của API. Các hàm dựng thẻ đều đọc bằng `?? []`
  // nên một trường thiếu chỉ làm mất phần đó chứ không làm hỏng cả khối.
  return nguon === "media"
    ? { nguon, payload: payload as MediaSearchResponse }
    : { nguon, payload: payload as RegulationSearchResponse };
}
