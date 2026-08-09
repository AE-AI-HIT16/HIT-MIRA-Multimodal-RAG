/**
 * Nhận diện câu xã giao để **không** gọi truy xuất.
 *
 * Vì sao cần: thẻ kết quả chạy song song với stream của agent (xem
 * `napTheKetQua` trong `app/page.tsx`), nên nó gọi `/api/media/search` trước
 * khi biết người dùng hỏi gì. Qdrant luôn trả về `top_k` láng giềng gần nhất và
 * hệ thống **cố ý không lọc theo điểm** (ngưỡng "không tìm thấy" chưa hiệu
 * chuẩn được — kho nội quy mới có 4 chunk), nên "hello" vẫn ra một nắm video
 * ngẫu nhiên với điểm ~0.2. Agent trả lời chào hỏi đúng rồi; chỗ sai chỉ là
 * nhánh thẻ kết quả.
 *
 * Nguyên tắc: **mặc định là tra cứu**. Chỉ bỏ qua khi TOÀN BỘ tin nhắn khớp một
 * mẫu xã giao đã liệt kê. Bỏ nhầm một câu hỏi thật tệ hơn nhiều so với hiện thừa
 * vài thẻ, nên ở đây không có luật suy đoán nào — không có "câu ngắn thì bỏ",
 * không có "chứa từ chào thì bỏ" ("chào mừng tân sinh viên" là truy vấn thật).
 */

/**
 * Hạ chữ thường, bỏ dấu, bỏ dấu câu và emoji.
 *
 * `đ` phải đổi thành `d` **trước** khi bỏ dấu: NFD không tách `đ` nên bước strip
 * combining mark sẽ xoá luôn chữ đó — "đi đâu đấy" thành "i au ay". Đây đúng là
 * cái bẫy đã cắn bộ lọc boilerplate của ASR và hàm `slugify` bên API.
 */
export function chuanHoa(s: string): string {
  return s
    .toLowerCase()
    .replace(/đ/g, "d")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

// Xưng hô và tiểu từ tình thái: đi kèm được nhưng không thêm nội dung nào.
// Chỉ dùng bên trong mẫu đã neo hai đầu, không dùng để "bào" chuỗi tự do —
// bào tự do sẽ nuốt mất truy vấn thật ("ảnh mọi người" rụng hết còn rỗng).
const XH = "(?:ban|em|anh|chi|minh|cau|may|tao|moi nguoi|cac ban|ae|mira|bot|ad|admin)";
const TT = "(?:a|ah|oi|nhe|nha|nhi|the|vay|voi|day|do|nay|ne|hen|na|nho|roi|nhieu)";
/** Đuôi tuỳ chọn: vài xưng hô rồi vài tiểu từ, ví dụ "chào bạn nhé ạ". */
const DUOI = `(?: ${XH})*(?: ${TT})*`;

const MAU_XA_GIAO: RegExp[] = [
  // Chào hỏi
  new RegExp(`^(?:xin )?(?:chao|hi+|hey+|hello+|helo+|halo|hallo|alo+|yo)${DUOI}$`),
  // Cảm ơn
  new RegExp(
    `^(?:ok(?:e|ay|ie)? )?(?:cam on|cam on nhieu|camon|cam on nhe|thank you|thankyou|thanks|thank|tks|thks|thx|ty)${DUOI}$`,
  ),
  // Tạm biệt
  new RegExp(
    `^(?:tam biet|bye+|bai bai|goodbye|good bye|hen gap lai|gap lai sau|ngu ngon|good night)${DUOI}$`,
  ),
  // Đáp ngắn, tán thành, cười
  new RegExp(
    `^(?:ok(?:e|ay|ie|la)?|dc|duoc|duoc roi|vang|da|u+|um|uh|ukm|uk|hieu roi|ro roi|biet roi|tot|hay qua|tuyet|tuyet voi|good|nice|great|cool|ha+ha+|hi+hi+|he+he+|lol)${DUOI}$`,
  ),
  // Hỏi danh tính / năng lực của bot — agent tự trả lời từ prompt, không cần kho
  new RegExp(
    `^(?:${XH} )?(?:la ai|ten (?:la )?gi|ten gi|lam duoc gi|lam duoc nhung gi|giup duoc gi|giup duoc nhung gi|co the lam gi|lam gi duoc|biet lam gi|giup gi duoc|co gi hay)${DUOI}$`,
  ),
  // Hỏi thăm
  new RegExp(
    `^(?:${XH} )?(?:khoe khong|co khoe khong|dao nay (?:the nao|sao roi)|the nao roi|sao roi|on khong|dang lam gi)${DUOI}$`,
  ),
  // Gõ thử
  /^(?:test|testing|thu xem|123|1234|abc|a+|x+)$/,
];

/**
 * `true` khi cả tin nhắn chỉ là xã giao — không có gì để tra trong kho.
 *
 * Tin nhắn chỉ gồm emoji hoặc dấu câu (":))", "😊", "???") chuẩn hoá xong thành
 * rỗng: cũng không có gì để tra, nên tính là xã giao.
 */
export function laXaGiao(text: string): boolean {
  const s = chuanHoa(text);
  if (!s) return true;
  return MAU_XA_GIAO.some((mau) => mau.test(s));
}
