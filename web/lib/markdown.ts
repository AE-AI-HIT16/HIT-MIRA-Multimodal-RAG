/**
 * Bộ phân tích markdown tối giản cho câu trả lời của agent.
 *
 * Vì sao tự viết thay vì thêm thư viện: câu trả lời do LLM sinh ra, và gần như
 * mọi thư viện markdown đều kết thúc ở `dangerouslySetInnerHTML`. Ở đây hàm chỉ
 * trả về **dữ liệu**, phần render dựng thành phần tử React — nên không có đường
 * nào để chuỗi từ LLM biến thành HTML. Đổi lại là chỉ hỗ trợ đúng tập cú pháp
 * agent thật sự dùng; thứ gì không nhận ra thì hiện nguyên văn, không nuốt mất.
 *
 * Cần nhớ khi mở rộng: dùng lúc đang stream nữa, nên chuỗi thường bị cắt giữa
 * chừng. Một `**` chưa đóng phải hiện ra như chữ thường chứ không được nuốt
 * phần còn lại của câu.
 */

/** Một mẩu chữ trong dòng. */
export type Manh =
  | { loai: "chu"; chu: string }
  | { loai: "dam"; chu: string }
  | { loai: "ma"; chu: string }
  | { loai: "lien_ket"; chu: string; href: string };

/** Một mục trong danh sách. `cap` 1 là mục con. */
export type MucDanhSach = { cap: 0 | 1; noi_dung: Manh[] };

/** Một khối trong câu trả lời. */
export type Khoi =
  | { loai: "doan"; dong: Manh[][] }
  | { loai: "danh_sach"; co_so: boolean; muc: MucDanhSach[] };

// Gạch đầu dòng: "- ", "* ", "• ". Đánh số: "1. ", "2) ".
const RE_GACH_DAU_DONG = /^\s{0,6}[-*•]\s+(.*)$/;
const RE_DANH_SO = /^\s{0,6}\d{1,2}[.)]\s+(.*)$/;
// Nội quy dùng "+" cho mục con: "- Đối với cá nhân." rồi "+ Lần 1: Phạt 20.000 đồng."
// Gộp phẳng vào cùng một mức thì mất hẳn quan hệ "mức phạt thuộc về đối tượng nào".
const RE_MUC_CON = /^\s{0,6}\+\s+(.*)$/;

/**
 * Cắt một dòng thành các mẩu chữ đậm / mã / liên kết.
 *
 * Quét một lượt bằng chỉ số thay vì thay thế lồng nhau, nên `**a `b` c**` không
 * làm hỏng nhau và cặp chưa đóng ở cuối chuỗi rơi về chữ thường.
 */
export function tachManh(dong: string): Manh[] {
  const ket_qua: Manh[] = [];
  let dem = "";
  const day = () => {
    if (dem) {
      ket_qua.push({ loai: "chu", chu: dem });
      dem = "";
    }
  };

  let i = 0;
  while (i < dong.length) {
    // **đậm**
    if (dong.startsWith("**", i)) {
      const dong_cuoi = dong.indexOf("**", i + 2);
      if (dong_cuoi > i + 2) {
        day();
        ket_qua.push({ loai: "dam", chu: dong.slice(i + 2, dong_cuoi) });
        i = dong_cuoi + 2;
        continue;
      }
    }
    // `mã`
    if (dong[i] === "`") {
      const dong_cuoi = dong.indexOf("`", i + 1);
      if (dong_cuoi > i + 1) {
        day();
        ket_qua.push({ loai: "ma", chu: dong.slice(i + 1, dong_cuoi) });
        i = dong_cuoi + 1;
        continue;
      }
    }
    // [nhãn](url)
    if (dong[i] === "[") {
      const het_nhan = dong.indexOf("](", i);
      const het_url = het_nhan > 0 ? dong.indexOf(")", het_nhan) : -1;
      if (het_nhan > i && het_url > het_nhan) {
        const href = dong.slice(het_nhan + 2, het_url);
        if (/^https?:\/\//i.test(href)) {
          day();
          ket_qua.push({ loai: "lien_ket", chu: dong.slice(i + 1, het_nhan), href });
          i = het_url + 1;
          continue;
        }
      }
    }
    // URL trần
    if (dong.startsWith("http://", i) || dong.startsWith("https://", i)) {
      const con_lai = dong.slice(i);
      const khop = con_lai.match(/^https?:\/\/[^\s<>"')]+/);
      if (khop) {
        // Dấu câu cuối câu không thuộc về URL: "…posts/123." → bỏ dấu chấm ra.
        const url = khop[0].replace(/[.,;:!?]+$/, "");
        day();
        ket_qua.push({ loai: "lien_ket", chu: url, href: url });
        i += url.length;
        continue;
      }
    }
    dem += dong[i];
    i += 1;
  }
  day();
  return ket_qua;
}

/**
 * Cắt câu trả lời thành các khối đoạn văn và danh sách.
 *
 * Dòng trống ngăn khối. Các dòng gạch đầu dòng liền nhau gộp thành một danh
 * sách; dòng thường liền nhau gộp thành một đoạn và **giữ nguyên chỗ ngắt dòng**
 * — agent xuống dòng là có ý, nối lại thành một khối liền là đúng cái lỗi đang
 * làm nội quy dính thành bức tường chữ.
 */
export function tachKhoi(van_ban: string): Khoi[] {
  const khoi: Khoi[] = [];
  let doan: Manh[][] = [];
  let muc: MucDanhSach[] = [];
  let co_so = false;

  const chot_doan = () => {
    if (doan.length) khoi.push({ loai: "doan", dong: doan });
    doan = [];
  };
  const chot_danh_sach = () => {
    if (muc.length) khoi.push({ loai: "danh_sach", co_so, muc });
    muc = [];
  };

  for (const dong_tho of (van_ban ?? "").split("\n")) {
    const dong = dong_tho.trimEnd();
    if (!dong.trim()) {
      // Dòng trống chốt ĐOẠN nhưng không chốt DANH SÁCH. Chunk nội quy ngăn các
      // gạch đầu dòng bằng đúng một dòng trống (đo trên chunk thật), nên chốt
      // vội ở đây làm mỗi gạch đầu dòng thành một danh sách một-mục riêng lẻ.
      // Danh sách đóng lại khi gặp dòng chữ thường, hoặc khi hết chuỗi.
      chot_doan();
      continue;
    }
    const gach = dong.match(RE_GACH_DAU_DONG);
    const so = dong.match(RE_DANH_SO);
    const con = dong.match(RE_MUC_CON);
    if (gach || so || con) {
      chot_doan();
      const la_so = !!so;
      // Đổi kiểu danh sách giữa chừng thì mở danh sách mới, đừng trộn.
      // Mục con "+" luôn thuộc về danh sách đang mở, nên không tính là đổi kiểu.
      if (muc.length && !con && la_so !== co_so) chot_danh_sach();
      if (!con) co_so = la_so;
      muc.push({
        cap: con ? 1 : 0,
        noi_dung: tachManh((gach ?? so ?? con)![1]),
      });
      continue;
    }
    // Dòng chữ thường mới là thứ kết thúc một danh sách.
    chot_danh_sach();
    doan.push(tachManh(dong));
  }
  chot_danh_sach();
  chot_doan();
  return khoi;
}
