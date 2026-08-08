import type {
  ChatItem,
  MediaSearchResponse,
  RegulationSearchResponse,
} from "./types";

/**
 * Đổi kết quả API thật thành `ChatItem` mà `ResultCard` đã biết render.
 *
 * `ResultCard` được viết cho một API `/chat/message` chưa bao giờ tồn tại: nó
 * chờ `items` với id kiểu số, `article`/`clause`, `frames[]`. API thật trả về
 * `clips` / `videos` / `results` với id UUID và khoá object MinIO. Lớp chuyển
 * đổi mỏng này rẻ hơn nhiều so với viết lại toàn bộ giao diện, và giữ nguyên
 * những thứ đã làm kỹ trong đó (clamp caption, chip mốc thời gian, link nguồn).
 */

/** Ảnh tĩnh và keyframe video → thẻ ảnh / thẻ video có mốc thời gian. */
export function clipsToItems(res: MediaSearchResponse): ChatItem[] {
  return (res.clips ?? []).map((clip) => ({
    score: clip.score,
    caption: clip.caption,
    source_url: clip.source_url,
    object_key: clip.frame_object_key,
    // Ảnh tĩnh KHÔNG được mang mốc giây — payload đã cố ý bỏ trống, nên ở đây
    // chỉ chuyển tiếp chứ tuyệt đối không tự suy ra một con số.
    timestamp: clip.media_kind === "video_frame" ? clip.timestamp_sec ?? undefined : undefined,
    video_uid: clip.media_kind === "video_frame" ? clip.video_id : null,
    media_type: clip.media_kind,
  }));
}

/** Lời thoại: mỗi video một thẻ, các đoạn khớp thành chip mốc thời gian. */
export function transcriptsToItems(res: MediaSearchResponse): ChatItem[] {
  return (res.videos ?? []).map((video) => ({
    score: video.score,
    video_uid: video.video_id,
    source_url: video.source_url,
    moments: (video.moments ?? []).map((m) => ({
      start_sec: m.start_sec,
      end_sec: m.end_sec,
      text: m.text,
    })),
  }));
}

/**
 * Ngưỡng điểm top-1 để coi câu hỏi là thật sự thuộc nhánh nội quy.
 *
 * Kho nội quy chỉ có 4 chunk còn `top_k` là 4, nên **mọi** truy vấn đều nhận về
 * nguyên kho — hỏi "teambuilding 2025" vẫn ra đủ 4 mục nội quy. Khi thẻ kết quả
 * còn nằm cuối trang thì vô hại; đặt vào trong bong bóng trả lời thì thành lạc đề.
 *
 * Đo ngày 08/08/2026 trên chính kho đang chạy, điểm top-1 của `/api/retrieval/search`:
 *
 *   teambuilding 2025            0.144   ] câu không thuộc nhánh nội quy
 *   ảnh du lịch Bản Lác          0.294   ]
 *   giá vàng hôm nay             0.275   ]
 *   làm mất chìa khóa phòng      0.545   ] câu thuộc nhánh nội quy
 *   quy định sử dụng phòng CLB   0.656   ]
 *
 * Hai nhóm tách hẳn (0.294 vs 0.545), nên 0.42 nằm giữa khoảng trống. Ba điều
 * phải nhớ trước khi động vào con số này:
 *
 * - **Chỉ đúng cho không gian Azure `text-embedding-3-small` của nhánh nội quy.**
 *   Trên không gian media, `docs/eval-report.md` §4 đo được điểm trong miền và
 *   ngoài miền chồng lấn — ở đó không ngưỡng nào tách được, nên đừng nhân bản
 *   ý tưởng này sang `clipsToItems`.
 * - **Khoảng cách hạng 1 với hạng 2 không dùng được**: 0.016 ở câu đúng nhánh,
 *   0.029 ở câu lạc đề — ngược dấu. Phải nhìn điểm tuyệt đối của hạng 1.
 * - **n = 5 và kho mới có 4 chunk.** Đo lại khi kho vượt ~50 chunk, cùng lúc với
 *   phép đo top-1 mà `CLAUDE.md` đã hẹn.
 *
 * Dưới ngưỡng thì thẻ **không bị xoá**, chỉ bị gấp lại — im lặng vứt bằng chứng
 * là thứ khiến người dùng tưởng kho không có gì.
 */
export const NGUONG_NOI_QUY_DUNG_NHANH = 0.42;

/** Nội quy: dòng đầu của chunk chính là tiêu đề mục, dùng luôn làm nhãn. */
export function regulationsToItems(res: RegulationSearchResponse): ChatItem[] {
  const diemCaoNhat = Math.max(0, ...(res.results ?? []).map((hit) => hit.score ?? 0));
  const dungNhanh = diemCaoNhat >= NGUONG_NOI_QUY_DUNG_NHANH;

  return (res.results ?? []).map((hit) => {
    const dongDau = (hit.text ?? "").split("\n")[0]?.trim() ?? "";
    return {
      score: hit.score,
      text: hit.text,
      // `article`/`clause` là kiểu số của thiết kế cũ; nội quy thật không đánh
      // số điều/khoản mà chia theo mục có tiêu đề, nên dùng nhãn chuỗi.
      label: dongDau.length > 90 ? `${dongDau.slice(0, 90)}…` : dongDau || "Nội quy",
      heading: hit.section,
      // Cả cụm cùng đứng hoặc cùng gấp: 4 chunk luôn về cùng nhau, tách lẻ từng
      // chunk chỉ tạo ra một mảnh nội quy mồ côi giữa lưới ảnh.
      lienQuanYeu: !dungNhanh,
    };
  });
}

/**
 * Gộp cả ba nguồn, xếp theo điểm giảm dần để thẻ mạnh nhất lên trước.
 *
 * Lưu ý điểm của hai nhánh **không cùng thang**: media là cosine trong không gian
 * Jina-CLIP v2, nội quy là cosine trong không gian Azure 1.536 chiều. Xếp chung
 * chỉ là xấp xỉ cho dễ nhìn, không phải một phép so sánh có nghĩa. Nó chạy được
 * là nhờ thẻ nội quy lạc nhánh đã bị gắn `lienQuanYeu` và đẩy xuống cuối trước.
 */
export function gopKetQua(
  media?: MediaSearchResponse | null,
  noiQuy?: RegulationSearchResponse | null,
): ChatItem[] {
  const items: ChatItem[] = [
    ...(media ? clipsToItems(media) : []),
    ...(media ? transcriptsToItems(media) : []),
    ...(noiQuy ? regulationsToItems(noiQuy) : []),
  ];
  return items.sort((a, b) => {
    if (!!a.lienQuanYeu !== !!b.lienQuanYeu) return a.lienQuanYeu ? 1 : -1;
    return (b.score ?? 0) - (a.score ?? 0);
  });
}
