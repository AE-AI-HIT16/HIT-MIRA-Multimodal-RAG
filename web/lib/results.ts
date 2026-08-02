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

/** Nội quy: dòng đầu của chunk chính là tiêu đề mục, dùng luôn làm nhãn. */
export function regulationsToItems(res: RegulationSearchResponse): ChatItem[] {
  return (res.results ?? []).map((hit) => {
    const dongDau = (hit.text ?? "").split("\n")[0]?.trim() ?? "";
    return {
      score: hit.score,
      text: hit.text,
      // `article`/`clause` là kiểu số của thiết kế cũ; nội quy thật không đánh
      // số điều/khoản mà chia theo mục có tiêu đề, nên dùng nhãn chuỗi.
      label: dongDau.length > 90 ? `${dongDau.slice(0, 90)}…` : dongDau || "Nội quy",
      heading: hit.section,
    };
  });
}

/** Gộp cả ba nguồn, xếp theo điểm giảm dần để thẻ mạnh nhất lên trước. */
export function gopKetQua(
  media?: MediaSearchResponse | null,
  noiQuy?: RegulationSearchResponse | null,
): ChatItem[] {
  const items: ChatItem[] = [
    ...(media ? clipsToItems(media) : []),
    ...(media ? transcriptsToItems(media) : []),
    ...(noiQuy ? regulationsToItems(noiQuy) : []),
  ];
  return items.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
}
