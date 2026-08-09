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

/**
 * Ảnh tĩnh và keyframe video → thẻ ảnh / thẻ video có mốc thời gian.
 *
 * **Nhiều keyframe của cùng một video gộp thành MỘT thẻ**, các mốc còn lại thành
 * chip bấm để tua. Trước đây mỗi keyframe là một thẻ riêng, nên `item.frames`
 * chưa bao giờ được điền và dãy chip mốc thời gian trong `ResultCard` là code
 * chết. Hậu quả thấy được: một lượt trả về 5 keyframe của cùng một video chiếm
 * trọn 5 ô lưới bằng 5 trình phát cùng nội dung, đẩy hết ảnh của video khác ra
 * sau nút "Xem thêm".
 *
 * `soTrichDan` là các số `[n]` trong `context` — đánh theo vị trí trong `clips`,
 * đúng thứ tự `_format_context` của API. Thẻ video gộp mang **tất cả** số của
 * những khung nó ôm vào, chứ không chỉ số của khung khớp mạnh nhất.
 */
export function clipsToItems(res: MediaSearchResponse): ChatItem[] {
  const items: ChatItem[] = [];
  const theoVideo = new Map<string, ChatItem>();

  (res.clips ?? []).forEach((clip, i) => {
    const stt = i + 1;
    // Chỉ gộp khi có ĐỦ cả video_id lẫn mốc giây. Thiếu mốc thì không có gì để
    // làm chip, mà bịa một con số 0 là dựng ra một khoảnh khắc không tồn tại —
    // đúng thứ payload cố ý bỏ trống để chặn.
    const laKhungVideo =
      clip.media_kind === "video_frame" && !!clip.video_id && clip.timestamp_sec != null;

    if (laKhungVideo) {
      const khoa = clip.video_id as string;
      const moc = {
        timestamp: clip.timestamp_sec as number,
        score: clip.score,
        frame_path: clip.frame_object_key,
      };
      const da_co = theoVideo.get(khoa);
      if (da_co) {
        // Thẻ gộp phải mang MỌI số trích dẫn nó đại diện, kể cả khi khung bị
        // trùng giây và không thành chip riêng — agent vẫn có thể trích số đó.
        da_co.soTrichDan?.push(stt);
        // Cùng một giây xuất hiện hai lần thì chỉ giữ một chip.
        if (!da_co.frames?.some((f) => f.timestamp === moc.timestamp)) {
          da_co.frames?.push(moc);
        }
        return;
      }
      const the: ChatItem = {
        soTrichDan: [stt],
        score: clip.score,
        caption: clip.caption,
        source_url: clip.source_url,
        object_key: clip.frame_object_key,
        timestamp: moc.timestamp,
        video_uid: khoa,
        media_type: clip.media_kind,
        frames: [moc],
      };
      theoVideo.set(khoa, the);
      items.push(the);
      return;
    }

    items.push({
      soTrichDan: [stt],
      score: clip.score,
      caption: clip.caption,
      source_url: clip.source_url,
      object_key: clip.frame_object_key,
      // Ảnh tĩnh KHÔNG được mang mốc giây — payload đã cố ý bỏ trống, nên ở đây
      // chỉ chuyển tiếp chứ tuyệt đối không tự suy ra một con số.
      timestamp: undefined,
      video_uid: null,
      media_type: clip.media_kind,
    });
  });

  // Chip xếp theo thời gian để đọc ra một dòng thời gian, còn `timestamp` của
  // thẻ vẫn là khung khớp mạnh nhất — đó là khung mở ra khi bấm phát.
  for (const the of theoVideo.values()) {
    the.frames?.sort((a, b) => a.timestamp - b.timestamp);
    the.soTrichDan?.sort((a, b) => a - b);
  }
  return items;
}

/**
 * Lời thoại: mỗi video một thẻ, các đoạn khớp thành chip mốc thời gian.
 *
 * `soClip` là số phần tử `clips` đứng trước — `_format_context` đánh số clips
 * rồi mới tới videos, nên số trích dẫn của video bắt đầu từ đó.
 */
export function transcriptsToItems(res: MediaSearchResponse): ChatItem[] {
  const soClip = (res.clips ?? []).length;
  return (res.videos ?? []).map((video, i) => ({
    soTrichDan: [soClip + i + 1],
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

/**
 * Nội quy: dòng đầu của chunk chính là tiêu đề mục, dùng luôn làm nhãn.
 *
 * `daTraMedia` cho biết lượt này agent có tra kho media nữa không. Nó quyết định
 * việc gấp thẻ: xem chú thích của `NGUONG_NOI_QUY_DUNG_NHANH` bên dưới.
 */
export function regulationsToItems(
  res: RegulationSearchResponse,
  daTraMedia: boolean,
): ChatItem[] {
  const diemCaoNhat = Math.max(0, ...(res.results ?? []).map((hit) => hit.score ?? 0));
  // Agent chỉ tra nội quy → nó đã tự kết luận đây là câu hỏi nội quy, và kết
  // luận đó đáng tin hơn một ngưỡng cosine chưa hiệu chuẩn. Vẫn dùng ngưỡng lúc
  // này là tự bắn vào chân: đo được câu "mượn phòng báo trước bao lâu" chấm 0.41
  // — dưới ngưỡng 0.42 đúng 0.01 — và cả bốn mục đúng sẽ bị gấp lại thành
  // "khớp yếu" ngay dưới một câu trả lời nội quy hoàn toàn chính xác.
  const dungNhanh = !daTraMedia || diemCaoNhat >= NGUONG_NOI_QUY_DUNG_NHANH;

  return (res.results ?? []).map((hit) => {
    const dong = (hit.text ?? "").split("\n");
    const dongDau = dong[0]?.trim() ?? "";
    // Tiêu đề mục nằm ở CẢ `section` lẫn dòng đầu của `text`. In cả hai thì
    // người đọc thấy đúng một câu hai lần liên tiếp, nên bỏ dòng đầu khỏi thân.
    const tieuDe = (hit.section ?? "").trim() || dongDau;
    const than = (dongDau && dongDau === tieuDe ? dong.slice(1) : dong)
      .join("\n")
      .trim();
    return {
      score: hit.score,
      text: than || hit.text,
      // `article`/`clause` là kiểu số của thiết kế cũ; nội quy thật không đánh
      // số điều/khoản mà chia theo mục có tiêu đề, nên dùng nhãn chuỗi.
      label: tieuDe.length > 90 ? `${tieuDe.slice(0, 90)}…` : tieuDe || "Nội quy",
      heading: hit.section,
      // Cả cụm cùng đứng hoặc cùng gấp: 4 chunk luôn về cùng nhau, tách lẻ từng
      // chunk chỉ tạo ra một mảnh nội quy mồ côi giữa lưới ảnh.
      lienQuanYeu: !dungNhanh,
    };
  });
}

/** Thẻ kết quả kèm những gì API tự nói về lượt tra đó. */
export interface KetQuaGop {
  items: ChatItem[];
  notes: string[];
}

/**
 * Gộp hai nguồn thành danh sách thẻ, **xếp trong từng nguồn chứ không xếp chéo**.
 *
 * Điểm của hai nhánh không cùng thang: media là cosine trong không gian
 * Jina-CLIP v2, nội quy là cosine trong không gian Azure 1.536 chiều. Bản trước
 * `sort` cả mảng gộp, tức là so hai con số không so được với nhau; nó không gây
 * hại chỉ vì `MessageBubble` tách lại thành hai nhóm ngay sau đó. Xếp riêng từng
 * nguồn cho ra đúng thứ tự ấy mà không phải mượn một phép so sánh vô nghĩa.
 *
 * Trong nhánh media thì clip và lời thoại **có** chung thang — cùng một không
 * gian Jina — nên xếp chung là đúng, và đó chính là tính chất một-lần-nhúng.
 */
export function gopKetQua(
  media?: MediaSearchResponse | null,
  noiQuy?: RegulationSearchResponse | null,
): KetQuaGop {
  const theMedia = media
    ? [...clipsToItems(media), ...transcriptsToItems(media)].sort(
        (a, b) => (b.score ?? 0) - (a.score ?? 0),
      )
    : [];
  const theNoiQuy = noiQuy
    ? regulationsToItems(noiQuy, !!media).sort((a, b) => {
        if (!!a.lienQuanYeu !== !!b.lienQuanYeu) return a.lienQuanYeu ? 1 : -1;
        return (b.score ?? 0) - (a.score ?? 0);
      })
    : [];

  return { items: [...theNoiQuy, ...theMedia], notes: docNotes(media) };
}

/**
 * Những gì API tự nói về lượt tra — nhánh bị bỏ qua, rỗng vì bộ lọc, nhánh lỗi.
 *
 * API sinh ra `notes` để chặn đúng một cách hiểu sai: đọc `videos: []` thành
 * "CLB không nói gì về chuyện này", trong khi thật ra nhánh lời thoại đã bị bỏ
 * qua vì truy vấn chỉ có ảnh. Giao diện khai trường này trong `types.ts` nhưng
 * chưa từng đọc, nên lời cảnh báo đó chưa bao giờ tới được người đọc.
 *
 * `errors` gộp chung vào đây: với người dùng thì "nhánh này hỏng" và "nhánh này
 * bị bỏ qua" cùng là một điều cần biết — kết quả đang thiếu một phần.
 */
function docNotes(media?: MediaSearchResponse | null): string[] {
  if (!media) return [];
  const notes = [...(media.notes ?? [])];
  for (const loi of media.errors ?? []) {
    notes.push(`Một nhánh truy xuất gặp lỗi (${loi}) — kết quả có thể thiếu.`);
  }
  return notes;
}
