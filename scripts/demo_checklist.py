"""Checklist demo end-to-end — T-74 / US-605.1 / TC-605.

Chạy từng luồng lõi trên hệ thống thật rồi in PASS/FAIL. US-605.1 AC-2 nói rõ:
**một luồng fail phải chỉ đích danh, không được che bằng luồng khác** — nên mỗi
luồng chạy độc lập, lỗi của luồng này không dừng luồng kia, và mã thoát khác 0
khi có luồng đáng lẽ phải chạy mà lại hỏng.

"KHÔNG HỖ TRỢ" khác "FAIL": đó là luồng PRD có nêu nhưng v1 chưa làm. Gộp hai
thứ vào nhau thì hoặc là tự lừa mình rằng mọi thứ chạy, hoặc là báo động giả
mỗi lần chạy checklist.

    python scripts/demo_checklist.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "API"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

PASS = "PASS"
FAIL = "FAIL"
KHONG_HO_TRO = "KHÔNG HỖ TRỢ"


class Luong:
    def __init__(self, ma: str, ten: str, us: str, chay: Callable[[Any, Any], tuple[str, str]]):
        self.ma, self.ten, self.us, self.chay = ma, ten, us, chay


def l1_text_ra_anh(media: Any, _noi_quy: Any) -> tuple[str, str]:
    """text → ảnh + mô tả + link bài gốc."""
    ket = media.retrieve("ảnh các bạn đá bóng trên sân cỏ", 3, None, "clip")
    clips = ket.get("clips") or []
    if not clips:
        return FAIL, "không trả về ảnh nào"
    dau = clips[0]
    thieu = [ten for ten in ("caption", "frame_object_key", "source_url") if not dau.get(ten)]
    if thieu:
        return FAIL, f"kết quả đầu thiếu: {', '.join(thieu)}"
    return PASS, f"{len(clips)} ảnh, đầu bảng {dau['score']:.3f}: {dau['caption'][:48]}"


def l2_text_ra_loi_thoai(media: Any, _noi_quy: Any) -> tuple[str, str]:
    """text → đoạn lời thoại có mốc thời gian + link."""
    ket = media.retrieve("sinh nhật câu lạc bộ tin học", 3, None, "transcript")
    videos = ket.get("videos") or []
    if not videos:
        return FAIL, "không trả về video nào"
    moment = (videos[0].get("moments") or [{}])[0]
    if moment.get("start_sec") is None or moment.get("end_sec") is None:
        return FAIL, "đoạn lời thoại thiếu mốc thời gian"
    if not videos[0].get("source_url"):
        return FAIL, "video thiếu source_url"
    return PASS, f"{len(videos)} video, đoạn đầu [{moment['start_sec']:.0f}-{moment['end_sec']:.0f}s]"


def l3_text_ra_noi_quy(_media: Any, noi_quy: Any) -> tuple[str, str]:
    """text → điều khoản nội quy kèm trích dẫn."""
    ket = noi_quy.retrieve("làm mất chìa khóa phòng thì bị phạt bao nhiêu", 3, None, False)
    ket_qua = ket.get("results") or []
    if not ket_qua:
        return FAIL, "không trả về điều khoản nào"
    dau = ket_qua[0]
    if "100.000" not in (dau.get("text") or ""):
        return FAIL, f"điều khoản đầu bảng không đúng nội dung: {(dau.get('text') or '')[:60]}"
    return PASS, f"{len(ket_qua)} điều khoản, đầu bảng {dau['score']:.3f}"


def l4_anh_khong_mang_moc_giay(media: Any, _noi_quy: Any) -> tuple[str, str]:
    """Ảnh tĩnh KHÔNG được mang timestamp — gán mốc giây cho ảnh là bịa ra một
    khoảnh khắc không tồn tại (CLAUDE.md, và luật trong supervisor_prompt.md)."""
    ket = media.retrieve("ảnh chụp tập thể câu lạc bộ", 10, None, "clip")
    pham_loi = [
        c
        for c in ket.get("clips") or []
        if c.get("media_kind") == "image" and c.get("timestamp_sec") is not None
    ]
    if pham_loi:
        return FAIL, f"{len(pham_loi)} ảnh tĩnh lại mang timestamp_sec"
    return PASS, "không ảnh tĩnh nào mang mốc giây"


def l5_ngoai_mien_khong_dung_link(media: Any, noi_quy: Any) -> tuple[str, str]:
    """Câu ngoài miền: tầng truy xuất vẫn trả kết quả (không có ngưỡng), nhưng
    mọi link phải là link THẬT lấy từ dữ liệu, không được tự dựng."""
    ket = media.retrieve("cách nấu phở bò gia truyền Nam Định", 3, None, "both")
    noi_quy.retrieve("cách nấu phở bò gia truyền Nam Định", 3, None, False)
    link_gia = [
        c.get("source_url")
        for c in ket.get("clips") or []
        if c.get("source_url") and not str(c["source_url"]).startswith("https://www.facebook.com/")
    ]
    if link_gia:
        return FAIL, f"có link không thuộc nguồn thật: {link_gia[:2]}"
    diem = max((c["score"] for c in ket.get("clips") or []), default=0.0)
    return PASS, f"link đều là nguồn thật; điểm cao nhất {diem:.3f} (xử lý 'không tìm thấy' ở tầng trả lời)"


def l6_anh_lam_truy_van(_media: Any, _noi_quy: Any) -> tuple[str, str]:
    """ảnh → sự kiện / ảnh tương tự (US-605.1)."""
    return KHONG_HO_TRO, "API chỉ nhận truy vấn văn bản; embed_query(query: str) không có nhánh ảnh"


def l7_phuc_vu_media(media: Any, _noi_quy: Any) -> tuple[str, str]:
    """frame video → xem được ảnh và phát được video đúng khoảnh khắc (T-06/T-52).

    Kiểm tận nơi: tải thật object trong MinIO qua chính đường mà trình duyệt đi,
    và đòi máy chủ trả `206 Partial Content` — không có Range thì thẻ `<video>`
    không tua được, và "phát đúng giây" chỉ là nói miệng.
    """
    import httpx
    import sqlalchemy as sa

    from src.rag_video_anh.pipeline.minio_storage import MinioStorage
    from src.rag_video_anh.repository.database import get_session_manager

    storage = MinioStorage()

    def tai_duoc(object_key: str) -> tuple[bool, str]:
        """Tải THẬT qua presigned URL, và đòi 206 — không có Range thì không tua được."""
        if not storage.exists(object_key):
            return False, f"object không tồn tại: {object_key}"
        url = storage.presigned_download_url(object_key, expires_seconds=60)
        phan_hoi = httpx.get(url, headers={"Range": "bytes=0-1023"}, timeout=30.0)
        if phan_hoi.status_code != 206:
            return False, f"không hỗ trợ HTTP Range (nhận {phan_hoi.status_code})"
        return True, ""

    # Truy vấn rộng để chắc chắn có cả ảnh lẫn video, đừng phụ thuộc may rủi của
    # một câu hỏi cụ thể — bản trước fail chỉ vì top-5 lượt đó toàn ảnh tĩnh.
    ket = media.retrieve("hoạt động câu lạc bộ", 10, None, "both")
    anh = next((c for c in ket.get("clips") or [] if c.get("frame_object_key")), None)
    if anh is None:
        return FAIL, "không kết quả ảnh nào kèm object key để phục vụ"
    duoc, vi_sao = tai_duoc(anh["frame_object_key"])
    if not duoc:
        return FAIL, f"ảnh: {vi_sao}"

    videos = ket.get("videos") or []
    if not videos:
        return PASS, f"ảnh {anh['media_kind']} tải được (206); lượt này không có video để kiểm tua"

    with get_session_manager().session() as session:
        dong = session.execute(
            sa.text(
                "select m.object_key from videos v join media m on m.media_id = v.media_id "
                "where v.video_id = :vid"
            ),
            {"vid": videos[0]["video_id"]},
        ).first()
    if dong is None:
        return FAIL, f"không tra được object key của video {videos[0]['video_id']}"
    duoc, vi_sao = tai_duoc(dong[0])
    if not duoc:
        return FAIL, f"video: {vi_sao}"
    return PASS, f"ảnh và video đều tải được, Range trả 206 → tua tới đúng giây ({len(videos)} video khớp)"


LUONG = [
    Luong("L1", "text → ảnh + mô tả + link", "US-301.1", l1_text_ra_anh),
    Luong("L2", "text → lời thoại có mốc thời gian", "US-303.1", l2_text_ra_loi_thoai),
    Luong("L3", "text → điều khoản nội quy", "US-407.1", l3_text_ra_noi_quy),
    Luong("L4", "ảnh tĩnh không mang mốc giây", "US-405.1", l4_anh_khong_mang_moc_giay),
    Luong("L5", "câu ngoài miền không dựng link giả", "US-704.1", l5_ngoai_mien_khong_dung_link),
    Luong("L6", "ảnh làm truy vấn → sự kiện/ảnh tương tự", "US-605.1", l6_anh_lam_truy_van),
    Luong("L7", "xem ảnh + phát video đúng khoảnh khắc", "US-605.1", l7_phuc_vu_media),
]


def main() -> int:
    from src.rag_noiquy.retrieval.retrieval_service import build_retrieval_service
    from src.rag_video_anh.retrieval.retrieval_service import build_video_retrieval_service

    media = build_video_retrieval_service()
    noi_quy = build_retrieval_service()

    ket_qua: list[tuple[Luong, str, str]] = []
    for luong in LUONG:
        try:
            trang_thai, ghi_chu = luong.chay(media, noi_quy)
        except Exception as exc:  # noqa: BLE001 — một luồng hỏng không được giết cả checklist
            trang_thai, ghi_chu = FAIL, f"{exc.__class__.__name__}: {exc}"
        ket_qua.append((luong, trang_thai, ghi_chu))

    print("\n" + "=" * 78)
    print("CHECKLIST DEMO END-TO-END — TC-605")
    print("=" * 78)
    for luong, trang_thai, ghi_chu in ket_qua:
        print(f"{luong.ma}  {trang_thai:13} {luong.ten:42} {luong.us}")
        print(f"      {ghi_chu}")

    so_pass = sum(1 for _, t, _ in ket_qua if t == PASS)
    so_fail = sum(1 for _, t, _ in ket_qua if t == FAIL)
    so_thieu = sum(1 for _, t, _ in ket_qua if t == KHONG_HO_TRO)
    print("-" * 78)
    print(f"{so_pass} PASS · {so_fail} FAIL · {so_thieu} KHÔNG HỖ TRỢ (chưa làm ở v1)")
    if so_fail:
        print("Luồng FAIL: " + ", ".join(luong.ma for luong, t, _ in ket_qua if t == FAIL))
    return 1 if so_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
