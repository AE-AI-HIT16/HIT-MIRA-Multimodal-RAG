"""Mốc thời gian của bài phải đi hết đường từ PostgreSQL tới kết quả truy hồi.

Không có nó thì câu hỏi "ảnh sự kiện năm 2024" không lọc được gì: `post_id` là
UUID vô nghĩa, còn `media.created_at` chỉ là lúc chạy script đăng ký — mọi ảnh
đều mang cùng một ngày, không liên quan gì tới lúc sự việc xảy ra.

Dữ liệu vốn có sẵn (`posts.created_time`, 496/496 bài, trải 2021→2026) và
`ImageUnit` vốn đã mang `post_created_at`. Đường ống đứt ở **hai** chỗ nối tiếp
nhau, nên sửa một chỗ thì test vẫn đỏ:

1. `ImageUnit.to_dict()` không xuất hai khoá này ra dict.
2. `_image_payload()` không đọc chúng để ghi vào payload Qdrant.

`event_key` đi chung đường ống nên ghim luôn ở đây; hiện nó là `None` với mọi
bài vì bốn bảng sự kiện còn rỗng, nhưng cái phải giữ đúng là *đường đi*.
"""

from __future__ import annotations

from src.rag_video_anh.retrieval.indexing_service import VideoRetrievalIndexingService
from src.rag_video_anh.retrieval.retrieval_units import (
    ImageUnit,
    MediaClipUnit,
    TranscriptContext,
    VideoTranscriptUnit,
)
from src.rag_video_anh.retrieval.retriever import MediaClipHit

MOC = "2024-11-08T03:15:00Z"

# Các trường nội dung không liên quan tới phép kiểm ở đây, gom lại cho gọn.
NOI_DUNG_ANH = {
    "caption": "c",
    "ocr_text": "",
    "vision_metadata": {},
    "detected_objects": [],
    "object_counts": {},
}


def test_image_unit_xuat_moc_thoi_gian_ra_dict() -> None:
    unit = ImageUnit(
        unit_id="image:m1",
        image_media_id="m1",
        post_id="p1",
        bucket_name="b",
        object_key="k.jpg",
        post_created_at=MOC,
        event_key="hit-open-day-2025",
        **NOI_DUNG_ANH,
    )

    d = unit.to_dict()

    assert d["post_created_at"] == MOC
    assert d["event_key"] == "hit-open-day-2025"


def test_payload_anh_giu_moc_thoi_gian() -> None:
    payload = VideoRetrievalIndexingService._image_payload(
        {"unit_id": "image:m1", "post_id": "p1", "post_created_at": MOC, "event_key": None}
    )

    assert payload["post_created_at"] == MOC
    # Ghi cả khi rỗng: thiếu hẳn khoá thì Qdrant không lọc `event_key IS NULL`
    # được, mà đó chính là cách tìm những bài chưa gắn sự kiện.
    assert "event_key" in payload


def test_payload_keyframe_va_transcript_cung_giu() -> None:
    """Ba nguồn phải mang cùng bộ khoá, nếu không một bộ lọc năm sẽ âm thầm
    bỏ sót video trong khi vẫn trả về đủ ảnh — kiểu sai khó nhận ra nhất."""

    clip = VideoRetrievalIndexingService._media_clip_payload(
        {"unit_id": "media_clip:f1", "post_created_at": MOC, "event_key": "e"}
    )
    transcript = VideoRetrievalIndexingService._video_transcript_payload(
        {"unit_id": "transcript:t1", "post_created_at": MOC, "event_key": "e"}
    )

    assert clip["post_created_at"] == transcript["post_created_at"] == MOC
    assert clip["event_key"] == transcript["event_key"] == "e"


def test_don_vi_video_xuat_du_hai_khoa() -> None:
    clip = MediaClipUnit(
        unit_id="media_clip:f1",
        video_id="v1",
        video_media_id="vm1",
        post_id="p1",
        frame_media_id="f1",
        frame_index=0,
        timestamp_sec=1.0,
        bucket_name="b",
        frame_object_key="k.jpg",
        post_created_at=MOC,
        transcript_context=TranscriptContext(text="", source_segment_ids=[], segments=[]),
        **NOI_DUNG_ANH,
    ).to_dict()
    transcript = VideoTranscriptUnit(
        unit_id="transcript:t1",
        video_id="v1",
        post_id="p1",
        start_sec=0.0,
        end_sec=1.0,
        text="xin chào",
        language="vi",
        source_segment_ids=[],
        post_created_at=MOC,
    ).to_dict()

    assert clip["post_created_at"] == transcript["post_created_at"] == MOC
    assert "event_key" in clip and "event_key" in transcript


def test_ket_qua_truy_hoi_tra_moc_thoi_gian_ra_ngoai() -> None:
    """Ghi được vào payload mà không đọc ra thì tầng trả lời vẫn không dùng được."""

    hit = MediaClipHit(
        score=0.5,
        media_kind="image",
        video_id=None,
        unit_id="image:m1",
        timestamp_sec=None,
        caption="c",
        ocr_text="",
        bucket_name="b",
        frame_object_key="k.jpg",
        payload={"post_created_at": MOC, "event_key": "hit-open-day-2025"},
    )

    d = hit.as_dict()

    assert d["post_created_at"] == MOC
    assert d["event_key"] == "hit-open-day-2025"


def test_diem_index_truoc_khi_co_khoa_nay_khong_lam_vo() -> None:
    """1.628 điểm đã index từ trước không có hai khoá — phải ra None, không nổ."""

    hit = MediaClipHit(
        score=0.5,
        media_kind="image",
        video_id=None,
        unit_id="image:cu",
        timestamp_sec=None,
        caption="c",
        ocr_text="",
        bucket_name="b",
        frame_object_key="k.jpg",
        payload={"post_id": "p1"},
    )

    assert hit.as_dict()["post_created_at"] is None
    assert hit.as_dict()["event_key"] is None
