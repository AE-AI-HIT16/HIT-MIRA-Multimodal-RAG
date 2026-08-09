"""Đơn vị nhúng văn bản cho nhánh media — TC-905.

Nhánh media chuyển sang chỉ nhúng caption/OCR/transcript. Ba tính chất phải giữ:
mỗi point một loại nội dung, OCR không mất bố cục, và không đoạn con nào được
mang mốc thời gian mà ASR chưa hề cung cấp.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.rag_video_anh.retrieval import text_units as tu


def anh(caption="", ocr=""):
    return SimpleNamespace(
        image_media_id="img-1", post_id="post-1", bucket_name="b", object_key="k.jpg",
        caption=caption, ocr_text=ocr, source_url="https://fb/1",
    )


def keyframe(caption="", ocr=""):
    return SimpleNamespace(
        video_id="vid-1", video_media_id="vm-1", frame_media_id="fr-1", frame_index=3,
        timestamp_sec=12.5, post_id="post-1", bucket_name="b", frame_object_key="f.jpg",
        caption=caption, ocr_text=ocr, source_url=None,
    )


def doan(text, start=0.0, end=60.0, ids=("s1",), video="vid-1"):
    return SimpleNamespace(
        video_id=video, post_id="post-1", start_sec=start, end_sec=end,
        text=text, language="vi", source_segment_ids=list(ids), source_url=None,
    )


# ---------------------------------------------------------------- chuẩn hoá


def test_ocr_giu_ranh_gioi_dong() -> None:
    """`clean_text` gộp mọi khoảng trắng, làm dính hai dòng thành một câu vô nghĩa."""
    tho = "HIT CLB TIN HỌC\n12h30 - 20/04/2024\n\n  Lửa   trại  "
    assert tu.chuan_hoa_ocr(tho) == "HIT CLB TIN HỌC\n12h30 - 20/04/2024\nLửa trại"


def test_ocr_khong_ha_chu_thuong_khong_bo_dau() -> None:
    """Mã, số hiệu và tên riêng là thứ duy nhất OCR có mà caption không có."""
    assert tu.chuan_hoa_ocr("GEN 15 — Bản Lác") == "GEN 15 — Bản Lác"


def test_caption_gop_khoang_trang() -> None:
    assert tu.chuan_hoa_caption("Một   nhóm\nbạn trẻ  ") == "Một nhóm bạn trẻ"


def test_text_rong_hoac_khong_phai_chuoi_thi_khong_sinh_point() -> None:
    assert tu.units_tu_anh(anh(caption="   ", ocr=None)) == []


# ---------------------------------------------------------------- một point một loại


def test_caption_va_ocr_thanh_hai_point_rieng() -> None:
    """Nối chung thì không biết hit đến từ chữ trên ảnh hay từ mô tả."""
    units = tu.units_tu_anh(anh(caption="Nhóm bạn chụp ảnh", ocr="HIT"))

    assert [u.source_type for u in units] == [tu.SOURCE_CAPTION, tu.SOURCE_OCR]
    assert [u.text for u in units] == ["Nhóm bạn chụp ảnh", "HIT"]
    assert all("Nhóm bạn" not in u.text for u in units if u.source_type == tu.SOURCE_OCR)


def test_text_dem_nhung_khong_co_tien_to_mo_ta() -> None:
    """Tiền tố có mặt ở mọi point nên không phân biệt gì, chỉ tốn token."""
    u = tu.units_tu_anh(anh(caption="Ảnh sự kiện"))[0]
    assert u.text == "Ảnh sự kiện"


def test_ocr_rat_ngan_van_duoc_giu() -> None:
    """'HIT', 'GEN 15' đúng là thứ người ta tìm — bỏ vì ngắn là bỏ đúng cái cần."""
    units = tu.units_tu_anh(anh(ocr="HIT"))
    assert [u.text for u in units] == ["HIT"]


# ---------------------------------------------------------------- payload


def test_anh_khong_mang_timestamp_hay_video_id() -> None:
    """Ảnh tĩnh không nằm trên trục thời gian nào; bịa mốc là tạo trích dẫn sai."""
    u = tu.units_tu_anh(anh(caption="x"))[0]
    assert "timestamp_sec" not in u.payload
    assert "video_id" not in u.payload
    assert u.payload["media_kind"] == "image"


def test_keyframe_mang_du_moc_de_trich_dan() -> None:
    u = tu.units_tu_keyframe(keyframe(caption="x"))[0]
    assert u.payload["video_id"] == "vid-1"
    assert u.payload["timestamp_sec"] == 12.5
    assert u.payload["object_key"] == "f.jpg"   # vẫn hiển thị được ảnh dù không nhúng ảnh


def test_unit_id_on_dinh_de_index_lai_ghi_de() -> None:
    a, b = tu.units_tu_anh(anh(caption="x"))[0], tu.units_tu_anh(anh(caption="x"))[0]
    assert a.unit_id == b.unit_id == "caption:image:img-1"


def test_content_hash_theo_van_ban_da_chuan_hoa() -> None:
    """Caption trùng nhau giữa nhiều frame chỉ phải gọi API nhúng một lần."""
    x = tu.units_tu_anh(anh(caption="Cùng một caption"))[0]
    y = tu.units_tu_keyframe(keyframe(caption="Cùng  một   caption"))[0]
    assert x.content_hash == y.content_hash
    assert x.unit_id != y.unit_id      # cùng vector, khác point và khác metadata


def test_to_dict_mang_format_version() -> None:
    d = tu.units_tu_anh(anh(caption="x"))[0].to_dict()
    assert d["embedding_format_version"] == tu.EMBEDDING_FORMAT_VERSION
    assert d["source_type"] == tu.SOURCE_CAPTION


# ---------------------------------------------------------------- cắt đoạn


def test_ocr_dai_bi_cat_va_danh_so() -> None:
    dai = "\n".join(f"dòng số {i} với một ít chữ tiếng Việt để tốn token" for i in range(200))
    units = tu.units_tu_anh(anh(ocr=dai))

    assert len(units) > 1
    assert all(u.source_type == tu.SOURCE_OCR for u in units)
    assert [u.payload["ocr_chunk_index"] for u in units] == list(range(len(units)))
    assert all(tu.dem_token(u.text) <= tu.OCR_MAX_TOKENS + tu.OCR_OVERLAP_TOKENS for u in units)


def test_ocr_ngan_khong_bi_danh_so_thua() -> None:
    u = tu.units_tu_anh(anh(ocr="HIT\nGEN 15"))[0]
    assert u.unit_id == "ocr:image:img-1"      # không có hậu tố #0
    assert u.payload["ocr_chunk_total"] == 1


# ---------------------------------------------------------------- transcript


def test_doan_ngan_duoc_gop_va_hop_khoang_thoi_gian() -> None:
    units = tu.units_tu_transcript([
        doan("Chào các bạn", 0.0, 5.0, ["s1"]),
        doan(" ".join(["nội dung buổi nói chuyện hôm nay"] * 12), 5.0, 65.0, ["s2"]),
    ])

    assert len(units) == 1
    assert units[0].payload["start_sec"] == 0.0
    assert units[0].payload["end_sec"] == 65.0
    assert units[0].payload["source_segment_ids"] == ["s1", "s2"]
    assert units[0].text.startswith("Chào các bạn")


def test_khong_gop_qua_hai_video_khac_nhau() -> None:
    units = tu.units_tu_transcript([
        doan("ngắn quá", 0.0, 3.0, ["s1"], video="vid-1"),
        doan(" ".join(["câu nói dài của video khác"] * 12), 0.0, 60.0, ["s2"], video="vid-2"),
    ])
    theo_video = {u.payload["video_id"] for u in units}
    assert theo_video == {"vid-1", "vid-2"}


def test_doan_qua_dai_bi_cat_nhung_giu_moc_cua_doan_cha() -> None:
    """ASR không có mốc theo từ, nên nội suy mốc cho đoạn con là bịa."""
    dai = " ".join(["một câu nói rất dài lặp đi lặp lại nhiều lần"] * 90)
    units = tu.units_tu_transcript([doan(dai, 10.0, 70.0, ["s1"])])

    assert len(units) > 1
    assert {u.payload["start_sec"] for u in units} == {10.0}
    assert {u.payload["end_sec"] for u in units} == {70.0}
    assert [u.payload["chunk_index"] for u in units] == list(range(len(units)))


def test_doan_vua_tam_giu_nguyen() -> None:
    vua = " ".join(["nội dung buổi sinh hoạt câu lạc bộ"] * 12)
    units = tu.units_tu_transcript([doan(vua, 0.0, 60.0, ["s1"])])

    assert len(units) == 1
    assert units[0].payload["chunk_total"] == 1
    assert units[0].payload["media_kind"] == "transcript"


@pytest.mark.parametrize("ham", [tu.units_tu_anh, tu.units_tu_keyframe])
def test_khong_co_noi_dung_thi_khong_sinh_point(ham) -> None:
    """Ảnh không có cả caption lẫn OCR thì bỏ qua, không tạo point rỗng."""
    doi_tuong = anh() if ham is tu.units_tu_anh else keyframe()
    assert ham(doi_tuong) == []


# ---------------------------------------------------------------- OCR end-to-end


def test_clean_ocr_layout_giu_xuong_dong_con_clean_ocr_thi_khong() -> None:
    """Đây là lỗi thật: builder gọi clean_text nên OCR tới text_units đã phẳng."""
    from uuid import uuid4

    from src.rag_video_anh.repository import OcrResultRecord
    from src.rag_video_anh.retrieval.retrieval_units import clean_ocr, clean_ocr_layout

    ban_ghi = OcrResultRecord(
        media_id=uuid4(), ocr_status="DONE",
        ocr_text="HIT CLB TIN HỌC\n12h30 - 20/04/2024\nLửa trại",
    )
    assert clean_ocr(ban_ghi) == "HIT CLB TIN HỌC 12h30 - 20/04/2024 Lửa trại"
    assert clean_ocr_layout(ban_ghi) == "HIT CLB TIN HỌC\n12h30 - 20/04/2024\nLửa trại"


def test_unit_uu_tien_ban_ocr_con_bo_cuc() -> None:
    doi_tuong = anh(ocr="đã bị làm phẳng hết rồi")
    doi_tuong.ocr_text_layout = "HIT\n12h30 - 20/04/2024"
    assert [u.text for u in tu.units_tu_anh(doi_tuong)] == ["HIT\n12h30 - 20/04/2024"]


def test_thieu_ban_bo_cuc_thi_van_dung_ocr_text() -> None:
    """Mất bố cục còn hơn mất chữ."""
    assert [u.text for u in tu.units_tu_anh(anh(ocr="HIT GEN 15"))] == ["HIT GEN 15"]


def test_ocr_layout_van_loc_van_ban_trong_nhu_thong_bao_loi() -> None:
    """Một nhánh không được nhận vào thứ mà nhánh kia đã loại."""
    from uuid import uuid4

    from src.rag_video_anh.repository import OcrResultRecord
    from src.rag_video_anh.retrieval.retrieval_units import clean_ocr_layout

    assert clean_ocr_layout(OcrResultRecord(media_id=uuid4(), ocr_status="DONE", ocr_text="không có chữ")) == ""
    assert clean_ocr_layout(OcrResultRecord(media_id=uuid4(), ocr_status="FAILED", ocr_text="HIT")) == ""


def test_end_to_end_ocr_giu_xuong_dong_tu_db_toi_text_don_vi() -> None:
    """Nối cả chuỗi: bản ghi OCR -> builder -> ImageUnit -> TextRetrievalUnit.

    Hai test trước kiểm hai nửa riêng. Nửa nào cũng đúng mà chỗ nối sai thì
    không test nào bắt được — mà chỗ nối chính là nơi lỗi đã xảy ra.
    """
    from types import SimpleNamespace
    from uuid import uuid4

    from src.rag_video_anh.repository import CaptionResultRecord, MediaType, OcrResultRecord
    from src.rag_video_anh.retrieval.retrieval_units import ImageRetrievalUnitBuilder

    media_id = uuid4()
    post_id = uuid4()
    tho = "HIT CLB TIN HỌC ĐH CNHN\nTimeLine\n12h30 - 20/04/2024 Tập trung tại trường"

    class FakeUow:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        media = SimpleNamespace(get_media=lambda _: SimpleNamespace(
            media_id=media_id, post_id=post_id, media_type=MediaType.IMAGE.value,
            bucket_name="hit-mira-media", object_key="raw/x/photo_01.jpg",
        ))
        results = SimpleNamespace(
            get_caption_result=lambda _: CaptionResultRecord(
                media_id=media_id, caption_status="DONE", caption_text="Một tấm poster sự kiện"),
            get_ocr_result=lambda _: OcrResultRecord(
                media_id=media_id, ocr_status="DONE", ocr_text=tho),
            get_object_result=lambda _: None,
        )
        posts = SimpleNamespace(get_post=lambda _: None)

    unit = ImageRetrievalUnitBuilder(uow_factory=FakeUow, check_image_exists=False).build(media_id)

    # Payload cũ vẫn phẳng — hợp đồng của collection Jina không đổi.
    assert "\n" not in unit.ocr_text
    # Text đem nhúng giữ nguyên bố cục, NHƯNG dòng watermark đầu bị lọc: nó có
    # trên mọi ảnh nên không phân biệt được gì.
    van_ban = {u.source_type: u.text for u in tu.units_tu_anh(unit)}
    assert van_ban["ocr"] == "TimeLine\n12h30 - 20/04/2024 Tập trung tại trường"
    assert van_ban["caption"] == "Một tấm poster sự kiện"
    # Nhưng OCR THÔ trong PostgreSQL không hề bị đụng tới.
    assert "HIT CLB TIN HỌC ĐH CNHN" in unit.ocr_text


# ---------------------------------------------------------------- metadata cấp bài


def anh_co_meta(caption="x", ocr="", **meta):
    u = anh(caption=caption, ocr=ocr)
    for k, v in {"post_created_at": "2024-10-28T08:00:00Z", "event_key": "team-building", **meta}.items():
        setattr(u, k, v)
    return u


def test_moi_point_cua_cung_anh_mang_cung_metadata_bai() -> None:
    """Caption và OCR của một ảnh phải lọc được bằng cùng một điều kiện."""
    units = tu.units_tu_anh(anh_co_meta(caption="mô tả", ocr="HIT\nGEN 15"))

    assert len(units) == 2
    for u in units:
        assert u.payload["post_created_at"] == "2024-10-28T08:00:00Z"
        assert u.payload["event_key"] == "team-building"


def test_moi_manh_ocr_cua_cung_anh_cung_mang_metadata() -> None:
    dai = "\n".join(f"dòng {i} với khá nhiều chữ tiếng Việt để tốn token" for i in range(200))
    units = tu.units_tu_anh(anh_co_meta(caption="", ocr=dai))

    assert len(units) > 1
    assert {u.payload["event_key"] for u in units} == {"team-building"}


def test_keyframe_cung_mang_metadata_bai() -> None:
    kf = keyframe(caption="cảnh mở đầu", ocr="HIT")
    kf.post_created_at, kf.event_key = "2024-10-28T08:00:00Z", "team-building"
    units = tu.units_tu_keyframe(kf)

    assert {u.payload["event_key"] for u in units} == {"team-building"}


def test_transcript_cung_mang_metadata_bai() -> None:
    """Thiếu ở transcript thì filter event tìm được caption/OCR mà loại sạch lời thoại."""
    d = doan(" ".join(["nội dung buổi nói chuyện"] * 15))
    d.post_created_at, d.event_key = "2024-10-28T08:00:00Z", "team-building"
    units = tu.units_tu_transcript([d])

    assert units[0].payload["event_key"] == "team-building"
    assert units[0].payload["post_created_at"] == "2024-10-28T08:00:00Z"


def test_bai_khong_nhan_dien_duoc_su_kien_thi_khong_co_khoa() -> None:
    """Ghi 'unknown' sẽ gom mọi bài chưa nhận diện được vào cùng một nhóm giả."""
    units = tu.units_tu_anh(anh_co_meta(event_key=None, post_created_at=None))

    assert "event_key" not in units[0].payload
    assert "post_created_at" not in units[0].payload


def test_metadata_lay_tu_post_index_metadata_mot_lan_moi_bai() -> None:
    """Đọc lại theo từng point thì một video 200 keyframe là 200 lượt truy vấn posts."""
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from src.rag_video_anh.retrieval.retrieval_units import _post_index_metadata

    class Repo:
        def __init__(self): self.so_lan = 0
        def get(self, post_id):
            self.so_lan += 1
            return SimpleNamespace(
                post_url="https://fb/1",
                created_time=datetime(2024, 10, 28, 8, 0, tzinfo=timezone.utc),
            )

    repo = Repo()
    meta = _post_index_metadata(SimpleNamespace(posts=repo), "post-1")

    assert meta.post_created_at == "2024-10-28T08:00:00Z"
    assert meta.event_key is None          # bảng sự kiện rỗng, chưa có repo events
    assert repo.so_lan == 1


@pytest.mark.parametrize(
    ("kieu", "mong_doi"),
    [
        # Đúng thứ PostgreSQL trả về: `parse_created_time()` đã gỡ timezone.
        ("naive", "2024-04-11T12:59:01Z"),
        ("utc", "2024-04-11T12:59:01Z"),
        ("mui_khac", "2024-04-11T12:59:01Z"),
    ],
)
def test_rfc3339_luon_co_mui_gio(kieu: str, mong_doi: str) -> None:
    """Thiếu `Z` hoặc offset là không đúng RFC3339; Qdrant có quyền hiểu theo múi khác."""
    from datetime import datetime, timedelta, timezone

    from src.rag_video_anh.retrieval.retrieval_units import _rfc3339

    gia_tri = {
        "naive": datetime(2024, 4, 11, 12, 59, 1),
        "utc": datetime(2024, 4, 11, 12, 59, 1, tzinfo=timezone.utc),
        "mui_khac": datetime(2024, 4, 11, 19, 59, 1, tzinfo=timezone(timedelta(hours=7))),
    }[kieu]
    assert _rfc3339(gia_tri) == mong_doi


def test_rfc3339_bo_qua_gia_tri_khong_phai_thoi_gian() -> None:
    from src.rag_video_anh.retrieval.retrieval_units import _rfc3339

    assert _rfc3339(None) is None
    assert _rfc3339(12345) is None
    assert _rfc3339("   ") is None


# ---------------------------------------------------------------- lọc watermark/hoa văn


def test_timeline_ngay_gio_duoc_giu_nguyen() -> None:
    """Đây là loại dữ liệu quý nhất OCR mang lại; lọc nhầm nó là mất trắng."""
    tho = ("Timeline Bản lác-Mai Châu\n"
           "12h30 - 20/04/2024 Tập trung tại trường\n"
           "20h00 - 20/04/2024 Lửa trại văn nghệ")
    assert tu.chuan_hoa_ocr(tho) == tho


def test_hoa_van_nhi_phan_bi_loai() -> None:
    assert tu.chuan_hoa_ocr("SPORT 0101010101010101") == "SPORT"
    assert tu.chuan_hoa_ocr("SPORT 0101 1010 0011 0101") == "SPORT"


def test_ma_lop_1010_khong_bi_xoa() -> None:
    """Dãy 0/1 ngắn, đứng lẻ giữa chữ, là dữ liệu chứ không phải hoa văn."""
    assert tu.chuan_hoa_ocr("Mã lớp 1010") == "Mã lớp 1010"
    assert tu.chuan_hoa_ocr("Phòng 101 lớp 1010") == "Phòng 101 lớp 1010"


def test_day_ngan_nhung_day_dac_van_bi_loai() -> None:
    """Luật thứ hai: phần lớn token là 0/1 thì chúng là hoa văn."""
    ra = tu.chuan_hoa_ocr("HIT CLUB BEE GEE 1 0 0 1 0 01 0 1")
    assert "0" not in ra and "1" not in ra
    assert "BEE GEE" in ra


def test_chi_co_watermark_thi_khong_sinh_point() -> None:
    """Watermark có trên MỌI ảnh nên giá trị phân biệt gần bằng không."""
    for tho in ("HIT CLUB", "HITCLUB", "CLB TIN HỌC ĐH CNHN",
                "HITclub HITclub HITclub", "HTC HITC", "HIT CLUB\nHITCLUB"):
        assert tu.chuan_hoa_ocr(tho) == "", tho
    assert tu.units_tu_anh(anh(ocr="HITclub HITclub 010100101 010110 11010")) == []


def test_watermark_kem_noi_dung_thi_giu_noi_dung() -> None:
    tho = "HIT CLUB\nTEAM BUILDING 2024\n30/10 - CÔNG VIÊN HÒA BÌNH"
    assert tu.chuan_hoa_ocr(tho) == "TEAM BUILDING 2024\n30/10 - CÔNG VIÊN HÒA BÌNH"


def test_khong_thay_the_toan_cuc_htc_thanh_hit() -> None:
    """'HTC' có thể là nội dung thật ở ảnh khác; chỉ bỏ khi nó là cả dòng."""
    assert tu.chuan_hoa_ocr("Điện thoại HTC One M8") == "Điện thoại HTC One M8"


def test_ocr_rong_khong_sinh_point() -> None:
    assert tu.chuan_hoa_ocr("") == ""
    assert tu.units_tu_anh(anh(caption="chỉ có caption", ocr="")) == [
        tu.units_tu_anh(anh(caption="chỉ có caption", ocr=""))[0]
    ]
    units = tu.units_tu_anh(anh(caption="chỉ có caption", ocr="HIT CLUB"))
    assert [u.source_type for u in units] == [tu.SOURCE_CAPTION]
