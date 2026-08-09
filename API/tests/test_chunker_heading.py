"""Nhận diện tiêu đề mục khi chia chunk nội quy.

Lỗi thật, đo ngày 07/08/2026 trên `public/Nội Quy CLB 2022.docx`:
`Docx2txtLoader` không giữ dòng trống giữa các gạch đầu dòng, nên cả phần thân
về thành MỘT đoạn 1.461 ký tự không có xuống dòng. `HEADING_RE` chỉ khớp phần
ĐẦU dòng, mà đoạn đó mở đầu bằng "NỘI QUY SỬ DỤNG PHÒNG" — thế là cả đoạn thành
tên mục, rồi tên mục được ghép vào trước mọi chunk của mục đó.

Kết quả: 1.697 ký tự tài liệu nở thành **73 chunk / 107.220 ký tự**, gấp 63 lần.
Không có lỗi nào được ném ra — chỉ là 63 lần tiền nhúng và một collection đầy
chữ lặp, khiến mọi truy vấn trả về cùng một nội dung dưới nhiều dạng.
Bản PDF cùng nội dung ra 4 chunk / 2.076 ký tự.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.rag_noiquy.pipeline.chunker import StructureAwareChunker


@pytest.fixture()
def chunker() -> StructureAwareChunker:
    config = SimpleNamespace(
        pipeline=SimpleNamespace(chunk_size=1200, chunk_overlap=150, minimum_chunk_size=250)
    )
    return StructureAwareChunker(config=config)


THAN_BAI = (
    "Phải là thành viên chính thức của câu lạc bộ. "
    "Cần báo trước cho người quản lý phòng ít nhất 1 ngày để mượn chìa khóa. "
    "Không được tự ý dịch chuyển hoặc thay đổi bất kỳ thiết bị nào trong phòng. "
    "Nghiêm cấm hút thuốc và phải giữ gìn vệ sinh chung, ăn nói lễ phép, có văn hóa."
)


# ── Ngưỡng độ dài ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "dong",
    [
        "NỘI QUY SỬ DỤNG PHÒNG",
        "1. Đối với người sử dụng phòng",
        "3. Hình thức xử phạt",
        "Điều 5. Quyền và nghĩa vụ của thành viên",
        "Chương II",
    ],
)
def test_tieu_de_that_van_duoc_nhan(chunker, dong):
    """Siết quá tay còn tệ hơn: mất tiêu đề là mất luôn neo trích dẫn."""
    assert chunker._is_heading(dong) is True


def test_ca_doan_dai_mo_dau_bang_noi_quy_khong_phai_tieu_de(chunker):
    """Đúng chuỗi đã làm nổ 63 lần: mở đầu khớp mẫu, nhưng dài 1.400+ ký tự."""
    doan = "NỘI QUY SỬ DỤNG PHÒNG " + THAN_BAI * 5

    assert len(doan) > chunker.MAX_HEADING_CHARS
    assert chunker._is_heading(doan) is False


def test_nguong_cat_dung_o_max_heading_chars(chunker):
    vua_du = "Điều 1. " + "a" * (chunker.MAX_HEADING_CHARS - 8)
    qua_dai = vua_du + "a"

    assert chunker._is_heading(vua_du) is True
    assert chunker._is_heading(qua_dai) is False


def test_dong_dai_khong_khop_mau_van_khong_phai_tieu_de(chunker):
    assert chunker._is_heading("Các bạn nhớ giữ gìn vệ sinh chung nhé") is False


# ── Hệ quả ở tầng chia chunk ─────────────────────────────────────────────────


def tai_lieu(text: str) -> list[dict]:
    return [{"text": text, "page": 1, "source": "noi-quy.docx", "metadata": {"page": 1}}]


def test_khong_lam_phinh_van_ban_khi_ca_muc_nam_tren_mot_dong(chunker):
    """Bài kiểm tra thật sự: tổng chữ ra không được vượt xa chữ vào.

    Trước khi sửa, tên mục dài bằng cả mục và được ghép vào trước từng chunk,
    nên tổng đầu ra lớn gấp hàng chục lần đầu vào.
    """
    goc = "NỘI QUY SỬ DỤNG PHÒNG " + THAN_BAI * 5
    chunks = chunker.chunk_documents(tai_lieu(goc), document_id="nq", filename="noi-quy.docx")
    ra = sum(len(c["text"]) for c in chunks)

    # chunk_overlap=150 nên có dôi ra chút ít, nhưng không thể gấp đôi.
    assert ra < len(goc) * 2, f"{len(goc)} ký tự vào -> {ra} ký tự ra ({len(chunks)} chunk)"


def test_tieu_de_ngan_van_duoc_gan_lam_section(chunker):
    """Ngưỡng độ dài không được làm mất khả năng neo mục — thứ trích dẫn dựa vào."""
    goc = "NỘI QUY SỬ DỤNG PHÒNG\n\n1. Đối với người sử dụng phòng\n\n" + THAN_BAI

    chunks = chunker.chunk_documents(tai_lieu(goc), document_id="nq", filename="noi-quy.docx")

    assert any("Đối với người sử dụng phòng" in (c["section"] or "") for c in chunks)
