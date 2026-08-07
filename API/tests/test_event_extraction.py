"""Cửa lọc của `scripts/extract_post_events.py` — chỗ chặn LLM bịa sự kiện.

Script này để LLM đọc 496 bài rồi ghi thẳng vào DB, nên nó phải giả định LLM sẽ
sai. Hai cửa chặn được ghim ở đây:

1. **Trích dẫn phải có thật trong bài.** Đây là bằng chứng duy nhất cho thấy LLM
   đọc bài chứ không nhớ ra một sự kiện quen thuộc. Một sự kiện bịa sẽ kéo cả
   chùm ảnh của bài đó vào một sự kiện chưa từng xảy ra.
2. **Lượt gom không được đánh rơi tên.** LLM bỏ sót vài tên trong danh sách dài
   là chuyện thường; im lặng đánh rơi bài mới là vấn đề, nên phần thiếu được vá
   lại thành chuỗi riêng chứ không biến mất.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "extract_post_events", REPO_ROOT / "scripts/extract_post_events.py"
)
extract_events = importlib.util.module_from_spec(_spec)
sys.modules["extract_post_events"] = extract_events
_spec.loader.exec_module(extract_events)

BAI = (
    "🔥 [HIT CONTEST SERIES 2021 SEASON 2 - SỰ TRỞ LẠI BÙNG NỔ] 🔥\n\n"
    "Vậy là sự kiện HIT CONTEST SERIES 2021 đã quay trở lại với anh em nhà HIT."
)


def ban_ghi(**ghi_de):
    goc = {
        "co_su_kien": True,
        "ten_su_kien": "HIT Contest Series",
        "nhan_ky": "2021 Season 2",
        "nam": 2021,
        "do_tin_cay": 0.95,
        "trich_dan": "sự kiện HIT CONTEST SERIES 2021 đã quay trở lại",
    }
    goc.update(ghi_de)
    return goc


# ── Cửa trích dẫn ────────────────────────────────────────────────────────────


def test_trich_dan_co_that_thi_nhan():
    assert extract_events.hop_le(ban_ghi(), BAI, 0.6) == (True, "")


def test_trich_dan_bia_thi_loai():
    """Nghe rất hợp lý, và không có một chữ nào như vậy trong bài."""
    gia = ban_ghi(trich_dan="đêm gala kỷ niệm 15 năm thành lập CLB")

    assert extract_events.hop_le(gia, BAI, 0.6) == (False, "trich_dan_khong_co_trong_bai")


def test_bo_qua_emoji_va_xuong_dong_khi_so_khop():
    """LLM chép đúng chữ nhưng bỏ emoji hoặc gộp dòng — vẫn phải tính là có thật,
    nếu không thì cửa này loại sạch cả những kết quả đúng."""
    van_ban = ban_ghi(trich_dan="HIT CONTEST SERIES 2021 SEASON 2 - SỰ TRỞ LẠI BÙNG NỔ")

    assert extract_events.hop_le(van_ban, BAI, 0.6)[0] is True


def test_thieu_trich_dan_thi_loai():
    assert extract_events.hop_le(ban_ghi(trich_dan=""), BAI, 0.6) == (False, "thieu_trich_dan")


# ── Các cửa còn lại ──────────────────────────────────────────────────────────


def test_bai_khong_co_su_kien_bi_loai_dung_ly_do():
    assert extract_events.hop_le({"co_su_kien": False}, BAI, 0.6) == (False, "khong_co_su_kien")


def test_do_tin_cay_thap_thi_loai():
    assert extract_events.hop_le(ban_ghi(do_tin_cay=0.4), BAI, 0.6) == (False, "do_tin_cay_thap")


def test_do_tin_cay_khong_doc_duoc_thi_loai_chu_khong_no():
    assert extract_events.hop_le(ban_ghi(do_tin_cay="cao"), BAI, 0.6)[0] is False


def test_thieu_ten_thi_loai():
    assert extract_events.hop_le(ban_ghi(ten_su_kien="  "), BAI, 0.6) == (False, "thieu_ten")


def test_ten_toan_emoji_thi_loai():
    """`slugify` ra chuỗi rỗng thì `event_key` sẽ rỗng — không ghi vào payload được."""
    assert extract_events.hop_le(ban_ghi(ten_su_kien="🔥🔥🔥"), BAI, 0.6) == (
        False,
        "ten_khong_tao_duoc_slug",
    )


# ── Đọc JSON do LLM trả về ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw",
    [
        '{"co_su_kien": true}',
        '```json\n{"co_su_kien": true}\n```',
        '```\n{"co_su_kien": true}\n```',
        '  {"co_su_kien": true}  ',
    ],
)
def test_go_rao_markdown_quanh_json(raw):
    assert extract_events.doc_json(raw) == {"co_su_kien": True}


# ── Lượt gom ─────────────────────────────────────────────────────────────────


class LlmGia:
    def __init__(self, tra_ve: str) -> None:
        self.tra_ve = tra_ve

    def invoke(self, _prompt):
        return type("Msg", (), {"content": self.tra_ve})()


def dem(*ten):
    from collections import Counter

    return Counter(ten)


def test_gom_cac_cach_viet_ve_mot_chuoi():
    llm = LlmGia('[{"ten_chuan": "HIT Contest Series", "cac_ten_goc": ["HIT Contest Series", "Contest Series"]}]')

    cum = extract_events.gom_ten(llm, dem("HIT Contest Series", "Contest Series"))

    assert cum == [{"ten_chuan": "HIT Contest Series", "cac_ten_goc": ["HIT Contest Series", "Contest Series"]}]


def test_ten_bi_llm_bo_sot_van_thanh_mot_chuoi_rieng():
    """Bỏ sót mà im lặng thì những bài mang tên đó biến mất khỏi kết quả."""
    llm = LlmGia('[{"ten_chuan": "Lớp hè", "cac_ten_goc": ["Lớp hè"]}]')

    cum = extract_events.gom_ten(llm, dem("Lớp hè", "Code War"))

    assert {n["ten_chuan"] for n in cum} == {"Lớp hè", "Code War"}


def test_ten_llm_tu_nghi_ra_bi_bo():
    """Chỉ những tên thực sự trích được mới thành sự kiện."""
    llm = LlmGia('[{"ten_chuan": "Gala", "cac_ten_goc": ["Gala cuối năm", "Lớp hè"]}]')

    cum = extract_events.gom_ten(llm, dem("Lớp hè"))

    assert cum == [{"ten_chuan": "Gala", "cac_ten_goc": ["Lớp hè"]}]


def test_mot_ten_khong_the_nam_o_hai_nhom():
    """Trùng nhóm thì `ten_goc_toi_chuan` sẽ chọn nhóm cuối một cách tuỳ tiện."""
    llm = LlmGia(
        '[{"ten_chuan": "A", "cac_ten_goc": ["Lớp hè"]},'
        ' {"ten_chuan": "B", "cac_ten_goc": ["Lớp hè", "Code War"]}]'
    )

    cum = extract_events.gom_ten(llm, dem("Lớp hè", "Code War"))

    assert cum == [{"ten_chuan": "A", "cac_ten_goc": ["Lớp hè"]}, {"ten_chuan": "B", "cac_ten_goc": ["Code War"]}]


def test_nhom_thieu_ten_chuan_thi_lay_ten_goc_dau_tien():
    llm = LlmGia('[{"cac_ten_goc": ["Lớp hè"]}]')

    assert extract_events.gom_ten(llm, dem("Lớp hè"))[0]["ten_chuan"] == "Lớp hè"


def test_khong_co_ten_nao_thi_khong_goi_llm():
    class KhongDuocGoi:
        def invoke(self, _prompt):
            raise AssertionError("không được gọi LLM khi danh sách rỗng")

    assert extract_events.gom_ten(KhongDuocGoi(), dem()) == []


def test_llm_tra_ve_khong_phai_mang_thi_bao_loi():
    """Nuốt lỗi ở đây thì mọi tên thành chuỗi riêng và không ai biết lượt gom đã chết."""
    with pytest.raises(ValueError):
        extract_events.gom_ten(LlmGia('{"ten_chuan": "A"}'), dem("Lớp hè"))


# ── Neo vào lịch sự kiện thường niên do CLB cung cấp ─────────────────────────


def lich(tmp_path, noi_dung: str):
    duong = tmp_path / "su_kien_chuan.yaml"
    duong.write_text(noi_dung, encoding="utf-8")
    return extract_events.doc_su_kien_chuan(duong)


def test_moi_cach_viet_deu_tra_ve_ten_chuan(tmp_path):
    """Đây là thứ chữa lỗi đã đo được: "Team Building" và "Teambuilding" phải về
    cùng một chuỗi mà KHÔNG cần tốn một lời gọi gom."""
    chuan = lich(tmp_path, """
su_kien:
  - ten: HIT Teambuilding
    mo_ta: Team building theo khoá.
    alias: [Team Building, Teambuilding]
""")
    bang = extract_events.ten_chuan_da_biet(chuan)

    assert bang[extract_events.normalize_alias("Teambuilding")] == "HIT Teambuilding"
    assert bang[extract_events.normalize_alias("team  building")] == "HIT Teambuilding"
    assert bang[extract_events.normalize_alias("HIT Teambuilding")] == "HIT Teambuilding"


def test_hai_su_kien_clb_co_y_tach_thi_khong_dung_chung_alias(tmp_path):
    """CLB tách "Tuyển thành viên HIT" với "Tuyển CTV"; LLM đã từng gộp hai cái
    này. Alias trùng nhau sẽ dựng lại đúng cái bẫy đó."""
    chuan = lich(tmp_path, """
su_kien:
  - ten: Tuyển thành viên HIT
    alias: [Tuyển thành viên]
  - ten: Tuyển CTV
    alias: [Tuyển cộng tác viên]
""")
    bang = extract_events.ten_chuan_da_biet(chuan)

    assert bang[extract_events.normalize_alias("Tuyển thành viên")] == "Tuyển thành viên HIT"
    assert bang[extract_events.normalize_alias("Tuyển cộng tác viên")] == "Tuyển CTV"


def test_alias_trung_nhau_thi_muc_dau_tien_thang(tmp_path):
    """Không được để mục sau lặng lẽ cướp alias của mục trước — người sửa file
    sẽ không hiểu vì sao một sự kiện bỗng nuốt hết bài của sự kiện khác."""
    chuan = lich(tmp_path, """
su_kien:
  - ten: A
    alias: [chung]
  - ten: B
    alias: [chung]
""")

    assert extract_events.ten_chuan_da_biet(chuan)[extract_events.normalize_alias("chung")] == "A"


def test_thieu_file_lich_thi_chay_khong_neo_chu_khong_chet(tmp_path):
    """Không neo là trạng thái cũ, vẫn dùng được — không đáng để chặn cả mẻ."""
    chuan = extract_events.doc_su_kien_chuan(tmp_path / "khong-ton-tai.yaml")

    assert chuan == []
    assert "chưa có lịch chuẩn" in extract_events.mo_ta_lich_chuan(chuan)


def test_muc_thieu_ten_bi_bo_qua(tmp_path):
    chuan = lich(tmp_path, """
su_kien:
  - ten: "  "
    alias: [x]
  - ten: HIT Open Day
""")

    assert [sk.ten for sk in chuan] == ["HIT Open Day"]


# ── Chia lô: gửi cả 142 tên một lần đã đo được là timeout ────────────────────


class LlmNhieuLuot:
    """Trả lời khác nhau theo từng lời gọi, để dựng cảnh gom hai vòng."""

    def __init__(self, *tra_ve: str) -> None:
        self.tra_ve = list(tra_ve)
        self.prompts: list[str] = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        noi_dung = self.tra_ve[min(len(self.prompts) - 1, len(self.tra_ve) - 1)]
        return type("Msg", (), {"content": noi_dung})()


def test_danh_sach_ngan_van_chi_mot_loi_goi():
    llm = LlmNhieuLuot('[{"ten_chuan": "Lớp hè", "cac_ten_goc": ["Lớp hè", "Lop he"]}]')

    extract_events.gom_ten(llm, dem("Lớp hè", "Lop he"), kich_thuoc_lo=40)

    assert len(llm.prompts) == 1


def test_chia_lo_roi_gom_lai_lan_hai():
    """Hai cách viết rơi vào hai lô khác nhau vẫn phải về được cùng một chuỗi."""
    llm = LlmNhieuLuot(
        '[{"ten_chuan": "Team Building", "cac_ten_goc": ["Team Building"]}]',
        '[{"ten_chuan": "Teambuilding", "cac_ten_goc": ["Teambuilding"]}]',
        '[{"ten_chuan": "Team Building", "cac_ten_goc": ["Team Building", "Teambuilding"]}]',
    )

    cum = extract_events.gom_ten(llm, dem("Team Building", "Teambuilding"), kich_thuoc_lo=1)

    assert len(llm.prompts) == 3, "hai lô + một lượt gom lại"
    assert cum == [{"ten_chuan": "Team Building", "cac_ten_goc": ["Team Building", "Teambuilding"]}]


def test_sap_xep_theo_ten_da_chuan_hoa_truoc_khi_chia_lo():
    """Xếp cạnh nhau thì các biến thể rơi cùng lô — chỗ này là thứ khiến việc
    chia lô không làm hỏng kết quả gom."""
    llm = LlmNhieuLuot('[{"ten_chuan": "x", "cac_ten_goc": []}]')

    extract_events.gom_ten(llm, dem("Zebra", "Team Building", "Teambuilding"), kich_thuoc_lo=2)

    lo_dau = llm.prompts[0]
    assert "Team Building" in lo_dau and "Teambuilding" in lo_dau
    assert "Zebra" not in lo_dau


def test_cache_gom_du_ten_thi_dung_lai(tmp_path):
    duong = tmp_path / "pass2.json"
    duong.write_text(
        '[{"ten_chuan": "Lớp hè", "cac_ten_goc": ["Lớp hè", "Lop he"]}]', encoding="utf-8"
    )

    assert extract_events.doc_cache_gom(duong, dem("Lớp hè", "Lop he"))


def test_cache_gom_thieu_ten_thi_bo_di(tmp_path):
    """Dùng lại bản gom thiếu tên thì tên mới lặng lẽ thành chuỗi riêng — đúng
    thứ lượt gom sinh ra để tránh."""
    duong = tmp_path / "pass2.json"
    duong.write_text('[{"ten_chuan": "Lớp hè", "cac_ten_goc": ["Lớp hè"]}]', encoding="utf-8")

    assert extract_events.doc_cache_gom(duong, dem("Lớp hè", "Code War")) == []


def test_cache_gom_hong_khong_lam_sap_chuong_trinh(tmp_path):
    duong = tmp_path / "pass2.json"
    duong.write_text("{ khong phai json", encoding="utf-8")

    assert extract_events.doc_cache_gom(duong, dem("Lớp hè")) == []


def test_so_luot_goi_bi_chan_tren_du_llm_khong_gom_duoc_gi():
    """Vòng hai là ĐÚNG một lời gọi, không đệ quy — nếu không thì LLM trả về mỗi
    tên một nhóm sẽ làm nó gọi lại mãi trên đúng danh sách cũ."""
    llm = LlmNhieuLuot("[]")

    cum = extract_events.gom_ten(llm, dem("A", "B", "C"), kich_thuoc_lo=1)

    assert len(llm.prompts) == 4, "ba lô của vòng đầu + đúng một lượt gom lại"
    assert sorted(n["ten_chuan"] for n in cum) == ["A", "B", "C"]
