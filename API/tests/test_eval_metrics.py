"""Chỉ số đánh giá phải khớp tính tay — TC-602 / US-602.1, US-603.1, US-601.1.

Test cố ý dùng ví dụ nhỏ tính nhẩm được. Một bộ đo mà không ai kiểm được bằng
tay thì con số nó in ra không có sức nặng nào khi đưa vào báo cáo.
"""

from __future__ import annotations

import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from src.eval.metrics import (  # noqa: E402
    BaoCao,
    KetQuaTruyVan,
    percentile,
    recall_at_k,
    reciprocal_rank,
)


def _ket_qua(**thay_doi) -> KetQuaTruyVan:
    mac_dinh = dict(
        id="X-1",
        loai="media_image",
        cau_hoi="câu hỏi",
        xep_hang=["a", "b", "c"],
        dap_an=["a"],
        diem_cao_nhat=0.5,
        giay=1.0,
    )
    mac_dinh.update(thay_doi)
    return KetQuaTruyVan(**mac_dinh)


# ---------------------------------------------------------------- recall@k


def test_recall_lay_duoc_2_trong_3_dap_an() -> None:
    # top-5 chứa a và c; có 3 đáp án đúng, min(3,5)=3 → 2/3
    assert recall_at_k(["a", "x", "c", "y", "z"], ["a", "b", "c"], 5) == 2 / 3


def test_recall_chuan_hoa_theo_k_khi_dap_an_nhieu_hon_k() -> None:
    """Câu có 77 đáp án đúng mà k=5 thì trần chỉ là 5, không phải 77.

    Không chuẩn hoá thì mọi câu nhãn rộng đều bị chấm trượt oan dù top-5 đúng
    sạch, và chỉ số chính bị dìm bởi cách viết nhãn chứ không bởi chất lượng.
    """
    dap_an = [f"d{i}" for i in range(77)]
    assert recall_at_k(dap_an[:5], dap_an, 5) == 1.0


def test_recall_khong_co_dap_an_dung_la_0() -> None:
    assert recall_at_k(["x", "y"], ["a"], 5) == 0.0


def test_recall_khong_co_nhan_la_0_khong_phai_chia_cho_0() -> None:
    assert recall_at_k(["a"], [], 5) == 0.0


def test_recall_khong_dem_ngoai_top_k() -> None:
    # 'a' đứng hạng 6, ngoài k=5
    assert recall_at_k(["x", "x", "x", "x", "x", "a"], ["a"], 5) == 0.0


# ------------------------------------------------------------------- MRR


def test_reciprocal_rank_theo_dap_an_dung_dau_tien() -> None:
    assert reciprocal_rank(["x", "a", "b"], ["a", "b"]) == 0.5
    assert reciprocal_rank(["a"], ["a"]) == 1.0
    assert reciprocal_rank(["x", "y", "z"], ["a"]) == 0.0


def test_mrr_la_trung_binh_cua_reciprocal_rank() -> None:
    """MRR = trung bình (1/1, 1/2, 0) = 0.5 — tính tay."""
    bao_cao = BaoCao(k=5, muc_tieu={})
    bao_cao.ket_qua = [
        _ket_qua(id="A", xep_hang=["a", "x"], dap_an=["a"]),
        _ket_qua(id="B", xep_hang=["x", "a"], dap_an=["a"]),
        _ket_qua(id="C", xep_hang=["x", "y"], dap_an=["a"]),
    ]
    assert bao_cao.chinh()["mrr"] == (1.0 + 0.5 + 0.0) / 3


# -------------------------------------------------------------- latency


def test_percentile_nearest_rank() -> None:
    assert percentile([1.0], 95) == 1.0
    assert percentile(list(range(1, 101)), 95) == 95
    assert percentile([], 95) == 0.0


def test_latency_khong_tinh_truy_van_loi() -> None:
    """US-603.1: truy vấn lỗi không được kéo latency thành công lên/xuống."""
    bao_cao = BaoCao(k=5, muc_tieu={})
    bao_cao.ket_qua = [
        _ket_qua(id="A", giay=1.0),
        _ket_qua(id="B", giay=3.0),
        _ket_qua(id="C", giay=99.0, loi="HTTPError: 500"),
    ]
    latency = bao_cao.latency()
    assert latency["so_mau"] == 2
    assert latency["trung_binh"] == 2.0


# ------------------------------------------------- loại mục thiếu nhãn


def test_muc_thieu_nhan_bi_loai_va_duoc_bao() -> None:
    """US-601.1 AC-2: thiếu nhãn thì loại khỏi phép tính, và phải nêu tên."""
    bao_cao = BaoCao(k=5, muc_tieu={})
    bao_cao.ket_qua = [
        _ket_qua(id="A", xep_hang=["a"], dap_an=["a"]),
        _ket_qua(id="B", xep_hang=["b"], dap_an=[]),
    ]
    assert bao_cao.chinh()["so_truy_van"] == 1
    assert bao_cao.bi_loai() == ["B"]


def test_ngoai_mien_khong_vao_chi_so_chinh() -> None:
    """Câu ngoài miền không có đáp án đúng nào trong kho, tính vào là vô nghĩa."""
    bao_cao = BaoCao(k=5, muc_tieu={})
    bao_cao.ket_qua = [
        _ket_qua(id="A", xep_hang=["a"], dap_an=["a"]),
        _ket_qua(id="OOD", loai="ngoai_mien", xep_hang=["x"], dap_an=[]),
    ]
    assert bao_cao.chinh()["so_truy_van"] == 1


def test_ocr_khong_vao_chi_so_chinh() -> None:
    """`media_image_ocr` đo phạm vi v2 (hybrid search), báo cáo riêng."""
    bao_cao = BaoCao(k=5, muc_tieu={})
    bao_cao.ket_qua = [
        _ket_qua(id="A", xep_hang=["a"], dap_an=["a"]),
        _ket_qua(id="O", loai="media_image_ocr", xep_hang=["x"], dap_an=["a"]),
    ]
    assert bao_cao.chinh()["so_truy_van"] == 1
    assert bao_cao.theo_loai()["media_image_ocr"]["so_truy_van"] == 1


# ------------------------------------------------------ cờ đạt / chưa đạt


def test_duoi_muc_tieu_thi_bi_danh_dau_chua_dat() -> None:
    """US-602.1 AC-2 và US-603.1 AC-2."""
    bao_cao = BaoCao(k=5, muc_tieu={"recall_at_k": 0.8, "mrr": 0.6, "latency_trung_binh": 5.0})
    # 'a' đứng hạng 6 nên nằm ngoài k=5: recall=0.0, rr=1/6.
    bao_cao.ket_qua = [
        _ket_qua(id="A", xep_hang=["x"] * 5 + ["a"], dap_an=["a"], giay=9.0)
    ]
    dat = bao_cao.to_dict()["dat_muc_tieu"]
    assert dat["recall_at_k"] is False   # 0.0 < 0.8
    assert dat["mrr"] is False           # 1/6 < 0.6
    assert dat["latency_trung_binh"] is False  # 9.0 > 5.0


def test_nguong_ngoai_mien_bao_ca_hai_dau_de_biet_co_chong_lan() -> None:
    """T-33 cần biết điểm câu ngoài miền có chạm điểm câu trong miền không."""
    bao_cao = BaoCao(k=5, muc_tieu={})
    bao_cao.ket_qua = [
        _ket_qua(id="A", dap_an=["a"], xep_hang=["a"], diem_cao_nhat=0.30),
        _ket_qua(id="OOD", loai="ngoai_mien", dap_an=[], xep_hang=[], diem_cao_nhat=0.40),
    ]
    nguong = bao_cao.diem_ngoai_mien()
    assert nguong["ngoai_mien_cao_nhat"] == 0.40
    assert nguong["trong_mien_thap_nhat"] == 0.30
    assert nguong["ngoai_mien_cao_nhat"] > nguong["trong_mien_thap_nhat"]
