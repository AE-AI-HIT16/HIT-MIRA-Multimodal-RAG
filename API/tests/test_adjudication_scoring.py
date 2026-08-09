"""Metric sau adjudication — TC-904 (quy tắc chốt trước khi có nhãn).

Giá trị kỳ vọng ở đây **tính tay**, không lấy từ đầu ra của chính script. Đó là
điểm mấu chốt: quy tắc đã khoá ngày 06/08/2026 lúc pool còn trắng 451 lượt chấm,
nên sau này đổi công thức cho hợp kết quả sẽ làm gãy test chứ không trôi lặng lẽ.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("score_pool", REPO_ROOT / "scripts/score_adjudicated_pool.py")
score_pool = importlib.util.module_from_spec(_spec)
sys.modules["score_pool"] = score_pool
_spec.loader.exec_module(score_pool)


# ---------------------------------------------------------------- rubric


def test_gain_dung_thang_2_1_0() -> None:
    assert score_pool.GAIN == {"relevant": 2, "partially_relevant": 1, "not_relevant": 0}


def test_strict_khong_gom_partial() -> None:
    """Trộn partial vào strict là làm strict mất hết ý nghĩa của nó."""
    assert score_pool.STRICT == {"relevant"}
    assert score_pool.LENIENT == {"relevant", "partially_relevant"}


# ---------------------------------------------------------------- recall


def test_recall_dem_theo_tong_so_bai_dung() -> None:
    """Mẫu số là TỔNG bài đúng, không phải số vị trí trong top-k."""
    xep = ["a", "x", "b", "y", "z", "c"]
    # 3 bài đúng, hai trong số đó (a, b) nằm trong top-5 -> 2/3
    assert score_pool.recall_at_k(xep, {"a", "b", "c"}) == pytest.approx(2 / 3)


def test_recall_khong_dinh_nghia_khi_khong_co_bai_dung() -> None:
    """Trả 0 thì câu đó kéo trung bình xuống dù model không hề sai — phải ném lỗi."""
    with pytest.raises(ValueError):
        score_pool.recall_at_k(["a"], set())


# ---------------------------------------------------------------- MRR


def test_mrr_lay_hang_cua_bai_dung_dau_tien() -> None:
    assert score_pool.mrr_at_k(["x", "y", "a", "b"], {"a", "b"}) == pytest.approx(1 / 3)


def test_mrr_bang_0_khi_khong_co_gi_trong_top_10() -> None:
    xep = [f"x{i}" for i in range(10)] + ["a"]
    assert score_pool.mrr_at_k(xep, {"a"}) == 0.0


# ---------------------------------------------------------------- nDCG


def test_ndcg_dung_gain_phan_muc_va_chiet_khau_log2() -> None:
    """Tính tay: xếp [partial(1), relevant(2)], lý tưởng [2, 1]."""
    gain = {"a": 1, "b": 2}
    dcg = 1 / math.log2(2) + 2 / math.log2(3)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert score_pool.ndcg_at_k(["a", "b"], gain) == pytest.approx(dcg / idcg)


def test_ndcg_phan_biet_partial_voi_relevant() -> None:
    """Nếu nhị phân hoá thì hai trường hợp này bằng nhau — đó là thứ graded phải bắt."""
    tot = score_pool.ndcg_at_k(["a", "b"], {"a": 2, "b": 1})
    xau = score_pool.ndcg_at_k(["b", "a"], {"a": 2, "b": 1})
    assert tot == pytest.approx(1.0)
    assert xau < tot


def test_ndcg_idcg_tinh_tren_toan_bo_gain_khong_chi_trong_top_k() -> None:
    """Bài đúng nằm ngoài top-k vẫn phải vào IDCG, nếu không nDCG bị thổi lên."""
    gain = {f"p{i}": 2 for i in range(12)}
    xep = [f"p{i}" for i in range(10)]
    assert score_pool.ndcg_at_k(xep, gain, k=10) == pytest.approx(1.0)
    # 10 vị trí, 12 bài đúng: lấp kín top-10 vẫn là hoàn hảo ở độ sâu 10.


def test_ndcg_bang_0_khi_khong_co_gain_duong() -> None:
    assert score_pool.ndcg_at_k(["a"], {"a": 0}) == 0.0


# ---------------------------------------------------------------- khoá pool


def pool_mau(judgment: str | None = "relevant") -> dict:
    return {
        "queries": [
            {"query_id": "TXT-01", "candidates": [
                {"facebook_post_id": "p1", "judgment": judgment},
                {"facebook_post_id": "p2", "judgment": "not_relevant"},
            ]}
        ]
    }


def test_doi_thu_tu_ung_vien_lam_hong_khoa() -> None:
    pool = pool_mau()
    lock = {"sha256_thu_tu_ung_vien": score_pool.bam_thu_tu(pool), "so_luot_cham": 2}
    score_pool.kiem_pool(pool, lock)                      # nguyên vẹn thì qua

    pool["queries"][0]["candidates"].reverse()            # chỉ xáo, không thêm bớt
    with pytest.raises(score_pool.PoolError, match="thứ tự ứng viên đã đổi"):
        score_pool.kiem_pool(pool, lock)


def test_them_ung_vien_lam_hong_khoa() -> None:
    pool = pool_mau()
    lock = {"sha256_thu_tu_ung_vien": score_pool.bam_thu_tu(pool), "so_luot_cham": 2}
    pool["queries"][0]["candidates"].append({"facebook_post_id": "p3", "judgment": "relevant"})
    with pytest.raises(score_pool.PoolError):
        score_pool.kiem_pool(pool, lock)


@pytest.mark.parametrize("nhan", [None, "", "khong_biet", "Relevant"])
def test_thieu_nhan_hoac_sai_nhan_thi_tu_choi_chay(nhan) -> None:
    """Chấm nửa chừng rồi công bố số là cách êm ái nhất để tự lừa mình."""
    pool = pool_mau(nhan)
    lock = {"sha256_thu_tu_ung_vien": score_pool.bam_thu_tu(pool), "so_luot_cham": 2}
    with pytest.raises(score_pool.PoolError, match="chưa chấm hoặc sai nhãn"):
        score_pool.kiem_pool(pool, lock)


def test_bam_khong_phu_judgment() -> None:
    """Băm cả file thì khoá tự vỡ ngay lượt chấm đầu tiên."""
    truoc = score_pool.bam_thu_tu(pool_mau("relevant"))
    sau = score_pool.bam_thu_tu(pool_mau("not_relevant"))
    assert truoc == sau


# ---------------------------------------------------------------- báo cả hai


def test_luon_bao_ca_strict_lan_lenient() -> None:
    """Không có cờ nào để chọn một chế độ — chọn sau khi nhìn kết quả là p-hacking."""
    qrel = {"Q1": {"gain": {"a": 2, "b": 1}, "strict": {"a"}, "lenient": {"a", "b"}}}
    ket = score_pool.tinh_diem({"Q1": ["b", "a"]}, qrel)

    assert set(ket) == {"strict", "lenient", "graded"}
    # strict: chỉ 'a' đúng, nằm hạng 2 -> recall 1/1, MRR 1/2
    assert ket["strict"]["recall_at_5"] == pytest.approx(1.0)
    assert ket["strict"]["mrr_at_10"] == pytest.approx(0.5)
    # lenient: cả hai đúng, bài đúng đầu tiên ở hạng 1 -> MRR 1.0
    assert ket["lenient"]["mrr_at_10"] == pytest.approx(1.0)


def test_cau_khong_co_bai_dung_bi_loai_khoi_trung_binh() -> None:
    qrel = {
        "Q1": {"gain": {"a": 2}, "strict": {"a"}, "lenient": {"a"}},
        "Q2": {"gain": {"b": 0}, "strict": set(), "lenient": set()},
    }
    ket = score_pool.tinh_diem({"Q1": ["a"], "Q2": ["b"]}, qrel)
    assert ket["strict"]["so_cau"] == 1
    assert ket["strict"]["recall_at_5"] == pytest.approx(1.0)
