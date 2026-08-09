"""Chỉ số đánh giá truy xuất — T-71/T-72, US-602.1/US-603.1.

Thuần tính toán: không chạm mạng, không chạm Qdrant, nhận vào danh sách id đã
xếp hạng và danh sách đáp án đúng. Tách ra như vậy để `TC-602` ("khớp tính
tay") kiểm được bằng test offline, và để đổi nguồn truy xuất mà không phải sửa
chỗ tính số.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable, Sequence


def recall_at_k(xep_hang: Sequence[str], dap_an: Iterable[str], k: int) -> float:
    """Tỷ lệ đáp án đúng lọt vào top-k.

    Dùng định nghĩa "bao nhiêu phần đáp án đúng lấy được", chứ không phải
    hit-rate ("có ít nhất một cái đúng"). Với câu có 77 đáp án đúng mà k=5 thì
    trần lý thuyết chỉ là 5/77 — nên phải chuẩn hoá theo `min(len(dap_an), k)`,
    nếu không mọi câu nhãn rộng đều bị chấm trượt oan.
    """
    dung = set(dap_an)
    if not dung or k <= 0:
        return 0.0
    lay_duoc = sum(1 for don_vi in xep_hang[:k] if don_vi in dung)
    return lay_duoc / min(len(dung), k)


def reciprocal_rank(xep_hang: Sequence[str], dap_an: Iterable[str]) -> float:
    """1/thứ hạng của đáp án đúng ĐẦU TIÊN; 0 nếu không có cái nào."""
    dung = set(dap_an)
    for thu_tu, don_vi in enumerate(xep_hang, start=1):
        if don_vi in dung:
            return 1.0 / thu_tu
    return 0.0


def percentile(gia_tri: Sequence[float], p: float) -> float:
    """Phân vị theo lối nearest-rank — không nội suy, không cần numpy.

    Thứ hạng là `ceil(p/100 * N)`. Dùng `round(... + 0.5)` thay cho `ceil` là
    sai: Python làm tròn về số chẵn, nên `round(95.5)` ra 96 — p95 của dãy
    1..100 khi đó báo 96 thay vì 95. Test `test_percentile_nearest_rank` canh.
    """
    if not gia_tri:
        return 0.0
    sap_xep = sorted(gia_tri)
    vi_tri = max(1, min(len(sap_xep), math.ceil(p / 100.0 * len(sap_xep))))
    return sap_xep[vi_tri - 1]


@dataclass
class KetQuaTruyVan:
    """Kết quả đo của MỘT truy vấn."""

    id: str
    loai: str
    cau_hoi: str
    xep_hang: list[str]
    dap_an: list[str]
    diem_cao_nhat: float | None
    giay: float
    loi: str | None = None

    @property
    def hop_le(self) -> bool:
        """US-601.1 AC-2: mục không có nhãn bị loại khỏi phép tính."""
        return self.loi is None and bool(self.dap_an)

    def recall(self, k: int) -> float:
        return recall_at_k(self.xep_hang, self.dap_an, k)

    def rr(self) -> float:
        return reciprocal_rank(self.xep_hang, self.dap_an)


@dataclass
class BaoCao:
    """Tổng hợp theo nhóm + toàn cục, kèm cờ đạt/chưa đạt so với mục tiêu."""

    k: int
    muc_tieu: dict[str, float]
    ket_qua: list[KetQuaTruyVan] = field(default_factory=list)
    loai_vao_chi_so_chinh: tuple[str, ...] = ("media_image", "media_transcript", "regulation")

    def _nhom(self, loai: str | None = None) -> list[KetQuaTruyVan]:
        return [
            r
            for r in self.ket_qua
            if r.hop_le and (loai is None or r.loai == loai)
        ]

    def _tong_hop(self, muc: list[KetQuaTruyVan]) -> dict[str, float | int]:
        if not muc:
            return {"so_truy_van": 0, "recall_at_k": 0.0, "mrr": 0.0}
        return {
            "so_truy_van": len(muc),
            "recall_at_k": mean(r.recall(self.k) for r in muc),
            "mrr": mean(r.rr() for r in muc),
        }

    def chinh(self) -> dict[str, float | int]:
        return self._tong_hop(
            [r for r in self.ket_qua if r.hop_le and r.loai in self.loai_vao_chi_so_chinh]
        )

    def theo_loai(self) -> dict[str, dict[str, float | int]]:
        moi_loai = sorted({r.loai for r in self.ket_qua})
        return {loai: self._tong_hop(self._nhom(loai)) for loai in moi_loai}

    def latency(self) -> dict[str, float | int]:
        """US-603.1: truy vấn lỗi KHÔNG được tính vào latency thành công."""
        giay = [r.giay for r in self.ket_qua if r.loi is None]
        return {
            "so_mau": len(giay),
            "trung_binh": mean(giay) if giay else 0.0,
            "p95": percentile(giay, 95),
            "cao_nhat": max(giay) if giay else 0.0,
        }

    def diem_ngoai_mien(self) -> dict[str, float | int]:
        """Phân bố điểm của câu ngoài miền — số liệu để đặt ngưỡng cho T-33."""
        diem = [r.diem_cao_nhat for r in self.ket_qua if r.loai == "ngoai_mien" and r.diem_cao_nhat is not None]
        trong_mien = [
            r.diem_cao_nhat
            for r in self.ket_qua
            if r.loai in self.loai_vao_chi_so_chinh and r.diem_cao_nhat is not None
        ]
        return {
            "so_mau_ngoai_mien": len(diem),
            "ngoai_mien_cao_nhat": max(diem) if diem else 0.0,
            "ngoai_mien_trung_binh": mean(diem) if diem else 0.0,
            "trong_mien_thap_nhat": min(trong_mien) if trong_mien else 0.0,
            "trong_mien_trung_binh": mean(trong_mien) if trong_mien else 0.0,
        }

    def bi_loai(self) -> list[str]:
        return [r.id for r in self.ket_qua if r.loai != "ngoai_mien" and not r.hop_le]

    def to_dict(self) -> dict:
        chinh = self.chinh()
        latency = self.latency()
        return {
            "k": self.k,
            "muc_tieu": self.muc_tieu,
            "chi_so_chinh": chinh,
            # US-602.1 AC-2 / US-603.1 AC-2: dưới mục tiêu phải được đánh dấu.
            "dat_muc_tieu": {
                "recall_at_k": chinh["recall_at_k"] >= self.muc_tieu.get("recall_at_k", 0.0),
                "mrr": chinh["mrr"] >= self.muc_tieu.get("mrr", 0.0),
                "latency_trung_binh": latency["trung_binh"]
                <= self.muc_tieu.get("latency_trung_binh", float("inf")),
            },
            "theo_loai": self.theo_loai(),
            "latency": latency,
            "nguong_ngoai_mien": self.diem_ngoai_mien(),
            "bi_loai_vi_thieu_nhan": self.bi_loai(),
            "chi_tiet": [
                {
                    "id": r.id,
                    "loai": r.loai,
                    "cau_hoi": r.cau_hoi,
                    "so_dap_an": len(r.dap_an),
                    "recall_at_k": r.recall(self.k) if r.hop_le else None,
                    "rr": r.rr() if r.hop_le else None,
                    "diem_cao_nhat": r.diem_cao_nhat,
                    "giay": round(r.giay, 3),
                    "loi": r.loi,
                }
                for r in self.ket_qua
            ],
        }
