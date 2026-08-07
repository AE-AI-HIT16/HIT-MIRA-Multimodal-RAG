"""Truy hồi media ba nhánh văn bản — TC-907.

Không gọi nhà cung cấp nhúng, không chạm Qdrant. Bốn tính chất phải giữ: một
query một lần nhúng, ngưỡng chạy trước hợp nhất, gom theo `max` chứ không theo
tổng, và nhánh rỗng phải được nói ra.
"""

from __future__ import annotations

import pytest

from src.rag_video_anh.retrieval.text_retrieval_service import (
    MAX_TRANSCRIPT_MOMENTS,
    RRF_K,
    MediaTextRetrievalService,
)
from src.rag_video_anh.retrieval.text_units import SOURCE_CAPTION, SOURCE_OCR, SOURCE_TRANSCRIPT


class FakeEmbedder:
    def __init__(self) -> None:
        self.so_lan = 0

    def embed_query(self, text: str) -> list[float]:
        self.so_lan += 1
        return [0.1] * 1536


class FakeStore:
    """Trả point theo `source_type` mà tầng trên xin, giữ nguyên thứ tự đã cho."""

    models = None       # để retriever dựng filter dạng dict, không cần qdrant thật

    def __init__(self, theo_nguon: dict[str, list[dict]]) -> None:
        self.theo_nguon = theo_nguon
        self.calls: list[dict] = []

    def search_points(self, *, collection_name, vector, limit, query_filter=None):
        nguon = (query_filter or {}).get("source_type")
        self.calls.append({"source_type": nguon, "limit": limit})
        return list(self.theo_nguon.get(nguon, []))[:limit]


def diem(score, *, source_type, unit_id, **payload):
    payload.setdefault("post_id", "post-1")
    payload.setdefault("text", f"nội dung {unit_id}")
    payload["unit_id"] = unit_id
    payload["source_type"] = source_type
    return {"id": unit_id, "score": score, "payload": payload}


def dung(theo_nguon, **kwargs):
    emb, store = FakeEmbedder(), FakeStore(theo_nguon)
    svc = MediaTextRetrievalService(text_embedder=emb, vector_store=store, **kwargs)
    return svc, emb, store


# ---------------------------------------------------------------- một lần nhúng


def test_query_chi_nhung_mot_lan_cho_ca_ba_nhanh() -> None:
    svc, emb, store = dung({
        SOURCE_CAPTION: [diem(0.7, source_type=SOURCE_CAPTION, unit_id="c1", image_media_id="img-1")],
    })
    svc.retrieve("sự kiện lửa trại")

    assert emb.so_lan == 1
    assert [c["source_type"] for c in store.calls] == [SOURCE_CAPTION, SOURCE_OCR, SOURCE_TRANSCRIPT]


def test_transcript_lay_nhieu_ung_vien_hon() -> None:
    """Nhiều chunk cùng thuộc một video, sau khi gom thì tụt rất nhanh."""
    svc, _, store = dung({})
    svc.retrieve("x")
    theo_nguon = {c["source_type"]: c["limit"] for c in store.calls}
    assert theo_nguon[SOURCE_TRANSCRIPT] > theo_nguon[SOURCE_CAPTION]


# ---------------------------------------------------------------- ngưỡng


def test_nguong_chay_truoc_hop_nhat_va_theo_tung_nguon() -> None:
    """RRF chỉ nhìn hạng: câu ngoài phạm vi vẫn có hạng 1, nên ngưỡng phải chặn trước."""
    svc, _, _ = dung(
        {
            SOURCE_CAPTION: [diem(0.30, source_type=SOURCE_CAPTION, unit_id="c1", image_media_id="img-1")],
            SOURCE_OCR: [diem(0.62, source_type=SOURCE_OCR, unit_id="o1", image_media_id="img-2")],
        },
        thresholds={SOURCE_CAPTION: 0.55, SOURCE_OCR: 0.55, SOURCE_TRANSCRIPT: 0.55},
    )
    ket = svc.retrieve("x")

    assert [r["entity_id"] for r in ket["results"]] == ["img-2"]
    assert "caption" not in ket["results"][0]["scores_by_source"]


def test_moi_nhanh_deu_duoi_nguong_thi_khong_tim_thay() -> None:
    svc, _, _ = dung(
        {SOURCE_CAPTION: [diem(0.20, source_type=SOURCE_CAPTION, unit_id="c1", image_media_id="img-1")]},
        thresholds={SOURCE_CAPTION: 0.9, SOURCE_OCR: 0.9, SOURCE_TRANSCRIPT: 0.9},
    )
    ket = svc.retrieve("chuyện hoàn toàn không liên quan")

    assert ket["found"] is False
    assert ket["results"] == []
    assert any("không gọi llm" in n.lower() for n in ket["notes"])


def test_nguong_mac_dinh_khong_loc_gi() -> None:
    """Đặt bừa một ngưỡng 'trông hợp lý' sẽ im lặng cắt mất kết quả đúng."""
    svc, _, _ = dung(
        {SOURCE_CAPTION: [diem(0.05, source_type=SOURCE_CAPTION, unit_id="c1", image_media_id="img-1")]}
    )
    assert svc.retrieve("x")["found"] is True


# ---------------------------------------------------------------- gom nhóm


def test_nhieu_manh_ocr_khong_cong_don_diem() -> None:
    """Lấy tổng thì ảnh nhiều mảnh OCR xếp cao chỉ vì có nhiều point."""
    svc, _, _ = dung({
        SOURCE_OCR: [
            diem(0.61, source_type=SOURCE_OCR, unit_id="o1#0", image_media_id="img-1"),
            diem(0.60, source_type=SOURCE_OCR, unit_id="o1#1", image_media_id="img-1"),
            diem(0.59, source_type=SOURCE_OCR, unit_id="o1#2", image_media_id="img-1"),
            diem(0.62, source_type=SOURCE_OCR, unit_id="o2", image_media_id="img-2"),
        ],
    })
    ket = svc.retrieve("x")

    assert [r["entity_id"] for r in ket["results"]] == ["img-2", "img-1"]
    assert ket["results"][1]["scores_by_source"]["ocr"] == pytest.approx(0.61)


def test_caption_va_ocr_cua_cung_anh_gop_thanh_mot_ket_qua() -> None:
    svc, _, _ = dung({
        SOURCE_CAPTION: [diem(0.7, source_type=SOURCE_CAPTION, unit_id="c1", image_media_id="img-1")],
        SOURCE_OCR: [diem(0.6, source_type=SOURCE_OCR, unit_id="o1", image_media_id="img-1")],
    })
    ket = svc.retrieve("x")

    assert len(ket["results"]) == 1
    assert {e["source_type"] for e in ket["results"][0]["evidence"]} == {"caption", "ocr"}


def test_moi_thu_thuoc_video_gom_ve_mot_ket_qua_video() -> None:
    svc, _, _ = dung({
        SOURCE_CAPTION: [diem(0.7, source_type=SOURCE_CAPTION, unit_id="c1",
                              video_id="vid-1", frame_media_id="fr-1", timestamp_sec=125.0)],
        SOURCE_TRANSCRIPT: [diem(0.66, source_type=SOURCE_TRANSCRIPT, unit_id="t1",
                                 video_id="vid-1", start_sec=120.0, end_sec=180.0)],
    })
    ket = svc.retrieve("lửa trại")

    assert len(ket["results"]) == 1
    r = ket["results"][0]
    assert r["entity_type"] == "video" and r["entity_id"] == "vid-1"
    # Mốc thời gian nằm trong evidence chứ không bị gom phẳng mất.
    assert [e.get("timestamp_sec") for e in r["evidence"] if e["source_type"] == "caption"] == [125.0]
    assert [e.get("start_sec") for e in r["evidence"] if e["source_type"] == "transcript"] == [120.0]


def test_moment_transcript_chong_thoi_gian_bi_loai() -> None:
    """Ba đoạn chồng nhau là ba lần kể cùng một câu."""
    svc, _, _ = dung({
        SOURCE_TRANSCRIPT: [
            diem(0.70, source_type=SOURCE_TRANSCRIPT, unit_id="t1", video_id="v", start_sec=100.0, end_sec=160.0),
            diem(0.69, source_type=SOURCE_TRANSCRIPT, unit_id="t2", video_id="v", start_sec=120.0, end_sec=180.0),
            diem(0.68, source_type=SOURCE_TRANSCRIPT, unit_id="t3", video_id="v", start_sec=300.0, end_sec=360.0),
        ],
    })
    moment = ket_moment(svc.retrieve("x"))

    assert [e["unit_id"] for e in moment] == ["t1", "t3"]


def test_gioi_han_so_moment_moi_video() -> None:
    svc, _, _ = dung({
        SOURCE_TRANSCRIPT: [
            diem(0.9 - i / 100, source_type=SOURCE_TRANSCRIPT, unit_id=f"t{i}", video_id="v",
                 start_sec=i * 100.0, end_sec=i * 100.0 + 50.0)
            for i in range(6)
        ],
    })
    assert len(ket_moment(svc.retrieve("x"))) == MAX_TRANSCRIPT_MOMENTS


def ket_moment(ket: dict) -> list[dict]:
    return [e for e in ket["results"][0]["evidence"] if e["source_type"] == SOURCE_TRANSCRIPT]


# ---------------------------------------------------------------- RRF


def test_trung_o_nhieu_nguon_thi_xep_cao_hon_du_diem_le_thap_hon() -> None:
    """Đó chính là điều RRF sinh ra để làm."""
    svc, _, _ = dung({
        SOURCE_CAPTION: [
            diem(0.80, source_type=SOURCE_CAPTION, unit_id="c1", image_media_id="chi-caption"),
            diem(0.70, source_type=SOURCE_CAPTION, unit_id="c2", image_media_id="ca-ba"),
        ],
        SOURCE_OCR: [diem(0.70, source_type=SOURCE_OCR, unit_id="o1", image_media_id="ca-ba")],
        SOURCE_TRANSCRIPT: [diem(0.70, source_type=SOURCE_TRANSCRIPT, unit_id="t1", video_id="ca-ba")],
    })
    ket = svc.retrieve("x")

    assert ket["results"][0]["entity_id"] == "ca-ba"
    assert ket["results"][0]["score"] > ket["results"][1]["score"]


def test_diem_rrf_dung_cong_thuc_hang() -> None:
    svc, _, _ = dung({
        SOURCE_CAPTION: [diem(0.7, source_type=SOURCE_CAPTION, unit_id="c1", image_media_id="img-1")],
        SOURCE_OCR: [diem(0.6, source_type=SOURCE_OCR, unit_id="o1", image_media_id="img-1")],
    })
    ket = svc.retrieve("x")
    # `to_dict` làm tròn 6 chữ số nên so xấp xỉ ở đúng độ chính xác đó.
    assert ket["results"][0]["score"] == pytest.approx(2 / (RRF_K + 1), abs=1e-6)


def test_trong_so_mac_dinh_bang_nhau() -> None:
    """Không đặt caption×0.5 + ocr×0.3 khi chưa có bằng chứng nào chống lưng."""
    svc, _, _ = dung({})
    assert len(set(svc.weights.values())) == 1


# ---------------------------------------------------------------- đa dạng hoá


def test_mot_bai_khong_chiem_het_trang_ket_qua() -> None:
    svc, _, _ = dung(
        {
            SOURCE_CAPTION: [
                *[diem(0.9 - i / 100, source_type=SOURCE_CAPTION, unit_id=f"a{i}",
                       image_media_id=f"img-{i}", post_id="post-A") for i in range(5)],
                diem(0.5, source_type=SOURCE_CAPTION, unit_id="b1", image_media_id="img-B", post_id="post-B"),
            ],
        },
        max_per_post=2,
        top_k=3,
    )
    ket = svc.retrieve("x")

    assert [r["post_id"] for r in ket["results"]] == ["post-A", "post-A", "post-B"]


def test_phan_bi_nen_xuong_cuoi_chu_khong_bi_vut() -> None:
    svc, _, _ = dung(
        {SOURCE_CAPTION: [
            diem(0.9 - i / 100, source_type=SOURCE_CAPTION, unit_id=f"a{i}",
                 image_media_id=f"img-{i}", post_id="post-A") for i in range(4)
        ]},
        max_per_post=1,
        top_k=4,
    )
    assert len(svc.retrieve("x")["results"]) == 4


# ---------------------------------------------------------------- ghi chú


def test_nhanh_rong_phai_duoc_noi_ra() -> None:
    """Coi 'không có transcript nào' là 'không ai nói gì về việc này' là hiểu sai."""
    svc, _, _ = dung({
        SOURCE_CAPTION: [diem(0.7, source_type=SOURCE_CAPTION, unit_id="c1", image_media_id="img-1")],
    })
    ghi_chu = " ".join(svc.retrieve("x")["notes"])

    assert "ocr" in ghi_chu and "transcript" in ghi_chu


def test_query_rong_bi_chan() -> None:
    svc, _, _ = dung({})
    with pytest.raises(ValueError):
        svc.retrieve("   ")
