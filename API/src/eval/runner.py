"""Chạy tập đánh giá qua đúng service mà router dùng — T-71/T-72.

Gọi thẳng `VideoRetrievalService` / `RetrievalService` (cùng hàm `build_*` mà
`src/routers/` gọi) thay vì đi vòng qua HTTP: chạy lại được mà không cần dựng
uvicorn, và số latency đo đúng phần truy xuất chứ không lẫn chi phí HTTP nội bộ.
Con số vì thế là **cận dưới** của độ trễ người dùng thấy — có ghi rõ trong báo cáo.

Truy vấn nội quy cố ý **tắt viết lại truy vấn**: bước rewrite gọi LLM, làm
latency dao động theo nhà cung cấp bên ngoài và khiến phép đo không tái lập
được. Muốn đo cả rewrite thì bật `--rewrite`.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from src.eval.metrics import BaoCao, KetQuaTruyVan

# Nhánh nào của hệ thống trả lời loại truy vấn nào.
NHANH_MEDIA = ("media_image", "media_image_ocr")
NHANH_TRANSCRIPT = ("media_transcript",)
NHANH_NOI_QUY = ("regulation",)


def _xep_hang_clip(ket_qua: dict[str, Any]) -> tuple[list[str], float | None]:
    clips = ket_qua.get("clips") or []
    xep_hang = [str(c.get("unit_id")) for c in clips if c.get("unit_id")]
    diem = float(clips[0]["score"]) if clips else None
    return xep_hang, diem


def _xep_hang_transcript(ket_qua: dict[str, Any]) -> tuple[list[str], float | None]:
    """Service gom moment theo video, nên phải trải phẳng lại theo điểm giảm dần.

    Người dùng thấy danh sách video, nhưng nhãn của tập đánh giá ở mức đoạn
    lời thoại, nên so ở mức đoạn mới đúng thứ đang đo.
    """
    moments: list[tuple[float, str]] = []
    for video in ket_qua.get("videos") or []:
        for moment in video.get("moments") or []:
            if moment.get("unit_id"):
                moments.append((float(moment.get("score") or 0.0), str(moment["unit_id"])))
    moments.sort(key=lambda item: item[0], reverse=True)
    return [unit_id for _, unit_id in moments], (moments[0][0] if moments else None)


def _xep_hang_noi_quy(ket_qua: dict[str, Any]) -> tuple[list[str], float | None]:
    ket = ket_qua.get("results") or []
    xep_hang = [str(r.get("chunk_id")) for r in ket if r.get("chunk_id")]
    diem = float(ket[0]["score"]) if ket else None
    return xep_hang, diem


def chay(
    tap_danh_gia: dict[str, Any],
    *,
    media_service: Any,
    noi_quy_service: Any,
    k: int | None = None,
    rewrite: bool = False,
    tien_trinh: Callable[[str], None] | None = None,
) -> BaoCao:
    meta = tap_danh_gia.get("meta") or {}
    k = k or int(meta.get("k") or 5)
    bao_cao = BaoCao(k=k, muc_tieu=dict(meta.get("muc_tieu") or {}))

    for muc in tap_danh_gia["truy_van"]:
        loai = muc["loai"]
        cau_hoi = muc["cau_hoi"]
        if tien_trinh:
            tien_trinh(f"{muc['id']:8} {cau_hoi[:52]}")

        loi: str | None = None
        xep_hang: list[str] = []
        diem: float | None = None
        bat_dau = time.perf_counter()
        try:
            if loai in NHANH_NOI_QUY:
                ket = noi_quy_service.retrieve(cau_hoi, k, None, rewrite)
                xep_hang, diem = _xep_hang_noi_quy(ket)
            elif loai in NHANH_TRANSCRIPT:
                ket = media_service.retrieve(cau_hoi, k, None, "transcript")
                xep_hang, diem = _xep_hang_transcript(ket)
            elif loai in NHANH_MEDIA:
                ket = media_service.retrieve(cau_hoi, k, None, "clip")
                xep_hang, diem = _xep_hang_clip(ket)
            else:
                # Ngoài miền: hỏi cả hai nhánh, lấy điểm cao nhất bất kể ở đâu —
                # ngưỡng "không tìm thấy" phải chặn được cả hai.
                media = media_service.retrieve(cau_hoi, k, None, "both")
                noi_quy = noi_quy_service.retrieve(cau_hoi, k, None, rewrite)
                diem_ung_vien = [
                    d
                    for d in (
                        _xep_hang_clip(media)[1],
                        _xep_hang_transcript(media)[1],
                        _xep_hang_noi_quy(noi_quy)[1],
                    )
                    if d is not None
                ]
                diem = max(diem_ung_vien) if diem_ung_vien else None
        except Exception as exc:  # noqa: BLE001 — một câu hỏng không được giết cả đợt
            loi = f"{exc.__class__.__name__}: {exc}"
        giay = time.perf_counter() - bat_dau

        bao_cao.ket_qua.append(
            KetQuaTruyVan(
                id=muc["id"],
                loai=loai,
                cau_hoi=cau_hoi,
                xep_hang=xep_hang,
                dap_an=list(muc.get("dap_an") or []),
                diem_cao_nhat=diem,
                giay=giay,
                loi=loi,
            )
        )
    return bao_cao
