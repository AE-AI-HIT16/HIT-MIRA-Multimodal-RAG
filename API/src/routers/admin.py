"""Số liệu kho, job nền và báo cáo đánh giá cho màn admin — T-62.

Ban đầu file này **cố ý chỉ có endpoint đọc**: chưa có xác thực (T-51) thì mở
một endpoint khởi chạy tiến trình con ra Internet là dựng sẵn đường chạy lệnh
từ xa cho bất kỳ ai. Nay `yeu_cau_admin` đã có, nên phần ghi được mở — nhưng
**mọi endpoint ở đây đều phải đứng sau `yeu_cau_admin`**, kể cả endpoint đọc.
Bỏ quên một cái là mở lại đúng lỗ hổng vừa nói.

Việc nặng không chạy trong tiến trình API mà đẩy sang `src/jobs/runner.py` —
lý do và giới hạn của cách đó nằm trong docstring của module ấy.

Số liệu lấy từ **nguồn thật**, không có bảng đếm sẵn: PostgreSQL cho dữ liệu
gốc, Qdrant cho phần đã index. Hai con số lệch nhau chính là thứ đáng xem —
đó là cách phát hiện "đã nạp nhưng chưa index".
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, status

from src.jobs.runner import JobDangChayError, JobRunner
from src.log.logger import logger
from src.rag_video_anh.repository.database import get_session_manager
from src.rag_video_anh.vector_store.vector_store import QdrantVideoVectorStore
from src.routers.auth import yeu_cau_admin

router = APIRouter(prefix="/admin", tags=["admin"])

REPO_ROOT = Path(__file__).resolve().parents[3]
BAO_CAO_TRUY_XUAT = REPO_ROOT / "data" / "eval" / "report.json"
BAO_CAO_NOI_QUY = REPO_ROOT / "data" / "eval" / "report_noiquy.json"

JOB_EVAL = "eval"
# Chỉ hai nhánh media có bước index riêng. Nội quy KHÔNG có mặt ở đây một cách
# có chủ ý: `/api/documents/upload` đã tách chunk + nhúng + ghi Qdrant ngay lúc
# nạp, nên một nút "Index nội quy" sẽ không có việc gì để làm.
TARGET_INDEX = ("media", "videos")

LENH_JOB: dict[str, list[str]] = {
    # `sys.executable` chứ không phải "python": API hay chạy trong venv/conda mà
    # "python" trên PATH lại là bản hệ thống, thiếu sạch dependency.
    "media": [sys.executable, "scripts/index_image_units.py", "--apply"],
    "videos": [sys.executable, "scripts/index_all_videos.py", "--apply"],
    JOB_EVAL: [sys.executable, "scripts/run_eval.py"],
}


@lru_cache(maxsize=1)
def lay_job_runner() -> JobRunner:
    return JobRunner(LENH_JOB, cwd=REPO_ROOT)


@lru_cache(maxsize=1)
def _vector_store() -> QdrantVideoVectorStore:
    return QdrantVideoVectorStore()


def _dem_qdrant(collection: str) -> int:
    try:
        return int(_vector_store().client.count(collection, exact=True).count)
    except Exception as exc:  # noqa: BLE001 — thiếu một collection không được giết cả trang
        logger.warning(f"Không đếm được collection '{collection}': {exc.__class__.__name__}: {exc}")
        return 0


def _dem_theo_payload(collection: str, khoa: str, gia_tri: Any) -> int:
    """Đếm điểm có `payload[khoa] == gia_tri`, cuộn qua cả collection.

    Qdrant đếm theo filter được, nhưng chỉ khi khoá đã lập chỉ mục payload —
    kho này thì chưa. Cuộn tay chậm hơn nhưng luôn đúng, và số điểm ở đây mới
    vài nghìn.
    """
    dem = 0
    offset = None
    try:
        while True:
            diem, offset = _vector_store().client.scroll(
                collection, limit=1000, offset=offset, with_payload=[khoa], with_vectors=False
            )
            dem += sum(1 for p in diem if (p.payload or {}).get(khoa) == gia_tri)
            if offset is None:
                break
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Không cuộn được '{collection}': {exc.__class__.__name__}: {exc}")
    return dem


def _dem_video_da_index() -> int:
    video_ids: set[str] = set()
    offset = None
    try:
        while True:
            diem, offset = _vector_store().client.scroll(
                "video_transcript", limit=1000, offset=offset, with_payload=["video_id"], with_vectors=False
            )
            video_ids.update(str(p.payload["video_id"]) for p in diem if (p.payload or {}).get("video_id"))
            if offset is None:
                break
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Không đếm được video đã index: {exc.__class__.__name__}: {exc}")
    return len(video_ids)


def _dem_tai_lieu_noi_quy() -> int:
    ids: set[str] = set()
    offset = None
    try:
        while True:
            diem, offset = _vector_store().client.scroll(
                "rag_documents", limit=1000, offset=offset, with_payload=["document_id"], with_vectors=False
            )
            ids.update(str(p.payload["document_id"]) for p in diem if (p.payload or {}).get("document_id"))
            if offset is None:
                break
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Không đếm được tài liệu nội quy: {exc.__class__.__name__}: {exc}")
    return len(ids)


@router.get("/stats")
def thong_ke(_admin: Any = Depends(yeu_cau_admin)) -> dict[str, int]:
    """Đếm dữ liệu gốc (PostgreSQL) và dữ liệu đã index (Qdrant), song song nhau.

    Khoá sau quyền admin (TC-505): số liệu này lộ quy mô kho và tiến độ index —
    không phải bí mật lớn, nhưng cũng không có lý do gì để mở cho người lạ.
    """
    with get_session_manager().session() as session:
        theo_loai = dict(
            session.execute(sa.text("select media_type, count(*) from media group by 1")).all()
        )
        so_post = int(session.execute(sa.text("select count(*) from posts")).scalar() or 0)

    chunk_noi_quy = _dem_qdrant("rag_documents")
    return {
        "posts": so_post,
        "images": int(theo_loai.get("image", 0)),
        "videos": int(theo_loai.get("video", 0)),
        "regulations_active": _dem_tai_lieu_noi_quy(),
        "rule_chunks": chunk_noi_quy,
        "indexed_images": _dem_theo_payload("media_clip", "media_kind", "image"),
        "indexed_videos": _dem_video_da_index(),
        "indexed_rule_chunks": chunk_noi_quy,
    }


@router.post("/index/{target}")
def chay_index(
    target: str,
    _admin: Any = Depends(yeu_cau_admin),
    runner: JobRunner = Depends(lay_job_runner),
) -> dict[str, Any]:
    """P4-3 ③: chạy pipeline index cho asset chưa index, dưới dạng job nền."""
    if target == "regulations":
        raise HTTPException(
            status_code=422,
            detail=(
                "Nội quy được tách chunk và nhúng ngay lúc nạp qua "
                "/api/documents/upload — không có bước index riêng để chạy."
            ),
        )
    if target not in TARGET_INDEX:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không có job index '{target}'. Hợp lệ: {', '.join(TARGET_INDEX)}.",
        )
    try:
        return runner.start(target).to_dict()
    except JobDangChayError as exc:
        # 409 chứ không 500: đây là người dùng bấm hai lần, không phải server hỏng.
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.get("/index/status")
def trang_thai_index(
    _admin: Any = Depends(yeu_cau_admin),
    runner: JobRunner = Depends(lay_job_runner),
) -> list[dict[str, Any]]:
    """Trạng thái MỌI job nền, gồm cả `eval` — web lọc theo `target` nó quan tâm."""
    return [trang_thai.to_dict() for trang_thai in runner.status_all()]


@router.post("/eval/run")
def chay_danh_gia(
    _admin: Any = Depends(yeu_cau_admin),
    runner: JobRunner = Depends(lay_job_runner),
) -> dict[str, Any]:
    """US-602.1: khởi chạy đánh giá. Trả về **trạng thái job**, không phải báo cáo.

    Một lượt đánh giá nhúng lại toàn bộ tập truy vấn và mất hàng chục giây tới
    vài phút. Treo nó vào một request HTTP là cầm chắc timeout ở proxy, và
    người dùng mất luôn kết quả của một lượt chạy đã tốn quota. Nên: chạy nền,
    theo dõi bằng `GET /admin/index/status` (target `eval`), rồi đọc kết quả ở
    `GET /admin/eval/report` khi job xong.
    """
    try:
        return runner.start(JOB_EVAL).to_dict()
    except JobDangChayError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


def _bao_cao_truy_xuat() -> dict[str, Any] | None:
    if not BAO_CAO_TRUY_XUAT.exists():
        return None
    return json.loads(BAO_CAO_TRUY_XUAT.read_text(encoding="utf-8"))


def _bao_cao_noi_quy() -> dict[str, Any] | None:
    if not BAO_CAO_NOI_QUY.exists():
        return None
    return json.loads(BAO_CAO_NOI_QUY.read_text(encoding="utf-8"))


@router.get("/eval/report")
def bao_cao_danh_gia(_admin: Any = Depends(yeu_cau_admin)) -> dict[str, Any]:
    """Gộp báo cáo truy xuất và báo cáo nhánh nội quy thành một khối cho web.

    Đọc file do `scripts/run_eval.py` / `run_eval_noiquy.py` sinh ra chứ không
    chạy lại phép đo: một lượt đánh giá gọi hàng chục lần LLM và mất vài phút,
    không thể treo vào một request HTTP.
    """
    truy_xuat = _bao_cao_truy_xuat()
    if truy_xuat is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chưa có báo cáo. Chạy: python scripts/run_eval.py",
        )
    noi_quy = _bao_cao_noi_quy() or {}
    tong_hop_noi_quy = noi_quy.get("tong_hop") or {}

    chinh = truy_xuat["chi_so_chinh"]
    k = truy_xuat["k"]
    latency = truy_xuat["latency"]
    ghi_chu = [
        f"Đo trên {chinh['so_truy_van']} truy vấn trong miền, k={k}.",
        "Latency đo trong tiến trình, chưa gồm chi phí HTTP — là cận dưới.",
    ]
    theo_loai = truy_xuat.get("theo_loai") or {}
    so_chunk_noi_quy = _dem_qdrant("rag_documents")
    if (theo_loai.get("regulation") or {}).get("so_truy_van") and so_chunk_noi_quy <= k:
        # Kho nhỏ hơn k thì mọi truy vấn đều trả về toàn bộ kho, nên Recall
        # không thể khác 1.0. Không nói ra thì con số này bị trích như một
        # thành tích.
        ghi_chu.append(
            f"Recall nhánh nội quy = 1.0 là con số rỗng: kho chỉ có {so_chunk_noi_quy} "
            f"chunk mà k={k}, nên mọi truy vấn đều trả về toàn bộ kho. Chỉ MRR mới có nghĩa."
        )
    return {
        "n_queries": chinh["so_truy_van"],
        "recall": {str(k): chinh["recall_at_k"]},
        "recall_by_category": {
            loai: so["recall_at_k"] for loai, so in theo_loai.items() if so["so_truy_van"]
        },
        "mrr": chinh["mrr"],
        "latency_ms": {
            "avg": round(latency["trung_binh"] * 1000, 1),
            "p95": round(latency["p95"] * 1000, 1),
        },
        "routing_accuracy": tong_hop_noi_quy.get("routing_accuracy"),
        "groundedness": tong_hop_noi_quy.get("groundedness"),
        "targets": truy_xuat.get("muc_tieu") or {},
        "passed": truy_xuat.get("dat_muc_tieu") or {},
        "notes": " ".join(ghi_chu),
    }
