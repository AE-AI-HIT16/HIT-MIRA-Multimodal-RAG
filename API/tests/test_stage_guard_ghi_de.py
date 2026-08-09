"""Khâu bị tắt theo route thì KHÔNG được ghi gì vào DB.

Bẫy có thật, đo ngày 08/08/2026: một lượt `caption_images.py` chạy với detection
tắt đã xoá 17.262 object YOLO khỏi PostgreSQL, để lại 4.986 hàng `object_results`
mang trạng thái `DONE` — không lỗi, không cảnh báo, và nhìn từ mọi phía đều
giống "YOLO chạy xong và không thấy vật gì". Dữ liệu chỉ cứu lại được vì artifact
gốc còn trong MinIO.

Cùng khuôn lỗi đó nằm ở 6 chỗ và còn ghi cả chuỗi chẩn đoán ("OCR disabled by
route") vào đúng ô nội dung, rồi chuỗi đó đi tiếp vào payload Qdrant.

Bộ test này khoá lại quy tắc: **không chạy thì không ghi**.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.configuration import AppConfig
from src.rag_video_anh.pipeline.media_task_worker import MediaTaskWorker
from src.rag_video_anh.pipeline.stage_guard import khau_da_chay
from src.rag_video_anh.pipeline.video_processing_worker import VideoProcessingWorker
from src.rag_video_anh.schemas import StageStatus


class FakeResultsRepo:
    """Ghi lại mọi lời gọi upsert để test khẳng định được là KHÔNG có lời gọi nào."""

    def __init__(self) -> None:
        self.ocr_calls: list[dict] = []
        self.caption_calls: list[dict] = []
        self.object_calls: list[dict] = []

    def upsert_ocr_result(self, media_id, **kwargs):
        self.ocr_calls.append({"media_id": media_id, **kwargs})

    def upsert_caption_result(self, media_id, **kwargs):
        self.caption_calls.append({"media_id": media_id, **kwargs})

    def upsert_object_result(self, media_id, **kwargs):
        self.object_calls.append({"media_id": media_id, **kwargs})


class FakeUow:
    """Unit of Work giả, chỉ lộ ra `results`."""

    def __init__(self, repo: FakeResultsRepo) -> None:
        self.results = repo
        self.media = None

    def __call__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def repo() -> FakeResultsRepo:
    return FakeResultsRepo()


@pytest.fixture
def worker(repo: FakeResultsRepo) -> VideoProcessingWorker:
    return VideoProcessingWorker(
        config=AppConfig(),
        storage=object(),
        pipeline=object(),
        uow_factory=FakeUow(repo),
    )


def _bo_ket_qua(reason: str):
    """Bộ kết quả rỗng kèm lý do — hình dạng của một khâu không sinh ra gì."""
    return SimpleNamespace(results=[], status=StageStatus.SKIPPED, reason=reason)


# --------------------------------------------------------------------------
# Chính hàm guard
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reason, mong_doi",
    [
        ("detection disabled by route", False),
        ("OCR disabled by route", False),
        ("caption disabled by route", False),
        ("Detection Disabled By Route", False),  # không phân biệt hoa thường
        ("no detections after filtering", True),  # đã chạy, không thấy gì
        ("no visible text found", True),
        (None, True),
        ("", True),
    ],
)
def test_guard_chi_chan_dung_truong_hop_tat_theo_route(reason, mong_doi):
    assert khau_da_chay(reason) is mong_doi


# --------------------------------------------------------------------------
# VideoProcessingWorker — 3 chỗ
# --------------------------------------------------------------------------


def test_detection_tat_theo_route_khong_xoa_object_cu(worker, repo):
    """Đây là lỗi đã xoá 17.262 object. `objects=[]` khác `objects=None`."""
    frame_ids = {"f1": uuid4(), "f2": uuid4()}
    result = SimpleNamespace(detection_results=_bo_ket_qua("detection disabled by route"))

    worker._persist_detections(FakeUow(repo), result, frame_ids)

    assert repo.object_calls == []


def test_detection_da_chay_ma_khong_thay_gi_van_ghi_rong(worker, repo):
    """Ngược lại: chạy thật mà không thấy vật thì phải ghi [] để dữ liệu cũ hết hiệu lực."""
    frame_ids = {"f1": uuid4(), "f2": uuid4()}
    result = SimpleNamespace(detection_results=_bo_ket_qua("no detections after filtering"))

    worker._persist_detections(FakeUow(repo), result, frame_ids)

    assert len(repo.object_calls) == 2
    assert all(call["objects"] == [] for call in repo.object_calls)


def test_ocr_tat_theo_route_khong_de_len_noi_dung(worker, repo):
    """Không có guard thì 'OCR disabled by route' thành chính ocr_text."""
    frame_ids = {"f1": uuid4()}
    result = SimpleNamespace(ocr_results=_bo_ket_qua("OCR disabled by route"))

    worker._persist_ocr(FakeUow(repo), result, frame_ids)

    assert repo.ocr_calls == []


def test_caption_tat_theo_route_khong_de_len_noi_dung(worker, repo):
    frame_ids = {"f1": uuid4()}
    result = SimpleNamespace(caption_results=_bo_ket_qua("caption disabled by route"))

    worker._persist_captions(FakeUow(repo), result, frame_ids)

    assert repo.caption_calls == []


# --------------------------------------------------------------------------
# MediaTaskWorker — 3 chỗ còn lại, cùng khuôn nhưng mỗi lần một frame
# --------------------------------------------------------------------------


@pytest.fixture
def task_worker(repo: FakeResultsRepo) -> MediaTaskWorker:
    worker = object.__new__(MediaTaskWorker)
    worker.config = AppConfig()
    worker.uow_factory = FakeUow(repo)
    return worker


def test_media_task_worker_khong_ghi_de_khi_tat_theo_route(task_worker, repo):
    frame_media_id = uuid4()

    task_worker._persist_ocr(frame_media_id, _bo_ket_qua("OCR disabled by route"))
    task_worker._persist_caption(frame_media_id, _bo_ket_qua("caption disabled by route"))
    task_worker._persist_detection(frame_media_id, _bo_ket_qua("detection disabled by route"))

    assert repo.ocr_calls == []
    assert repo.caption_calls == []
    assert repo.object_calls == []


# --------------------------------------------------------------------------
# Ghi đè cùng loại thì ĐÚNG; ghi chẩn đoán lên ô nội dung thì SAI.
# --------------------------------------------------------------------------


def test_ocr_moi_van_de_len_ocr_cu_binh_thuong(worker, repo):
    """Hành vi mong đợi: chạy lại là nội dung mới thay nội dung cũ."""
    frame_id = uuid4()
    result = SimpleNamespace(
        ocr_results=SimpleNamespace(
            results=[SimpleNamespace(frame_id="f1", full_text="CHỮ MỚI", status=StageStatus.DONE, reason=None)],
            status=StageStatus.DONE,
            reason=None,
        )
    )

    worker._persist_ocr(FakeUow(repo), result, {"f1": frame_id})

    assert repo.ocr_calls[0]["text"] == "CHỮ MỚI"


def test_ocr_chay_va_khong_thay_chu_thi_ghi_rong_that(worker, repo):
    """Rỗng thật khác với 'đừng đụng': ảnh không còn chữ thì phải xoá chữ cũ."""
    result = SimpleNamespace(
        ocr_results=SimpleNamespace(
            results=[SimpleNamespace(frame_id="f1", full_text="", status=StageStatus.NOT_FOUND, reason="no visible text found")],
            status=StageStatus.DONE,
            reason=None,
        )
    )

    worker._persist_ocr(FakeUow(repo), result, {"f1": uuid4()})

    assert repo.ocr_calls[0]["text"] == ""


def test_khung_hinh_loi_thi_giu_nguyen_noi_dung_cu(worker, repo):
    """Lỗi mạng một lượt không được xoá kết quả tốt của lượt trước."""
    result = SimpleNamespace(
        ocr_results=SimpleNamespace(
            results=[
                SimpleNamespace(
                    frame_id="f1",
                    full_text="",  # mặc định của schema, KHÔNG phải 'không có chữ'
                    status=StageStatus.ERROR,
                    reason="Qwen vision analysis failed: timeout",
                )
            ],
            status=StageStatus.ERROR,
            reason=None,
        )
    )

    worker._persist_ocr(FakeUow(repo), result, {"f1": uuid4()})

    # Trạng thái vẫn ghi để lượt sau biết thử lại, nhưng nội dung thì không đụng.
    assert repo.ocr_calls[0]["text"] is None
    assert repo.ocr_calls[0]["status"] == StageStatus.ERROR.value


def test_cau_chan_doan_khong_bao_gio_thanh_caption(worker, repo):
    """'Qwen vision analysis failed: ...' từng đi thẳng vào payload Qdrant."""
    result = SimpleNamespace(
        caption_results=SimpleNamespace(
            results=[
                SimpleNamespace(
                    frame_id="f1",
                    caption_text="",
                    generation_meta={},
                    status=StageStatus.ERROR,
                    reason="Qwen vision analysis failed: 429 rate limit",
                )
            ],
            status=StageStatus.ERROR,
            reason=None,
        )
    )

    worker._persist_captions(FakeUow(repo), result, {"f1": uuid4()})

    ghi = repo.caption_calls[0]
    assert ghi["caption_text"] is None
    # Model/metadata đi cùng caption; không có caption mới thì cũng đừng đổi.
    assert ghi["caption_model"] is None
    assert ghi["vision_metadata"] is None
