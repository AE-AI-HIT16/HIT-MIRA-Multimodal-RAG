"""Adapter RunPod Serverless — TC-901 (giao thức hàng đợi ↔ giao thức Jina).

Toàn bộ chạy ngoại tuyến: HTTP là fake, đồng hồ là fake. Không job nào được gửi
tới RunPod trong lúc chạy test.
"""

from __future__ import annotations

import pytest

from src.rag_video_anh.embedding import embedding_service as embedding_service_module
from src.rag_video_anh.embedding.embedding_service import (
    ImageEmbeddingConfigurationError,
    ImageEmbeddingProviderFatalError,
    ImageEmbeddingService,
    ImageEmbeddingServiceError,
)
from src.rag_video_anh.embedding.runpod_transport import (
    DEFAULT_API_BASE_URL,
    RunPodEmbeddingTransport,
    build_runpod_embedding_service,
    job_urls,
    runpod_endpoint_url,
)

ENDPOINT = "z1zyigyd4vl2fy"
RUN_URL = f"{DEFAULT_API_BASE_URL}/{ENDPOINT}/run"
STATUS_URL = f"{DEFAULT_API_BASE_URL}/{ENDPOINT}/status/job-1"
VECTOR = [0.001] * 1024
BIEN_RUNPOD = ("RUNPOD_API_KEY", "JINA_RUNPOD_ENDPOINT_ID", "RUNPOD_API_BASE_URL", "EMBED_MAX_BATCH_SIZE")


class FakeHTTPError(Exception):
    def __init__(self, response: "FakeResponse") -> None:
        super().__init__(f"HTTP {response.status_code}")
        self.response = response


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: object | None = None) -> None:
        self.status_code = status_code
        self.payload = payload if payload is not None else {}
        self.headers: dict[str, str] = {}
        self.text = str(payload)

    def json(self) -> object:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise FakeHTTPError(self)


class FakeQueue:
    """Ghi lại mọi lời gọi và trả lần lượt các response đã dựng sẵn."""

    def __init__(self, submit: list[FakeResponse], status: list[FakeResponse] | None = None) -> None:
        self._submit = list(submit)
        self._status = list(status or [])
        self.posts: list[dict] = []
        self.gets: list[dict] = []

    def post(self, url: str, *, headers=None, json=None, timeout=None) -> FakeResponse:
        self.posts.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return self._submit[min(len(self.posts) - 1, len(self._submit) - 1)]

    def get(self, url: str, *, headers=None, timeout=None) -> FakeResponse:
        self.gets.append({"url": url, "headers": headers, "timeout": timeout})
        return self._status[min(len(self.gets) - 1, len(self._status) - 1)]


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def completed(vectors: list[list[float]] | None = None) -> FakeResponse:
    rows = vectors if vectors is not None else [VECTOR]
    return FakeResponse(
        200,
        {
            "status": "COMPLETED",
            "output": {
                "object": "list",
                "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(rows)],
                "usage": {"items": len(rows)},
            },
        },
    )


def dung_service(queue: FakeQueue, clock: FakeClock, *, job_timeout: float = 900.0, **kwargs):
    transport = RunPodEmbeddingTransport(
        job_timeout=job_timeout,
        http_post=queue.post,
        http_get=queue.get,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    return ImageEmbeddingService(
        api_key="rpa_test", base_url=RUN_URL, dimensions=1024, tokens_per_minute=0, http_post=transport, **kwargs
    )


# ---------------------------------------------------------------- URL job


@pytest.mark.parametrize(
    "cau_hinh",
    [
        f"{DEFAULT_API_BASE_URL}/{ENDPOINT}",
        f"{DEFAULT_API_BASE_URL}/{ENDPOINT}/",
        f"{DEFAULT_API_BASE_URL}/{ENDPOINT}/run",
        f"{DEFAULT_API_BASE_URL}/{ENDPOINT}/runsync",
    ],
)
def test_url_chep_tu_runpod_deu_dung(cau_hinh: str) -> None:
    """Chép URL kiểu nào từ trang RunPod cũng ra cùng một cặp (run, status)."""
    run_url, status_prefix = job_urls(cau_hinh)
    assert run_url == RUN_URL
    assert status_prefix == f"{DEFAULT_API_BASE_URL}/{ENDPOINT}/status"


def test_url_giao_thuc_jina_truc_tiep_bi_chan() -> None:
    """`/v1/embeddings` là Pod chứ không phải hàng đợi; bọc `input` vào là 422 khó đoán."""
    with pytest.raises(ImageEmbeddingConfigurationError, match="không phải hàng đợi RunPod"):
        job_urls("http://1.2.3.4:8100/v1/embeddings")


def test_url_rong_bi_chan() -> None:
    with pytest.raises(ImageEmbeddingConfigurationError):
        job_urls("")


# ---------------------------------------------------------------- đường thành công


def test_nhung_text_qua_hang_doi() -> None:
    clock = FakeClock()
    queue = FakeQueue(
        submit=[FakeResponse(200, {"id": "job-1", "status": "IN_QUEUE"})],
        status=[FakeResponse(200, {"status": "IN_PROGRESS"}), completed()],
    )
    vectors = dung_service(queue, clock).embed_texts(["sự kiện HIT Open Day"])

    assert vectors == [VECTOR]
    # Payload Jina được bọc nguyên vẹn trong `input`, không đổi tên trường nào.
    assert queue.posts[0]["url"] == RUN_URL
    assert queue.posts[0]["json"]["input"]["input"] == [{"text": "sự kiện HIT Open Day"}]
    assert queue.posts[0]["json"]["input"]["task"] == "retrieval.query"
    assert queue.posts[0]["json"]["input"]["dimensions"] == 1024
    # Khoá do client truyền xuống được chuyển tiếp; adapter không giữ khoá riêng.
    assert queue.posts[0]["headers"]["Authorization"] == "Bearer rpa_test"
    assert [g["url"] for g in queue.gets] == [STATUS_URL, STATUS_URL]


def test_runsync_tra_ket_qua_ngay_thi_khong_hoi_trang_thai() -> None:
    """`/runsync` có thể xong ngay; lúc đó hỏi thêm một vòng là thừa."""
    clock = FakeClock()
    queue = FakeQueue(submit=[completed()])
    vectors = dung_service(queue, clock).embed_texts(["xin chào"])

    assert vectors == [VECTOR]
    assert queue.gets == []


def test_moi_lo_anh_la_mot_job_rieng(tmp_path) -> None:
    """Client vẫn chia lô như cũ; adapter không gộp hay tách lô của nó."""
    anh = []
    for i in range(3):
        duong_dan = tmp_path / f"keyframe-{i}.jpg"
        duong_dan.write_bytes(b"\xff\xd8\xff\xdb-anh-%d" % i)
        anh.append(duong_dan)

    clock = FakeClock()
    queue = FakeQueue(submit=[completed([VECTOR, VECTOR]), completed([VECTOR])])
    # max_image_side=0 tắt thu nhỏ, để test nói về việc chia lô chứ không về Pillow.
    service = dung_service(queue, clock, image_batch_size=2, max_image_side=0)
    vectors = service.embed_images(anh)

    assert len(vectors) == 3
    assert len(queue.posts) == 2
    assert [len(p["json"]["input"]["input"]) for p in queue.posts] == [2, 1]
    assert all(set(item) == {"image"} for p in queue.posts for item in p["json"]["input"]["input"])


# ---------------------------------------------------------------- phân loại lỗi


def test_job_failed_khong_thu_lai(monkeypatch: pytest.MonkeyPatch) -> None:
    """RunPod đã tự chạy lại job khi worker chết; phần còn lại là sai hợp đồng."""
    monkeypatch.setattr(embedding_service_module.time, "sleep", lambda _: None)
    clock = FakeClock()
    queue = FakeQueue(
        submit=[FakeResponse(200, {"id": "job-1", "status": "IN_QUEUE"})],
        status=[FakeResponse(200, {"status": "FAILED", "error": "job.input sai schema"})],
    )
    with pytest.raises(ImageEmbeddingServiceError):
        dung_service(queue, clock).embed_texts(["xin chào"])

    assert len(queue.posts) == 1


def test_job_timed_out_duoc_thu_lai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lần đầu gánh cold start, lượt sau thì không — đây là ca thử lại có ích."""
    monkeypatch.setattr(embedding_service_module.time, "sleep", lambda _: None)
    clock = FakeClock()
    queue = FakeQueue(
        submit=[FakeResponse(200, {"id": "job-1", "status": "IN_QUEUE"})],
        status=[FakeResponse(200, {"status": "TIMED_OUT"})],
    )
    with pytest.raises(ImageEmbeddingServiceError):
        dung_service(queue, clock).embed_texts(["xin chào"])

    assert len(queue.posts) == 3


def test_job_cancelled_khong_thu_lai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Có người chủ động huỷ; gửi lại là đi ngược đúng ý định đó."""
    monkeypatch.setattr(embedding_service_module.time, "sleep", lambda _: None)
    clock = FakeClock()
    queue = FakeQueue(
        submit=[FakeResponse(200, {"id": "job-1", "status": "IN_QUEUE"})],
        status=[FakeResponse(200, {"status": "CANCELLED"})],
    )
    with pytest.raises(ImageEmbeddingServiceError):
        dung_service(queue, clock).embed_texts(["xin chào"])

    assert len(queue.posts) == 1


def test_sai_khoa_runpod_la_loi_vinh_vien() -> None:
    """401 phải nổi lên tận ngoài cùng, không bị nuốt thành 'lô này hỏng'."""
    clock = FakeClock()
    queue = FakeQueue(submit=[FakeResponse(401, {"error": "Unauthorized"})])
    with pytest.raises(ImageEmbeddingProviderFatalError):
        dung_service(queue, clock).embed_texts(["xin chào"])

    assert queue.gets == []


def test_completed_nhung_output_khong_phai_object(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embedding_service_module.time, "sleep", lambda _: None)
    clock = FakeClock()
    queue = FakeQueue(
        submit=[FakeResponse(200, {"id": "job-1", "status": "IN_QUEUE"})],
        status=[FakeResponse(200, {"status": "COMPLETED", "output": None})],
    )
    with pytest.raises(ImageEmbeddingServiceError):
        dung_service(queue, clock).embed_texts(["xin chào"])

    assert len(queue.posts) == 1


def test_status_loi_tam_thoi_thi_hoi_lai() -> None:
    """`/status` hỏng không có nghĩa là job hỏng."""
    clock = FakeClock()
    queue = FakeQueue(
        submit=[FakeResponse(200, {"id": "job-1", "status": "IN_QUEUE"})],
        status=[FakeResponse(502, {"error": "bad gateway"}), completed()],
    )
    assert dung_service(queue, clock).embed_texts(["xin chào"]) == [VECTOR]
    assert len(queue.gets) == 2


def test_qua_han_cho_khong_gui_lai_job_trung(monkeypatch: pytest.MonkeyPatch) -> None:
    """Job có thể vẫn đang chạy; gửi lại là xếp thêm một job trùng và trả tiền hai lần."""
    monkeypatch.setattr(embedding_service_module.time, "sleep", lambda _: None)
    clock = FakeClock()
    queue = FakeQueue(
        submit=[FakeResponse(200, {"id": "job-1", "status": "IN_QUEUE"})],
        status=[FakeResponse(200, {"status": "IN_PROGRESS"})],
    )
    with pytest.raises(ImageEmbeddingServiceError) as loi:
        dung_service(queue, clock, job_timeout=3.0).embed_texts(["xin chào"])

    assert len(queue.posts) == 1
    assert sum(clock.slept) <= 3.0
    assert isinstance(loi.value.__cause__, Exception)
    assert "job-1" in str(loi.value.__cause__)


def test_nhan_job_ma_khong_co_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embedding_service_module.time, "sleep", lambda _: None)
    clock = FakeClock()
    queue = FakeQueue(submit=[FakeResponse(200, {"status": "IN_QUEUE"})])
    with pytest.raises(ImageEmbeddingServiceError):
        dung_service(queue, clock).embed_texts(["xin chào"])


# ---------------------------------------------------------------- cấu hình


@pytest.fixture
def moi_truong_runpod_sach(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """`.env` thật của máy dev chảy vào pytest; gỡ ra để test nói về code."""
    for ten in BIEN_RUNPOD:
        monkeypatch.delenv(ten, raising=False)
    return monkeypatch


def test_thieu_endpoint_bao_ro_nham_bien_nao(moi_truong_runpod_sach: pytest.MonkeyPatch) -> None:
    moi_truong_runpod_sach.setenv("RUNPOD_API_KEY", "rpa_test")
    with pytest.raises(ImageEmbeddingConfigurationError, match="JINA_RUNPOD_ENDPOINT_ID"):
        build_runpod_embedding_service()


def test_thieu_khoa_runpod(moi_truong_runpod_sach: pytest.MonkeyPatch) -> None:
    moi_truong_runpod_sach.setenv("JINA_RUNPOD_ENDPOINT_ID", ENDPOINT)
    with pytest.raises(ImageEmbeddingConfigurationError, match="RUNPOD_API_KEY"):
        build_runpod_embedding_service()


def test_dung_service_tu_bien_moi_truong(moi_truong_runpod_sach: pytest.MonkeyPatch) -> None:
    moi_truong_runpod_sach.setenv("RUNPOD_API_KEY", "rpa_test")
    moi_truong_runpod_sach.setenv("JINA_RUNPOD_ENDPOINT_ID", ENDPOINT)
    service = build_runpod_embedding_service()

    assert service.base_url == RUN_URL
    assert service.api_key == "rpa_test"
    # Tự host không có hạn mức token/phút của api.jina.ai; giữ nhịp chỉ làm chậm.
    assert service.tokens_per_minute == 0
    # Worker từ chối job quá EMBED_MAX_BATCH_SIZE bằng lỗi hợp đồng.
    assert service.text_batch_size == 64


def test_endpoint_url_ton_trong_api_base_url(moi_truong_runpod_sach: pytest.MonkeyPatch) -> None:
    moi_truong_runpod_sach.setenv("RUNPOD_API_BASE_URL", "https://api.runpod.ai/v2/")
    assert runpod_endpoint_url(ENDPOINT) == RUN_URL
