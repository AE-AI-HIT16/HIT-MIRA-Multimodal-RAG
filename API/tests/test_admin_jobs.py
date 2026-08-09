"""Job nền và các endpoint admin ghi — T-62 / P4-3 ③, US-602.1.

Chạy offline: `JobRunner` nhận một `popen` giả nên không có tiến trình con nào
được sinh ra, và các endpoint dùng `dependency_overrides` nên không cần DB,
Qdrant hay tài khoản admin thật.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from src.jobs.runner import DANG_CHAY, LOI, XONG, JobDangChayError, JobRunner
from src.routers.admin import lay_job_runner
from src.routers.auth import yeu_cau_admin
from src.server import app

LENH = {"media": ["python", "index_anh.py"], "videos": ["python", "index_video.py"], "eval": ["python", "eval.py"]}


class FakeProcess:
    """Tiến trình giả: trả sẵn vài dòng log rồi kết thúc với mã cho trước."""

    def __init__(self, dong_log: list[str], returncode: int = 0) -> None:
        self.stdout = iter(f"{dong}\n" for dong in dong_log)
        self._returncode = returncode

    def wait(self) -> int:
        return self._returncode


def runner_gia(dong_log=None, returncode: int = 0, popen=None) -> JobRunner:
    def _popen(lenh, **_kwargs):
        return FakeProcess(dong_log if dong_log is not None else ["đang chạy", "xong"], returncode)

    return JobRunner(LENH, cwd="/tmp", popen=popen or _popen)


def cho_job_xong(runner: JobRunner, target: str, timeout: float = 2.0) -> None:
    """Luồng theo dõi chạy nền — đợi nó ghi xong trạng thái."""
    het_han = time.monotonic() + timeout
    while time.monotonic() < het_han:
        if runner.status(target).state != DANG_CHAY:
            return
        time.sleep(0.01)
    raise AssertionError(f"job '{target}' không kết thúc trong {timeout}s")


# ── JobRunner ─────────────────────────────────────────────────────────────────


def test_job_moi_bat_dau_o_trang_thai_chua_chay() -> None:
    runner = runner_gia()

    assert [t.state for t in runner.status_all()] == ["idle", "idle", "idle"]


def test_job_chay_xong_ghi_ma_thoat_va_giu_duoi_log() -> None:
    runner = runner_gia(dong_log=["bắt đầu", "index 10 ảnh", "hoàn tất"])

    runner.start("media")
    cho_job_xong(runner, "media")

    trang_thai = runner.status("media")
    assert trang_thai.state == XONG
    assert trang_thai.returncode == 0
    assert "hoàn tất" in trang_thai.log_tail
    assert trang_thai.started_at is not None


def test_ma_thoat_khac_0_la_that_bai_chu_khong_phai_xong() -> None:
    """Script index trả mã lỗi khi có video hỏng — không được báo thành 'xong'."""
    runner = runner_gia(dong_log=["! hỏng"], returncode=1)

    runner.start("videos")
    cho_job_xong(runner, "videos")

    assert runner.status("videos").state == LOI
    assert runner.status("videos").returncode == 1


def test_khong_khoi_chay_duoc_thi_bao_loi_chu_khong_treo_o_dang_chay() -> None:
    def _no(*_args, **_kwargs):
        raise FileNotFoundError("không có python")

    runner = runner_gia(popen=_no)

    trang_thai = runner.start("media")

    assert trang_thai.state == LOI
    assert "FileNotFoundError" in trang_thai.log_tail


def test_khong_chay_chong_hai_job_cung_target() -> None:
    """Bấm nút hai lần không được sinh hai tiến trình cùng ghi vào Qdrant."""
    cho_phep_ket_thuc = False

    class TienTrinhTreo:
        stdout = None

        def wait(self):
            while not cho_phep_ket_thuc:
                time.sleep(0.01)
            return 0

    runner = JobRunner(LENH, cwd="/tmp", popen=lambda *a, **k: TienTrinhTreo())
    runner.start("media")

    with pytest.raises(JobDangChayError):
        runner.start("media")

    # Target khác vẫn chạy được — hai nhánh index độc lập nhau.
    assert runner.start("videos").state == DANG_CHAY
    cho_phep_ket_thuc = True


def test_giu_dung_so_dong_log_cuoi() -> None:
    runner = JobRunner(
        LENH,
        cwd="/tmp",
        so_dong_log=3,
        popen=lambda *a, **k: FakeProcess([f"dòng {i}" for i in range(10)]),
    )

    runner.start("media")
    cho_job_xong(runner, "media")

    assert runner.status("media").log_tail.splitlines() == ["dòng 7", "dòng 8", "dòng 9"]


# ── Endpoint ──────────────────────────────────────────────────────────────────


@pytest.fixture
def client():
    app.dependency_overrides[yeu_cau_admin] = lambda: {"email": "admin@hit", "role": "admin"}
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def dat_runner(runner: JobRunner) -> None:
    app.dependency_overrides[lay_job_runner] = lambda: runner


def test_cac_route_admin_ghi_da_dang_ky() -> None:
    duong_dan = {route.path for route in app.routes}
    assert "/api/admin/index/{target}" in duong_dan
    assert "/api/admin/index/status" in duong_dan
    assert "/api/admin/eval/run" in duong_dan


def test_chay_index_tra_trang_thai_job(client) -> None:
    runner = runner_gia()
    dat_runner(runner)

    body = client.post("/api/admin/index/media").json()

    assert body["target"] == "media"
    assert body["state"] in (DANG_CHAY, XONG)
    cho_job_xong(runner, "media")


def test_index_noi_quy_tra_422_kem_ly_do_chu_khong_im_lang(client) -> None:
    """Nội quy được nhúng ngay lúc nạp — nút này không có việc để làm."""
    dat_runner(runner_gia())

    phan_hoi = client.post("/api/admin/index/regulations")

    assert phan_hoi.status_code == 422
    assert "documents/upload" in phan_hoi.json()["detail"]


def test_index_target_la_404(client) -> None:
    dat_runner(runner_gia())

    phan_hoi = client.post("/api/admin/index/khong-ton-tai")

    assert phan_hoi.status_code == 404


def test_bam_lai_khi_dang_chay_tra_409_chu_khong_500(client) -> None:
    class TienTrinhTreo:
        stdout = None

        def wait(self):
            time.sleep(5)
            return 0

    dat_runner(JobRunner(LENH, cwd="/tmp", popen=lambda *a, **k: TienTrinhTreo()))

    assert client.post("/api/admin/index/media").status_code == 200
    assert client.post("/api/admin/index/media").status_code == 409


def test_trang_thai_index_liet_ke_moi_job_ke_ca_eval(client) -> None:
    dat_runner(runner_gia())

    body = client.get("/api/admin/index/status").json()

    assert {job["target"] for job in body} == {"media", "videos", "eval"}
    assert all({"state", "returncode", "started_at", "log_tail"} <= set(job) for job in body)


def test_chay_danh_gia_tra_trang_thai_job_khong_phai_bao_cao(client) -> None:
    """Đánh giá mất hàng chục giây — trả job để web hỏi lại, không treo request."""
    runner = runner_gia()
    dat_runner(runner)

    body = client.post("/api/admin/eval/run").json()

    assert body["target"] == "eval"
    assert "recall" not in body
    cho_job_xong(runner, "eval")


def test_endpoint_admin_ghi_deu_dung_sau_quyen_admin() -> None:
    """Bỏ quên guard ở một endpoint là mở lại đúng lỗ chạy lệnh từ xa."""
    app.dependency_overrides.clear()
    dat_runner(runner_gia())
    with TestClient(app) as khach:
        for phuong_thuc, duong_dan in (
            ("post", "/api/admin/index/media"),
            ("get", "/api/admin/index/status"),
            ("post", "/api/admin/eval/run"),
            ("get", "/api/admin/stats"),
            ("get", "/api/admin/eval/report"),
        ):
            phan_hoi = getattr(khach, phuong_thuc)(duong_dan)
            assert phan_hoi.status_code in (401, 403), f"{duong_dan} không yêu cầu đăng nhập"
    app.dependency_overrides.clear()
