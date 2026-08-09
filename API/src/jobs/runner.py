"""Chạy script dài trong tiến trình con và theo dõi trạng thái — T-62 (P4-3 ③).

Vì sao là tiến trình con chứ không phải một `async` task: index và đánh giá gọi
API nhúng hàng nghìn lần và mất hàng chục phút. NFR nói không đặt việc nặng lên
request path; chạy trong chính tiến trình API thì một lượt index sẽ ăn hết
threadpool và làm cả chat lẫn tìm kiếm chậm theo. Tiến trình con còn giữ đúng
những `--apply` đã có sẵn trong `scripts/`, không phải viết lại logic index lần
thứ hai — hai bản sao sẽ trôi khác nhau trong vài tuần.

**Trạng thái nằm trong RAM của một tiến trình.** Chạy nhiều worker uvicorn hoặc
`--reload` thì mỗi tiến trình thấy một bảng job khác nhau, và job đang chạy sẽ
biến mất khỏi danh sách sau khi reload. Chấp nhận được cho màn admin một người
dùng; muốn bền thì phải đẩy trạng thái xuống PostgreSQL — chưa cần cho v1.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.log.logger import logger

CHUA_CHAY = "idle"
DANG_CHAY = "running"
XONG = "done"
LOI = "failed"

SO_DONG_LOG_GIU = 40


@dataclass
class TrangThaiJob:
    """Đúng hình dạng `IndexJobStatus` mà `web/lib/types.ts` đang chờ."""

    target: str
    state: str = CHUA_CHAY
    returncode: int | None = None
    started_at: float | None = None
    log_tail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "state": self.state,
            "returncode": self.returncode,
            "started_at": self.started_at,
            "log_tail": self.log_tail,
        }


class JobDangChayError(RuntimeError):
    """Job của target này đang chạy — không được chạy chồng lên nhau."""


class JobRunner:
    """Bảng job theo target, mỗi target nhiều nhất một tiến trình đang chạy."""

    def __init__(
        self,
        lenh: dict[str, list[str]],
        *,
        cwd: Path,
        so_dong_log: int = SO_DONG_LOG_GIU,
        popen: Any = None,
    ) -> None:
        self.lenh = lenh
        self.cwd = Path(cwd)
        self.so_dong_log = so_dong_log
        self._popen = popen or subprocess.Popen
        self._khoa = threading.Lock()
        self._trang_thai: dict[str, TrangThaiJob] = {
            target: TrangThaiJob(target=target) for target in lenh
        }

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(self.lenh)

    def start(self, target: str) -> TrangThaiJob:
        if target not in self.lenh:
            raise KeyError(target)

        with self._khoa:
            hien_tai = self._trang_thai[target]
            if hien_tai.state == DANG_CHAY:
                raise JobDangChayError(f"Job '{target}' đang chạy")
            moi = TrangThaiJob(target=target, state=DANG_CHAY, started_at=time.time())
            self._trang_thai[target] = moi

        lenh = list(self.lenh[target])
        logger.info(f"Bắt đầu job '{target}': {' '.join(lenh)}")
        try:
            tien_trinh = self._popen(
                lenh,
                cwd=str(self.cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:  # noqa: BLE001 — lệnh sai/thiếu python phải hiện ra ở UI
            with self._khoa:
                moi.state = LOI
                moi.returncode = -1
                moi.log_tail = f"Không khởi chạy được: {exc.__class__.__name__}: {exc}"
                self._trang_thai[target] = moi
            logger.warning(f"Job '{target}' không khởi chạy được: {exc}")
            return self.status(target)

        luong = threading.Thread(
            target=self._theo_doi, args=(target, moi, tien_trinh), daemon=True
        )
        luong.start()
        return self.status(target)

    def _theo_doi(self, target: str, trang_thai: TrangThaiJob, tien_trinh: Any) -> None:
        """Đọc log tới hết rồi ghi kết quả. Chạy trong luồng riêng.

        Phải đọc stdout liên tục chứ không `wait()` rồi mới đọc: pipe đầy là
        tiến trình con bị chặn khi ghi, và một job in nhiều log sẽ treo vĩnh
        viễn ở khoảng 64KB đầu tiên.
        """
        duoi_log: deque[str] = deque(maxlen=self.so_dong_log)
        try:
            if tien_trinh.stdout is not None:
                for dong in tien_trinh.stdout:
                    duoi_log.append(dong.rstrip("\n"))
                    with self._khoa:
                        trang_thai.log_tail = "\n".join(duoi_log)
            ma_thoat = tien_trinh.wait()
        except Exception as exc:  # noqa: BLE001
            ma_thoat = -1
            duoi_log.append(f"Lỗi khi theo dõi tiến trình: {exc.__class__.__name__}: {exc}")

        with self._khoa:
            trang_thai.returncode = ma_thoat
            trang_thai.state = XONG if ma_thoat == 0 else LOI
            trang_thai.log_tail = "\n".join(duoi_log)
        logger.info(f"Job '{target}' kết thúc, mã thoát {ma_thoat}")

    def status(self, target: str) -> TrangThaiJob:
        with self._khoa:
            goc = self._trang_thai[target]
            return TrangThaiJob(**goc.to_dict())

    def status_all(self) -> list[TrangThaiJob]:
        return [self.status(target) for target in self.lenh]
