"""Trích keyframe từ video (OFFLINE). [BR-201 · US-201.1 · T-10]

Input : video_path, media_asset_id
Output: list Frame(timestamp_sec, frame_path); timestamp tăng dần, sai số ≤0.5s
v1 dùng ffmpeg lấy mẫu đều theo `step_sec` (fallback của TransNetV2 theo shot — đẩy v2).
Edge  : video hỏng/không probe được → log, trả [] (KHÔNG chặn batch).
Pass  : tests/test_pipeline.py::test_frames_have_timestamp
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Frame:
    media_asset_id: int
    timestamp_sec: float
    frame_path: str


def _probe_duration(video_path: str) -> float | None:
    """Thời lượng video (giây) qua ffprobe; lỗi/codec lạ → None."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=60, check=True,
        )
        return float(out.stdout.strip())
    except (subprocess.SubprocessError, ValueError) as exc:
        log.warning("Không probe được thời lượng %s: %s", video_path, exc)
        return None


def _extract_one(video_path: str, t: float, out_path: Path) -> bool:
    """Lấy 1 frame tại giây t. Trả True nếu ghi được file."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", video_path,
             "-frames:v", "1", "-q:v", "2", str(out_path)],
            capture_output=True, timeout=60, check=True,
        )
        return out_path.exists() and out_path.stat().st_size > 0
    except subprocess.SubprocessError as exc:
        log.warning("Bỏ frame @%.3fs của %s: %s", t, video_path, exc)
        return False


def extract_keyframes(video_path: str, media_asset_id: int, *,
                      step_sec: float = 2.0, max_frames: int = 60,
                      out_dir: str = "./data/frames") -> list[Frame]:
    duration = _probe_duration(video_path)
    if duration is None or duration <= 0:
        log.warning("Bỏ video (không có thời lượng hợp lệ): %s", video_path)
        return []

    dst = Path(out_dir) / str(media_asset_id)
    dst.mkdir(parents=True, exist_ok=True)

    frames: list[Frame] = []
    i, t = 0, 0.0
    while t < duration and i < max_frames:
        out_path = dst / f"{i:04d}_{int(t * 1000)}ms.jpg"
        if _extract_one(video_path, t, out_path):
            frames.append(Frame(media_asset_id, round(t, 3), str(out_path)))
        i += 1
        t += step_sec
    return frames
