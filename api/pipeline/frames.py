"""Trích keyframe từ video (OFFLINE). [BR-201 · US-201.1]

Hợp đồng — sinh viên implement:
  Input : video_path, media_asset_id
  Output: list Frame(timestamp_sec, frame_path); timestamp tăng dần, sai số ≤0.5s
  Gợi ý : dùng ffmpeg (hoặc TransNetV2) tách shot → lấy keyframe đại diện.
  Edge  : video hỏng/codec lạ → log lỗi, bỏ qua, KHÔNG chặn cả batch.
  Pass  : tests/test_pipeline.py::test_frames_have_timestamp
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Frame:
    media_asset_id: int
    timestamp_sec: float
    frame_path: str


def extract_keyframes(video_path: str, media_asset_id: int) -> list[Frame]:
    raise NotImplementedError("US-201.1: sinh viên trích keyframe + timestamp")
