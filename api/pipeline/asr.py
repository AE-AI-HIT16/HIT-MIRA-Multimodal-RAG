"""Tách audio → ASR → transcript theo timestamp (OFFLINE). [BR-208 · US-208.1]

Hợp đồng — sinh viên implement:
  Input : video_path, media_asset_id
  Output: list TranscriptSegment (start/end/text); rỗng nếu video câm (không lỗi)
  Gợi ý : tách audio (ffmpeg) → providers.asr.ASRModel.transcribe(...)
  Edge  : nhiều người nói/tạp âm → best-effort + confidence thấp; ngôn ngữ lạ → log.
  Pass  : tests/test_pipeline.py::test_transcript_segments
"""
from __future__ import annotations

from shared.providers.asr import TranscriptSegment


def extract_transcript(video_path: str, media_asset_id: int) -> list[TranscriptSegment]:
    raise NotImplementedError("US-208.1: sinh viên tách audio + ASR theo timestamp")
