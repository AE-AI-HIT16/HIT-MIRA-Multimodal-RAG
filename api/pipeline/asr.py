"""Tách audio → ASR → transcript theo timestamp + chunk (OFFLINE). [BR-208 · US-208.1 · T-11]

  extract_transcript : video → list TranscriptSegment (rỗng nếu câm/không audio, KHÔNG lỗi)
  chunk_transcript   : gộp segment thành chunk ~200–300 token, overlap ~15%, giữ timestamp
                       (đúng spec chunking ở docs/tech-pipeline.md §2)
  Gợi ý : tách audio (ffmpeg) → providers.asr.ASRModel.transcribe(...)
  Pass  : tests/test_pipeline.py::test_transcript_segments / test_chunk_transcript
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from shared.providers.asr import TranscriptSegment

log = logging.getLogger(__name__)


def _extract_audio(video_path: str, out_path: Path) -> Path | None:
    """Tách audio mono 16kHz WAV (chuẩn Whisper). Video câm/không audio → None."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
             "-f", "wav", str(out_path)],
            capture_output=True, timeout=300, check=True,
        )
        return out_path if out_path.exists() and out_path.stat().st_size > 0 else None
    except subprocess.SubprocessError as exc:
        log.warning("Không tách được audio từ %s (có thể video câm): %s", video_path, exc)
        return None


def extract_transcript(video_path: str, media_asset_id: int, *,
                       asr: Any | None = None,
                       out_dir: str = "./data/audio") -> list[TranscriptSegment]:
    if asr is None:
        from app.deps import get_asr

        asr = get_asr()

    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    audio = _extract_audio(video_path, dst / f"{media_asset_id}.wav")
    if audio is None:
        return []
    return asr.transcribe(str(audio))


def chunk_transcript(segments: list[TranscriptSegment], media_asset_id: int, *,
                     max_chars: int = 800, overlap_chars: int = 120) -> list[dict[str, Any]]:
    """Gộp segment thành chunk (~200–300 token ≈ 800 ký tự TV), overlap ~15%, giữ timestamp.

    Trả payload sẵn sàng cho embed_texts + build_index collection transcript.
    """
    def _mk(buf: list[TranscriptSegment]) -> dict[str, Any]:
        return {
            "video_id": media_asset_id,
            "start_sec": round(buf[0].start_sec, 3),
            "end_sec": round(buf[-1].end_sec, 3),
            "text": " ".join(s.text for s in buf).strip(),
        }

    chunks: list[dict[str, Any]] = []
    buf: list[TranscriptSegment] = []
    buf_len = 0
    emitted_upto = -1                      # chỉ số segment cuối đã nằm trong 1 chunk đầy
    for idx, seg in enumerate(segments):
        buf.append(seg)
        buf_len += len(seg.text)
        if buf_len >= max_chars:
            chunks.append(_mk(buf))
            emitted_upto = idx
            keep: list[TranscriptSegment] = []
            klen = 0
            for s in reversed(buf):        # giữ đuôi làm overlap
                if klen >= overlap_chars:
                    break
                keep.insert(0, s)
                klen += len(s.text)
            buf, buf_len = keep, klen
    # còn dư chứa nội dung MỚI (chưa flush) → chunk cuối; nếu buf chỉ là overlap thì bỏ
    if buf and emitted_upto < len(segments) - 1:
        chunks.append(_mk(buf))
    return chunks
