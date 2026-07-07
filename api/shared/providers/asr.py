"""Interface ASR (HỢP ĐỒNG cho sẵn — impl là bài của sinh viên). [BR-208]"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TranscriptSegment:
    start_sec: float
    end_sec: float
    text: str
    confidence: float | None = None


class ASRModel(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> list[TranscriptSegment]:
        """audio → list đoạn transcript cắt theo timestamp."""
        raise NotImplementedError


# TODO(sinh viên): impl bằng faster-whisper (model tiếng Việt).
