"""Interface ASR (HỢP ĐỒNG) + impl faster-whisper. [BR-208 · T-21]

PhoWhisper-large (VinAI) là SOTA WER tiếng Việt; chạy qua backend CTranslate2/
faster-whisper để có tốc độ + timestamp. `model_name` mặc định "large-v3"; trỏ
sang bản PhoWhisper đã convert CT2 khi có (giữ nguyên interface).
"""
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


class FasterWhisperASR(ASRModel):
    """faster-whisper (backend CTranslate2). Import nặng đặt trong __init__ (lazy)."""

    def __init__(self, model_name: str = "large-v3", *, device: str = "auto",
                 compute_type: str = "int8", language: str = "vi",
                 model: object | None = None) -> None:
        self.language = language
        if model is not None:  # inject fake khi test
            self._model = model
            return
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str) -> list[TranscriptSegment]:
        segments, _info = self._model.transcribe(
            audio_path, language=self.language, vad_filter=True,
        )
        return [
            TranscriptSegment(
                start_sec=round(s.start, 3),
                end_sec=round(s.end, 3),
                text=s.text.strip(),
                confidence=getattr(s, "avg_logprob", None),
            )
            for s in segments
        ]
