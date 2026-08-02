"""Automatic speech recognition component."""

from __future__ import annotations

import re
import subprocess
import tempfile
import unicodedata
import wave
from pathlib import Path
from typing import Any

from src.configuration import AppConfig
from src.log.logger import logger
from src.rag_video_anh.schemas import ASRRequest, StageStatus, TranscriptSegment, TranscriptSet

_ASR_BOILERPLATE_HALLUCINATION_PHRASES = (
    "hay subscribe",
    "subscribe cho kenh",
    "dang ky kenh",
    "ung ho kenh",
    "bam chuong",
    "nhan chuong",
    "like va share",
    "like share",
    "dung quen dang ky",
    "dung quen like",
    "cam on cac ban da xem",
    "khong bo lo nhung video",
    "nhung video hap dan",
)


class AsrService:
    """Transcribe media audio into time-aligned transcript segments."""

    def __init__(self, config: AppConfig | None = None, model: Any | None = None) -> None:
        self.config = config or AppConfig()
        self.pipeline_config = self.config.media_pipeline
        self.model_config = self.config.media_models
        self.model = model
        self._model_loaded = model is not None
        self._load_error: str | None = None

    def transcribe(self, request: ASRRequest) -> TranscriptSet:
        media_input = request.media_input
        validation_result = request.validation_result
        if validation_result is not None and validation_result.technical_profile is not None:
            if validation_result.technical_profile.audio_present is False:
                return TranscriptSet(
                    media_id=media_input.media_id,
                    status=StageStatus.SKIPPED,
                    reason="media does not contain an audio track",
                )

        media_path = self._resolve_media_path(media_input)
        if media_path is None:
            return TranscriptSet(
                media_id=media_input.media_id,
                status=StageStatus.ERROR,
                reason="media path is required for ASR",
            )

        model = self._load_model()
        if model is None:
            return TranscriptSet(
                media_id=media_input.media_id,
                status=StageStatus.SKIPPED,
                reason=self._load_error or "missing dependency 'sherpa-onnx'",
            )

        try:
            return self._transcribe_sherpa(model, request, media_path)
        except Exception as exc:
            return TranscriptSet(
                media_id=media_input.media_id,
                status=StageStatus.ERROR,
                reason=f"ASR failed: {exc.__class__.__name__}",
            )

    def _transcribe_sherpa(self, recognizer: Any, request: ASRRequest, media_path: Path) -> TranscriptSet:
        media_input = request.media_input
        language = request.asr_policy.get("language") or self.model_config.asr_language or media_input.language_hint
        raw_transcript_segments: list[TranscriptSegment] = []
        duration = 0.0
        with tempfile.TemporaryDirectory(prefix="hit-mira-sherpa-asr-") as tmp_dir:
            wav_path = Path(tmp_dir) / f"{media_input.media_id}.wav"
            self._extract_wav(media_path, wav_path)
            for index, samples, sample_rate, start_sec, end_sec in self._read_wave_chunks(wav_path):
                duration = max(duration, end_sec)
                stream = recognizer.create_stream()
                stream.accept_waveform(sample_rate, samples)
                recognizer.decode_streams([stream])
                text = str(getattr(getattr(stream, "result", None), "text", "")).strip()
                if text:
                    raw_transcript_segments.append(
                        self._timed_segment(media_input.media_id, index, text, start_sec, end_sec, language)
                    )

        segments, filtered_count = self._filter_hallucinated_segments(raw_transcript_segments)
        self._log_segment_count(media_input.media_id, segments, filtered_count)
        return TranscriptSet(
            media_id=media_input.media_id,
            segments=segments,
            language=language,
            transcription_meta={
                "backend": "sherpa_onnx",
                "raw_segment_count": len(raw_transcript_segments),
                "filtered_segment_count": filtered_count,
                "model": self._asr_model_name(),
                "provider": self.model_config.sherpa_provider,
                "encoder": self.model_config.sherpa_encoder_file,
                "decoder": self.model_config.sherpa_decoder_file,
                "joiner": self.model_config.sherpa_joiner_file,
                "tokens": self.model_config.sherpa_tokens_file,
                "num_threads": self.model_config.sherpa_num_threads,
                "sample_rate": self.model_config.sherpa_sample_rate,
                "feature_dim": self.model_config.sherpa_feature_dim,
                "decoding_method": self.model_config.sherpa_decoding_method,
                "chunk_duration_sec": self.model_config.sherpa_chunk_duration_sec,
                "duration": duration,
            },
            status=StageStatus.DONE if segments else StageStatus.NOT_FOUND,
            reason=self._transcript_reason(raw_transcript_segments, segments),
        )

    @staticmethod
    def _resolve_media_path(media_input: Any) -> Path | None:
        value = getattr(media_input, "media_path", None) or getattr(media_input, "source_ref", None)
        if not value:
            value = getattr(media_input, "metadata", {}).get("path")
        if not value:
            return None
        path = Path(str(value))
        return path if path.exists() else None

    def _filter_hallucinated_segments(
        self, segments: list[TranscriptSegment]
    ) -> tuple[list[TranscriptSegment], int]:
        filtered_segments = [segment for segment in segments if not self._is_boilerplate_hallucination(segment.text)]
        return filtered_segments, len(segments) - len(filtered_segments)

    def _log_segment_count(self, media_id: str, segments: list[TranscriptSegment], filtered_count: int) -> None:
        if filtered_count:
            logger.info(
                f"ASR filtered {filtered_count} boilerplate hallucination segment(s) "
                f"for media_id '{media_id}'"
            )
        logger.info(f"ASR produced {len(segments)} segment(s) for media_id '{media_id}'")

    @staticmethod
    def _transcript_reason(raw_segments: list[TranscriptSegment], segments: list[TranscriptSegment]) -> str | None:
        if segments:
            return None
        if raw_segments:
            return "transcript filtered by ASR hallucination gate"
        return "empty transcript"

    @staticmethod
    def _is_boilerplate_hallucination(text: str) -> bool:
        normalized = AsrService._normalize_transcript_text(text)
        if not normalized:
            return True
        return any(phrase in normalized for phrase in _ASR_BOILERPLATE_HALLUCINATION_PHRASES)

    @staticmethod
    def _normalize_transcript_text(text: str) -> str:
        ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        return re.sub(r"\s+", " ", ascii_text.lower()).strip()

    def _load_model(self) -> Any | None:
        if self._model_loaded:
            return self.model
        return self._load_sherpa_model()

    def _load_sherpa_model(self) -> Any | None:
        try:
            import sherpa_onnx
        except ImportError:
            self._model_loaded = True
            self._load_error = "missing dependency 'sherpa-onnx'"
            return None
        try:
            model_dir = self._resolve_sherpa_model_dir()
            encoder = self._required_model_file(model_dir, self.model_config.sherpa_encoder_file)
            decoder = self._required_model_file(model_dir, self.model_config.sherpa_decoder_file)
            joiner = self._required_model_file(model_dir, self.model_config.sherpa_joiner_file)
            tokens = self._required_model_file(model_dir, self.model_config.sherpa_tokens_file)
            self.model = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(encoder),
                decoder=str(decoder),
                joiner=str(joiner),
                tokens=str(tokens),
                num_threads=self.model_config.sherpa_num_threads,
                sample_rate=self.model_config.sherpa_sample_rate,
                feature_dim=self.model_config.sherpa_feature_dim,
                decoding_method=self.model_config.sherpa_decoding_method,
                debug=self.model_config.sherpa_debug,
                provider=self.model_config.sherpa_provider,
            )
            self._model_loaded = True
            return self.model
        except Exception as exc:
            self._model_loaded = True
            self._load_error = f"ASR model load failed: {exc.__class__.__name__}"
            return None

    def _resolve_sherpa_model_dir(self) -> Path:
        if self.model_config.sherpa_model_dir:
            return Path(self.model_config.sherpa_model_dir).expanduser().resolve()
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("missing dependency 'huggingface-hub'") from exc
        model_dir = snapshot_download(
            repo_id=self._asr_model_name(),
            revision=self.model_config.sherpa_revision,
            allow_patterns=[
                self.model_config.sherpa_encoder_file,
                self.model_config.sherpa_decoder_file,
                self.model_config.sherpa_joiner_file,
                self.model_config.sherpa_tokens_file,
                "README.md",
                "bpe.model",
            ],
        )
        return Path(model_dir)

    @staticmethod
    def _required_model_file(model_dir: Path, filename: str) -> Path:
        path = model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing sherpa model file: {path}")
        return path

    def _extract_wav(self, source_path: Path, target_path: Path) -> None:
        command = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(source_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(self.model_config.sherpa_sample_rate),
            "-sample_fmt",
            "s16",
            str(target_path),
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def _read_wave_chunks(self, wav_path: Path) -> list[tuple[int, Any, int, float, float]]:
        import numpy as np

        chunks = []
        with wave.open(str(wav_path), "rb") as wav_file:
            if wav_file.getnchannels() != 1:
                raise ValueError(f"ASR WAV must be mono, got {wav_file.getnchannels()} channels")
            if wav_file.getsampwidth() != 2:
                raise ValueError(f"ASR WAV must use 16-bit samples, got {wav_file.getsampwidth()} bytes")
            sample_rate = wav_file.getframerate()
            frames_per_chunk = max(1, int(float(self.model_config.sherpa_chunk_duration_sec) * sample_rate))
            index = 0
            frames_read = 0
            while True:
                samples = wav_file.readframes(frames_per_chunk)
                if not samples:
                    break
                num_samples = len(samples) // 2
                start_sec = float(frames_read) / float(sample_rate) if sample_rate else 0.0
                end_sec = float(frames_read + num_samples) / float(sample_rate) if sample_rate else start_sec
                frames_read += num_samples
                if end_sec - start_sec < 1.0:
                    continue
                samples_int16 = np.frombuffer(samples, dtype=np.int16)
                samples_float32 = samples_int16.astype(np.float32) / 32768.0
                chunks.append((index, samples_float32, sample_rate, start_sec, end_sec))
                index += 1
        return chunks

    @staticmethod
    def _timed_segment(
        media_id: str,
        index: int,
        text: str,
        start_sec: float,
        end_sec: float,
        language: str | None,
    ) -> TranscriptSegment:
        return TranscriptSegment(
            segment_id=f"{media_id}_s{index:06d}",
            start_ms=int(start_sec * 1000),
            end_ms=int(end_sec * 1000),
            start_sec=start_sec,
            end_sec=end_sec,
            text=text,
            confidence=None,
            language=language,
        )

    def _asr_model_name(self) -> str:
        return str(self.model_config.asr_model_name)
