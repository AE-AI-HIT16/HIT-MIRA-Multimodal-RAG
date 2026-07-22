"""Optical character recognition component."""

from __future__ import annotations

import os
from statistics import mean
from typing import Any

from src.configuration import AppConfig
from src.log.logger import logger
from src.rag_video_anh.schemas import OCRRequest, OCRResult, OCRResultSet, StageStatus, TextSpan


class OCRService:
    """Extract text from each keyframe."""

    def __init__(self, config: AppConfig | None = None, reader: Any | None = None) -> None:
        self.config = config or AppConfig()
        self.pipeline_config = self.config.media_pipeline
        self.model_config = self.config.media_models
        self.backend = os.getenv("OCR_BACKEND", "paddleocr").strip().lower()
        self._reader = reader
        self._reader_loaded = reader is not None

    def recognize(self, request: OCRRequest) -> OCRResultSet:
        if not request.keyframes.frames:
            return OCRResultSet(
                media_id=request.media_id,
                status=StageStatus.SKIPPED,
                reason="no keyframes available for OCR",
            )

        reader = self._load_reader()
        if reader is None:
            return OCRResultSet(
                media_id=request.media_id,
                status=StageStatus.SKIPPED,
                reason=f"missing dependency '{self.backend}'",
            )

        results: list[OCRResult] = []
        for frame in request.keyframes.frames:
            try:
                image = frame.image_payload if frame.image_payload is not None else frame.image_path
                spans = self._read_frame(reader, image)
                full_text = " ".join(span.text for span in spans if span.text).strip()
                confidences = [span.confidence for span in spans if span.confidence is not None]
                results.append(
                    OCRResult(
                        frame_id=frame.frame_id,
                        text_spans=spans,
                        full_text=full_text,
                        confidence=mean(confidences) if confidences else None,
                        language_hints=self.model_config.ocr_languages,
                        status=StageStatus.DONE if full_text else StageStatus.NOT_FOUND,
                        reason=None if full_text else "no text regions found",
                    )
                )
            except Exception as exc:
                results.append(
                    OCRResult(
                        frame_id=frame.frame_id,
                        status=StageStatus.ERROR,
                        reason=f"OCR failed: {exc.__class__.__name__}",
                    )
                )

        status = StageStatus.DONE if all(result.status != StageStatus.ERROR for result in results) else StageStatus.ERROR
        logger.info(f"OCR processed {len(results)} frame(s) for media_id '{request.media_id}'")
        return OCRResultSet(media_id=request.media_id, results=results, status=status)

    def _load_reader(self) -> Any | None:
        if self._reader_loaded:
            return self._reader
        if self.backend == "easyocr":
            return self._load_easyocr_reader()
        return self._load_paddleocr_reader()

    def _load_easyocr_reader(self) -> Any | None:
        try:
            import easyocr
        except ImportError:
            self._reader_loaded = True
            self._reader = None
            return None

        languages = self.model_config.ocr_languages or ["vi", "en"]
        self._reader = easyocr.Reader(languages, gpu=self.model_config.ocr_gpu)
        self._reader_loaded = True
        return self._reader

    def _load_paddleocr_reader(self) -> Any | None:
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            self._reader_loaded = True
            self._reader = None
            return None

        lang = self.model_config.ocr_languages[0] if self.model_config.ocr_languages else "en"
        self._reader = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=self.model_config.ocr_gpu)
        self._reader_loaded = True
        return self._reader

    def _read_frame(self, reader: Any, image: Any) -> list[TextSpan]:
        if image is None:
            raise ValueError("keyframe image is not available")
        if self.backend == "easyocr":
            return self._read_frame_easyocr(reader, image)
        return self._read_frame_paddleocr(reader, image)

    
    @staticmethod
    def _read_frame_easyocr(reader: Any, image: Any) -> list[TextSpan]:
        spans: list[TextSpan] = []
        raw_results = reader.readtext(image)
        for row in raw_results or []:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            box = row[0]
            text = str(row[1]).strip()
            confidence = float(row[2]) if len(row) > 2 and row[2] is not None else None
            if text:
                spans.append(TextSpan(text=text, confidence=confidence, bbox=box))
        return spans

    
    @staticmethod
    def _read_frame_paddleocr(reader: Any, image: Any) -> list[TextSpan]:
        raw_results = reader.ocr(image, cls=True)
        spans: list[TextSpan] = []
        for block in raw_results or []:
            rows = block if isinstance(block, list) else []
            for row in rows:
                if not isinstance(row, (list, tuple)) or len(row) < 2:
                    continue
                box = row[0]
                text_meta = row[1]
                if not isinstance(text_meta, (list, tuple)) or not text_meta:
                    continue
                text = str(text_meta[0]).strip()
                confidence = float(text_meta[1]) if len(text_meta) > 1 and text_meta[1] is not None else None
                if text:
                    spans.append(TextSpan(text=text, confidence=confidence, bbox=box))
        return spans
