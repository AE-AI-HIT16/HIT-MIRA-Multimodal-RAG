"""Optical character recognition component."""

from __future__ import annotations

import re
import unicodedata
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
                reason="missing dependency 'paddleocr'",
            )

        results: list[OCRResult] = []
        for frame in request.keyframes.frames:
            try:
                image = frame.image_payload if frame.image_payload is not None else frame.image_path
                spans = self._read_frame(reader, image)
                joined_text = " ".join(span.text for span in spans if span.text)
                full_text = (
                    self._clean_text(joined_text)
                    if getattr(self.pipeline_config, "ocr_text_cleanup", True)
                    else self._normalize_full_text(joined_text)
                )
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

        best_spans: list[TextSpan] = []
        best_score: tuple[float, float, int] | None = None
        for variant_name, variant_image in self._image_variants(image):
            raw_results = self._run_ocr(reader, variant_image)
            spans = self._parse_results(raw_results, variant_name)
            spans = self._filter_and_clean_spans(spans)
            score = self._score_spans(spans)
            if best_score is None or score > best_score:
                best_score = score
                best_spans = spans

        return best_spans

    @staticmethod
    def _run_ocr(reader: Any, image: Any) -> Any:
        ocr = getattr(reader, "ocr", None)
        if callable(ocr):
            try:
                return ocr(image, cls=True)
            except TypeError:
                return ocr(image)

        predict = getattr(reader, "predict", None)
        if callable(predict):
            return predict(image)

        raise AttributeError("PaddleOCR reader exposes neither 'ocr' nor 'predict'")

    def _image_variants(self, image: Any) -> list[tuple[str, Any]]:
        variants: list[tuple[str, Any]] = [("original", image)]
        if not getattr(self.pipeline_config, "ocr_enable_preprocessing", True):
            return variants

        cv_image = self._load_cv_image(image)
        if cv_image is None:
            return variants

        try:
            import cv2
        except ImportError:
            return variants

        factors = getattr(self.pipeline_config, "ocr_upscale_factors", [1.0, 2.0]) or [1.0]
        normalized_factors = []
        for factor in factors:
            try:
                value = float(factor)
            except (TypeError, ValueError):
                continue
            if value > 1.0 and value not in normalized_factors:
                normalized_factors.append(value)

        for factor in normalized_factors:
            width = max(1, int(cv_image.shape[1] * factor))
            height = max(1, int(cv_image.shape[0] * factor))
            upscaled = cv2.resize(cv_image, (width, height), interpolation=cv2.INTER_CUBIC)
            variants.append((f"upscale_{factor:g}x", upscaled))

            enhanced = self._enhance_for_text(upscaled)
            if enhanced is not None:
                variants.append((f"upscale_{factor:g}x_clahe_sharp", enhanced))

        if len(variants) == 1:
            enhanced = self._enhance_for_text(cv_image)
            if enhanced is not None:
                variants.append(("clahe_sharp", enhanced))

        return variants

    @staticmethod
    def _load_cv_image(image: Any) -> Any | None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            return None

        if isinstance(image, str):
            return cv2.imread(image)
        if isinstance(image, np.ndarray):
            return image
        return None

    @staticmethod
    def _enhance_for_text(image: Any) -> Any | None:
        try:
            import cv2
        except ImportError:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        sharpened = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        sharpened = cv2.addWeighted(enhanced, 1.6, sharpened, -0.6, 0)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    def _parse_results(self, raw_results: Any, variant_name: str) -> list[TextSpan]:
        mapping_spans = self._parse_mapping_results(raw_results, variant_name)
        if mapping_spans:
            return mapping_spans

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
                    spans.append(
                        TextSpan(
                            text=text,
                            confidence=confidence,
                            bbox=self._jsonable_bbox(box),
                            metadata={"ocr_variant": variant_name},
                        )
                    )
        return spans

    def _parse_mapping_results(self, raw_results: Any, variant_name: str) -> list[TextSpan]:
        candidates = raw_results if isinstance(raw_results, list) else [raw_results]
        spans: list[TextSpan] = []
        for candidate in candidates:
            if not hasattr(candidate, "get"):
                continue

            texts = candidate.get("rec_texts") or candidate.get("texts") or candidate.get("text") or []
            scores = candidate.get("rec_scores") or candidate.get("scores") or candidate.get("confidences") or []
            boxes = candidate.get("rec_polys") or candidate.get("dt_polys") or candidate.get("boxes") or []
            if isinstance(texts, str):
                texts = [texts]

            for index, item in enumerate(texts):
                text = str(item).strip()
                if not text:
                    continue
                confidence = None
                if index < len(scores) and scores[index] is not None:
                    confidence = float(scores[index])
                bbox = boxes[index] if index < len(boxes) else None
                spans.append(
                    TextSpan(
                        text=text,
                        confidence=confidence,
                        bbox=self._jsonable_bbox(bbox),
                        metadata={"ocr_variant": variant_name},
                    )
                )
        return spans

    @staticmethod
    def _jsonable_bbox(box: Any) -> Any:
        if hasattr(box, "tolist"):
            box = box.tolist()
        if isinstance(box, list) and len(box) >= 4 and all(isinstance(item, (int, float)) for item in box[:4]):
            x1, y1, x2, y2 = (float(item) for item in box[:4])
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        return box

    def _filter_and_clean_spans(self, spans: list[TextSpan]) -> list[TextSpan]:
        min_confidence = float(getattr(self.pipeline_config, "ocr_min_confidence", 0.15) or 0.0)
        enable_cleanup = bool(getattr(self.pipeline_config, "ocr_text_cleanup", True))
        cleaned_spans: list[TextSpan] = []

        for span in spans:
            text = span.text.strip()
            if not text:
                continue
            if span.confidence is not None and span.confidence < min_confidence and self._looks_like_noise(text):
                continue

            cleaned_text = self._clean_text(text) if enable_cleanup else self._normalize_full_text(text)
            if not cleaned_text:
                continue

            metadata = dict(span.metadata)
            if cleaned_text != text:
                metadata["raw_text"] = text
                metadata["cleanup"] = "vietnamese_ocr_v1"
            cleaned_spans.append(
                TextSpan(
                    text=cleaned_text,
                    confidence=span.confidence,
                    bbox=span.bbox,
                    metadata=metadata,
                )
            )
        return cleaned_spans

    @staticmethod
    def _score_spans(spans: list[TextSpan]) -> tuple[float, float, int]:
        if not spans:
            return (0.0, 0.0, 0)
        confidences = [span.confidence for span in spans if span.confidence is not None]
        average_confidence = mean(confidences) if confidences else 0.0
        text_length = sum(len(span.text) for span in spans)
        span_count = len(spans)
        length_score = min(text_length / 120.0, 1.0)
        combined_score = (average_confidence * 0.7) + (length_score * 0.3)
        return (combined_score, average_confidence, text_length + span_count)

    @staticmethod
    def _looks_like_noise(text: str) -> bool:
        alnum_count = sum(char.isalnum() for char in text)
        if alnum_count == 0:
            return True
        junk_count = sum(char in "_|/\\~`^*#=+<>" for char in text)
        return junk_count > alnum_count

    @classmethod
    def _clean_text(cls, text: str) -> str:
        normalized = cls._normalize_full_text(text)
        replacements = [
            (r"(?i)\bCNTT\b", "CNTT"),
            (r"(?i)0[oô][-\s]*h[oọộaàáảãạ]+", "đồ họa"),
            (r"(?i)ph[ẳăaảãạầấẩẫậằắẳẵặ]*[np]?\s*m[ễeềếểễệ]m", "phần mềm"),
            (r"(?i)ngh[êeềếểễệ]\s*th[iíìỉĩị]\s*[eêềếểễệ]t\s*k[eếềểễệ]", "nghề thiết kế"),
            (r"(?i)gi[aảãàáạ]i\s+k[l1i]uy[eêềếểễệ]n", "giải khuyến"),
            (r"(?i)kh[iíìỉĩị]ch", "khích"),
            (r"(?i)g[l1i]y\s*k[iíìỉĩị][i1l]en", "kỷ niệm"),
            (r"(?i)nche\s*tank", "nghệ thành"),
        ]
        for pattern, replacement in replacements:
            normalized = re.sub(pattern, replacement, normalized)
        normalized = re.sub(r"\bgiai\s+(giải khuyến)\b", r"\1", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\b(giải khuyến)\s+(khích)\b", r"\1 \2", normalized, flags=re.IGNORECASE)
        return cls._normalize_full_text(normalized)

    @staticmethod
    def _normalize_full_text(text: str) -> str:
        normalized = unicodedata.normalize("NFC", text or "")
        normalized = normalized.replace(" ", " ")
        normalized = re.sub(r"[_]{2,}", " ", normalized)
        normalized = re.sub(r"\s+_\s+", " ", normalized)
        normalized = re.sub(r"[|]{2,}", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip(" -:;,.|_/\\")
