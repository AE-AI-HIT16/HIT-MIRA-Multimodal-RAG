"""Optical character recognition component."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from statistics import mean
from typing import Any

from src.configuration import AppConfig
from src.log.logger import logger
from src.rag_video_anh.schemas import OCRRequest, OCRResult, OCRResultSet, StageStatus, TextSpan


@dataclass(frozen=True)
class ImageVariant:
    """OCR preprocessing output plus the scale needed to map boxes back."""

    name: str
    image: Any
    scale_x: float = 1.0
    scale_y: float = 1.0


class OCRService:
    """Extract text from each keyframe."""

    def __init__(self, config: AppConfig | None = None, reader: Any | None = None) -> None:
        self.config = config or AppConfig()
        self.pipeline_config = self.config.media_pipeline
        self.model_config = self.config.media_models
        self._reader = reader
        self._reader_loaded = reader is not None
        # Injected readers should be reused by default so tests and custom callers do not reload PaddleOCR.
        self._reader_device = ("gpu" if self.model_config.ocr_gpu else "cpu") if reader is not None else None

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
                try:
                    spans = self._read_frame(reader, image)
                except RuntimeError as exc:
                    if not self._should_retry_on_cpu(exc):
                        raise
                    logger.warning(
                        f"PaddleOCR failed on GPU for media_id '{request.media_id}', "
                        f"frame_id '{frame.frame_id}'; retrying OCR on CPU: {exc}"
                    )
                    reader = self._load_reader(force_device="cpu")
                    if reader is None:
                        raise
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

    def _load_reader(self, force_device: str | None = None) -> Any | None:
        device = force_device or ("gpu" if self.model_config.ocr_gpu else "cpu")
        if self._reader_loaded and self._reader_device == device:
            return self._reader
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            self._reader_loaded = True
            self._reader = None
            self._reader_device = device
            return None

        lang = self.model_config.ocr_languages[0] if self.model_config.ocr_languages else "en"
        try:
            # Explicit PP-OCRv5 server det/rec keeps model choice stable across PaddleOCR defaults.
            self._reader = PaddleOCR(**self._reader_kwargs(lang=lang, device=device))
        except ValueError as exc:
            if "Unknown argument" not in str(exc):
                raise
            # Older PaddleOCR versions expose the v2-style constructor; keep that path working.
            self._reader = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=(device == "gpu"))
        except (RuntimeError, ImportError) as exc:
            if device == "gpu" and self._should_retry_on_cpu(exc):
                logger.warning(f"PaddleOCR GPU initialization failed; retrying OCR on CPU: {exc}")
                return self._load_reader(force_device="cpu")
            raise
        self._reader_loaded = True
        self._reader_device = device
        return self._reader

    def _reader_kwargs(self, *, lang: str, device: str) -> dict[str, Any]:
        return {
            "lang": lang,
            "device": device,
            "ocr_version": getattr(self.model_config, "ocr_version", "PP-OCRv5"),
            "text_detection_model_name": getattr(self.model_config, "ocr_detection_model_name", "PP-OCRv5_server_det"),
            "text_recognition_model_name": getattr(self.model_config, "ocr_recognition_model_name", "PP-OCRv5_server_rec"),
            # Keyframes are not scanned documents; skipping document correction reduces memory and false transforms.
            "use_doc_orientation_classify": getattr(self.model_config, "ocr_use_doc_orientation_classify", False),
            "use_doc_unwarping": getattr(self.model_config, "ocr_use_doc_unwarping", False),
            "use_textline_orientation": getattr(self.model_config, "ocr_use_textline_orientation", True),
        }

    @staticmethod
    def _should_retry_on_cpu(exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "unsupported gpu architecture" in message
            or "compiled for" in message
            and "current gpu" in message
            or "libcuda" in message
        )

    def _read_frame(self, reader: Any, image: Any) -> list[TextSpan]:
        if image is None:
            raise ValueError("keyframe image is not available")

        image_shape = self._image_shape(image)
        spans: list[TextSpan] = []
        for variant in self._image_variants(image):
            raw_results = self._run_ocr(reader, variant.image)
            variant_spans = self._parse_results(raw_results, variant.name)
            # Variant boxes are mapped back so downstream merge/filtering sees one coordinate space.
            variant_spans = self._scale_spans_to_original(variant_spans, variant)
            spans.extend(variant_spans)

        spans = self._merge_spans(spans)
        spans = self._refine_small_regions(reader, image, spans)
        spans = self._merge_spans(spans)
        return self._filter_and_clean_spans(spans, image_shape=image_shape)

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

    def _image_variants(self, image: Any) -> list[ImageVariant]:
        variants: list[ImageVariant] = [ImageVariant("original", image)]
        if not getattr(self.pipeline_config, "ocr_enable_preprocessing", True):
            return variants

        cv_image = self._load_cv_image(image)
        if cv_image is None:
            return variants

        factors = self._normalized_upscale_factors()
        for factor in factors:
            if factor <= 1.0:
                continue
            variants.extend(self._scaled_variants(cv_image, factor))

        if len(variants) == 1:
            enhanced = self._enhance_for_text(cv_image)
            if enhanced is not None:
                variants.append(ImageVariant("clahe_sharp", enhanced))

        return variants

    def _normalized_upscale_factors(self) -> list[float]:
        factors = getattr(self.pipeline_config, "ocr_upscale_factors", [1.0, 2.0]) or [1.0]
        normalized: list[float] = []
        for factor in factors:
            try:
                value = float(factor)
            except (TypeError, ValueError):
                continue
            # Keeping 1x explicit documents the baseline while avoiding duplicate original variants.
            if value > 0 and value not in normalized:
                normalized.append(value)
        return normalized or [1.0]

    def _scaled_variants(self, cv_image: Any, factor: float) -> list[ImageVariant]:
        try:
            import cv2
        except ImportError:
            return []

        width = max(1, int(cv_image.shape[1] * factor))
        height = max(1, int(cv_image.shape[0] * factor))
        upscaled = cv2.resize(cv_image, (width, height), interpolation=cv2.INTER_CUBIC)
        variants = [ImageVariant(f"upscale_{factor:g}x", upscaled, scale_x=factor, scale_y=factor)]

        # CLAHE + sharpening remains the current contrast boost, isolated for future preprocessors.
        enhanced = self._enhance_for_text(upscaled)
        if enhanced is not None:
            variants.append(ImageVariant(f"upscale_{factor:g}x_clahe_sharp", enhanced, scale_x=factor, scale_y=factor))
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

    @staticmethod
    def _image_shape(image: Any) -> tuple[int, int] | None:
        cv_image = OCRService._load_cv_image(image)
        if cv_image is None:
            return None
        return int(cv_image.shape[0]), int(cv_image.shape[1])

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

    def _scale_spans_to_original(self, spans: list[TextSpan], variant: ImageVariant) -> list[TextSpan]:
        if variant.scale_x == 1.0 and variant.scale_y == 1.0:
            return spans

        scaled_spans: list[TextSpan] = []
        for span in spans:
            bbox = self._scale_bbox(span.bbox, scale_x=variant.scale_x, scale_y=variant.scale_y)
            metadata = dict(span.metadata)
            metadata["ocr_scale"] = {"x": variant.scale_x, "y": variant.scale_y}
            scaled_spans.append(TextSpan(text=span.text, confidence=span.confidence, bbox=bbox, metadata=metadata))
        return scaled_spans

    @staticmethod
    def _scale_bbox(bbox: Any, *, scale_x: float, scale_y: float) -> Any:
        if not bbox:
            return bbox
        return [[float(point[0]) / scale_x, float(point[1]) / scale_y] for point in bbox]

    def _refine_small_regions(self, reader: Any, image: Any, spans: list[TextSpan]) -> list[TextSpan]:
        cv_image = self._load_cv_image(image)
        if cv_image is None:
            return spans

        refined: list[TextSpan] = []
        for span in spans:
            replacement = self._recognize_upscaled_region(reader, cv_image, span)
            refined.append(replacement or span)
        return refined

    def _recognize_upscaled_region(self, reader: Any, cv_image: Any, span: TextSpan) -> TextSpan | None:
        rect = self._bbox_rect(span.bbox)
        if rect is None:
            return None

        x1, y1, x2, y2 = rect
        width = x2 - x1
        height = y2 - y1
        small_width = float(getattr(self.pipeline_config, "ocr_small_region_max_width", 220.0))
        small_height = float(getattr(self.pipeline_config, "ocr_small_region_max_height", 48.0))
        if width > small_width and height > small_height:
            return None

        try:
            import cv2
        except ImportError:
            return None

        margin = int(getattr(self.pipeline_config, "ocr_region_crop_margin", 4))
        image_height, image_width = cv_image.shape[:2]
        left = max(0, int(x1) - margin)
        top = max(0, int(y1) - margin)
        right = min(image_width, int(x2) + margin)
        bottom = min(image_height, int(y2) + margin)
        if right <= left or bottom <= top:
            return None

        crop = cv_image[top:bottom, left:right]
        factor = float(getattr(self.pipeline_config, "ocr_region_upscale_factor", 2.0) or 1.0)
        if factor > 1.0:
            crop = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

        raw_results = self._run_ocr(reader, crop)
        crop_spans = self._parse_results(raw_results, "region_upscale")
        best = self._best_span(crop_spans)
        if best is None or best.confidence is None:
            return None

        current_confidence = span.confidence if span.confidence is not None else 0.0
        improvement = float(getattr(self.pipeline_config, "ocr_region_refine_min_gain", 0.03))
        if best.confidence < current_confidence + improvement:
            return None

        metadata = dict(span.metadata)
        metadata["region_refined"] = True
        metadata["raw_text_before_region_refine"] = span.text
        metadata["region_refine_confidence_before"] = span.confidence
        return TextSpan(text=best.text, confidence=best.confidence, bbox=span.bbox, metadata=metadata)

    @staticmethod
    def _best_span(spans: list[TextSpan]) -> TextSpan | None:
        if not spans:
            return None
        return max(spans, key=lambda span: span.confidence if span.confidence is not None else 0.0)

    def _merge_spans(self, spans: list[TextSpan]) -> list[TextSpan]:
        threshold = float(getattr(self.pipeline_config, "ocr_merge_iou_threshold", 0.45))
        merged: list[TextSpan] = []
        for span in spans:
            match_index = self._matching_span_index(merged, span, threshold)
            if match_index is None:
                merged.append(span)
                continue
            if self._span_confidence(span) > self._span_confidence(merged[match_index]):
                metadata = dict(span.metadata)
                metadata["merged_from_variants"] = True
                merged[match_index] = TextSpan(text=span.text, confidence=span.confidence, bbox=span.bbox, metadata=metadata)
        return sorted(merged, key=self._span_sort_key)

    def _matching_span_index(self, spans: list[TextSpan], candidate: TextSpan, threshold: float) -> int | None:
        for index, span in enumerate(spans):
            if candidate.bbox and span.bbox and self._bbox_iou(candidate.bbox, span.bbox) >= threshold:
                return index
            if not candidate.bbox and not span.bbox and self._normalize_full_text(candidate.text) == self._normalize_full_text(span.text):
                return index
        return None

    @staticmethod
    def _span_confidence(span: TextSpan) -> float:
        return span.confidence if span.confidence is not None else 0.0

    @classmethod
    def _span_sort_key(cls, span: TextSpan) -> tuple[float, float]:
        rect = cls._bbox_rect(span.bbox)
        if rect is None:
            return (float("inf"), float("inf"))
        return (rect[1], rect[0])

    def _filter_and_clean_spans(self, spans: list[TextSpan], image_shape: tuple[int, int] | None = None) -> list[TextSpan]:
        min_confidence = float(getattr(self.pipeline_config, "ocr_min_confidence", 0.25) or 0.0)
        enable_cleanup = bool(getattr(self.pipeline_config, "ocr_text_cleanup", True))
        cleaned_spans: list[TextSpan] = []

        for span in spans:
            text = self._normalize_full_text(span.text)
            if not text:
                continue
            if span.confidence is not None and span.confidence < min_confidence:
                continue
            if self._looks_like_noise(text):
                continue

            metadata = dict(span.metadata)
            region_type = self._classify_region(span.bbox, image_shape, text)
            metadata["region_type"] = region_type
            if not self._should_keep_region(span.bbox, image_shape, text, region_type):
                continue

            # Cleanup is intentionally safe-only; domain regex rewrites caused hard-to-audit false corrections.
            cleaned_text = self._clean_text(text) if enable_cleanup else self._normalize_full_text(text)
            if not cleaned_text:
                continue

            if cleaned_text != span.text:
                metadata["raw_text"] = span.text
                metadata["cleanup"] = "safe_normalization_v1"
            cleaned_spans.append(
                TextSpan(
                    text=cleaned_text,
                    confidence=span.confidence,
                    bbox=span.bbox,
                    metadata=metadata,
                )
            )
        return cleaned_spans

    def _should_keep_region(
        self,
        bbox: Any,
        image_shape: tuple[int, int] | None,
        text: str,
        region_type: str,
    ) -> bool:
        rect = self._bbox_rect(bbox)
        if rect is None or image_shape is None:
            return True

        x1, y1, x2, y2 = rect
        width = x2 - x1
        height = y2 - y1
        image_height, image_width = image_shape
        area_ratio = (width * height) / max(float(image_width * image_height), 1.0)
        min_width = float(getattr(self.pipeline_config, "ocr_min_box_width", 8.0))
        min_height = float(getattr(self.pipeline_config, "ocr_min_box_height", 6.0))
        min_area_ratio = float(getattr(self.pipeline_config, "ocr_min_box_area_ratio", 0.00003))
        min_chars = int(getattr(self.pipeline_config, "ocr_min_text_chars", 2))
        logo_max_chars = int(getattr(self.pipeline_config, "ocr_logo_max_chars", 8))

        if width < min_width or height < min_height:
            return False
        if area_ratio < min_area_ratio and self._alnum_count(text) < min_chars:
            return False
        if region_type == "logo" and self._alnum_count(text) <= logo_max_chars:
            return False
        return True

    def _classify_region(self, bbox: Any, image_shape: tuple[int, int] | None, text: str) -> str:
        rect = self._bbox_rect(bbox)
        if rect is None or image_shape is None:
            return "other"

        x1, y1, x2, y2 = rect
        width = x2 - x1
        height = y2 - y1
        image_height, image_width = image_shape
        width_ratio = width / max(float(image_width), 1.0)
        height_ratio = height / max(float(image_height), 1.0)
        area_ratio = (width * height) / max(float(image_width * image_height), 1.0)
        center_x = (x1 + x2) / 2.0 / max(float(image_width), 1.0)
        center_y = (y1 + y2) / 2.0 / max(float(image_height), 1.0)

        corner_margin = float(getattr(self.pipeline_config, "ocr_logo_corner_margin_ratio", 0.18))
        logo_area = float(getattr(self.pipeline_config, "ocr_logo_max_area_ratio", 0.015))
        near_corner = center_x < corner_margin or center_x > 1.0 - corner_margin
        near_corner = near_corner and (center_y < corner_margin or center_y > 1.0 - corner_margin)
        if near_corner and area_ratio <= logo_area and self._alnum_count(text) <= int(getattr(self.pipeline_config, "ocr_logo_max_chars", 8)):
            return "logo"
        if center_y >= 0.62 and width_ratio >= 0.20 and height_ratio <= 0.16:
            return "subtitle"
        if width_ratio >= 0.45 and (center_y <= 0.30 or center_y >= 0.70):
            return "banner"
        if area_ratio >= 0.08 or (width_ratio >= 0.35 and height_ratio >= 0.08):
            return "slide"
        if 0.15 <= center_y <= 0.85 and area_ratio >= 0.02:
            return "screen"
        return "other"

    @staticmethod
    def _bbox_rect(bbox: Any) -> tuple[float, float, float, float] | None:
        if not bbox:
            return None
        try:
            xs = [float(point[0]) for point in bbox]
            ys = [float(point[1]) for point in bbox]
        except (TypeError, ValueError, IndexError):
            return None
        return min(xs), min(ys), max(xs), max(ys)

    @classmethod
    def _bbox_iou(cls, first: Any, second: Any) -> float:
        first_rect = cls._bbox_rect(first)
        second_rect = cls._bbox_rect(second)
        if first_rect is None or second_rect is None:
            return 0.0
        ax1, ay1, ax2, ay2 = first_rect
        bx1, by1, bx2, by2 = second_rect
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_width = max(0.0, inter_x2 - inter_x1)
        inter_height = max(0.0, inter_y2 - inter_y1)
        intersection = inter_width * inter_height
        first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _looks_like_noise(text: str) -> bool:
        alnum_count = OCRService._alnum_count(text)
        if alnum_count == 0:
            return True
        junk_count = sum(char in r"_|/\\~`^*#=+<>" for char in text)
        return junk_count > alnum_count

    @staticmethod
    def _alnum_count(text: str) -> int:
        return sum(char.isalnum() for char in text)

    @classmethod
    def _clean_text(cls, text: str) -> str:
        # Safe normalization only; model-specific corrections belong in eval-driven post-processing.
        return cls._normalize_full_text(text)

    @staticmethod
    def _normalize_full_text(text: str) -> str:
        normalized = unicodedata.normalize("NFC", text or "")
        normalized = normalized.replace(" ", " ")
        normalized = re.sub(r"[​‌‍﻿]", "", normalized)
        normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", normalized)
        normalized = re.sub(r"[_]{2,}", " ", normalized)
        normalized = re.sub(r"\s+_\s+", " ", normalized)
        normalized = re.sub(r"[|]{2,}", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip(r" -:;,.|_/\\")
