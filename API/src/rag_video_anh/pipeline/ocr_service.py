"""Optical character recognition component."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from statistics import mean
from typing import Any

from src.configuration import AppConfig
from src.log.logger import logger
from src.rag_video_anh.schemas import OCRRequest, OCRResult, OCRResultSet, StageStatus, TextSpan


@dataclass(frozen=True)
class OCRReader:
    """PaddleOCR split reader: one detector pass, then crop-level recognition."""

    detector: Any
    recognizer: Any
    mode: str = "detector_once_crop"


class OCRService:
    """Extract text from each keyframe."""

    def __init__(self, config: AppConfig | None = None, reader: Any | None = None) -> None:
        self.config = config or AppConfig()
        self.pipeline_config = self.config.media_pipeline
        self.model_config = self.config.media_models
        self._reader = reader
        self._reader_loaded = reader is not None
        # Injected split readers should be reused so tests/custom callers do not reload PaddleOCR.
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
                full_text = self._clean_text(joined_text)
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

        detection_model = getattr(self.model_config, "ocr_detection_model_name", "PP-OCRv5_server_det")
        recognition_model = getattr(self.model_config, "ocr_recognition_model_name", "latin_PP-OCRv5_mobile_rec")
        try:
            from paddleocr import TextDetection, TextRecognition
        except ImportError:
            self._reader_loaded = True
            self._reader = None
            self._reader_device = device
            return None

        try:
            # Split modules keep the detector fixed and avoid rerunning it for every recognition variant.
            self._reader = OCRReader(
                detector=TextDetection(model_name=detection_model, device=device),
                recognizer=TextRecognition(model_name=recognition_model, device=device),
            )
        except (RuntimeError, ImportError) as exc:
            if device == "gpu" and self._should_retry_on_cpu(exc):
                logger.warning(f"PaddleOCR GPU initialization failed; retrying OCR on CPU: {exc}")
                return self._load_reader(force_device="cpu")
            raise

        self._reader_loaded = True
        self._reader_device = device
        return self._reader

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
        if not isinstance(reader, OCRReader):
            raise TypeError("OCR reader must expose split PaddleOCR detector and recognizer modules")
        return self._read_frame_detector_once(reader, image)

    def _read_frame_detector_once(self, reader: OCRReader, image: Any) -> list[TextSpan]:
        cv_image = self._load_cv_image(image)
        if cv_image is None:
            raise ValueError("keyframe image is not available as an OpenCV image")

        image_shape = int(cv_image.shape[0]), int(cv_image.shape[1])
        raw_detections = self._run_text_detection(reader.detector, image)
        regions = self._parse_detection_results(raw_detections, "detector_once")
        if not regions:
            return []

        crops: list[Any] = []
        crop_regions: list[TextSpan] = []
        for region in regions:
            crop = self._crop_text_region(cv_image, region.bbox)
            if crop is None:
                continue
            crops.append(crop)
            crop_regions.append(region)

        recognition_items = self._recognize_crop_images(reader.recognizer, crops)
        spans: list[TextSpan] = []
        for region, item in zip(crop_regions, recognition_items):
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            metadata = dict(region.metadata)
            metadata["ocr_variant"] = "detector_once_crop"
            metadata["recognition_input"] = "bbox_crop"
            spans.append(
                TextSpan(
                    text=text,
                    confidence=item.get("confidence"),
                    bbox=region.bbox,
                    metadata=metadata,
                )
            )

        return self._filter_and_clean_spans(spans, image_shape=image_shape)

    def _run_text_detection(self, detector: Any, image: Any) -> Any:
        predict = getattr(detector, "predict", None)
        if not callable(predict):
            raise AttributeError("PaddleOCR detector exposes no 'predict' method")
        try:
            return list(predict(input=image, batch_size=1))
        except TypeError:
            return list(predict(image))

    def _parse_detection_results(self, raw_results: Any, variant_name: str) -> list[TextSpan]:
        candidates = raw_results if isinstance(raw_results, list) else [raw_results]
        spans: list[TextSpan] = []
        for candidate in candidates:
            payload = self._result_payload(candidate)
            if not hasattr(payload, "get"):
                continue
            boxes = payload.get("dt_polys") or payload.get("polys") or payload.get("boxes") or []
            scores = payload.get("dt_scores") or payload.get("scores") or []
            for index, box in enumerate(boxes):
                confidence = float(scores[index]) if index < len(scores) and scores[index] is not None else None
                spans.append(
                    TextSpan(
                        text="",
                        confidence=confidence,
                        bbox=self._jsonable_bbox(box),
                        metadata={"ocr_variant": variant_name, "detection_confidence": confidence},
                    )
                )
        return sorted(spans, key=self._span_sort_key)

    @staticmethod
    def _result_payload(item: Any) -> Any:
        if hasattr(item, "json"):
            item = item.json
        if hasattr(item, "dict"):
            item = item.dict()
        if hasattr(item, "get") and item.get("res") is not None:
            return item.get("res")
        return item

    def _crop_text_region(self, cv_image: Any, bbox: Any) -> Any | None:
        rect = self._bbox_rect(bbox)
        if rect is None:
            return None

        x1, y1, x2, y2 = rect
        image_height, image_width = cv_image.shape[:2]
        left = max(0, int(x1))
        top = max(0, int(y1))
        right = min(image_width, int(x2))
        bottom = min(image_height, int(y2))
        if right <= left or bottom <= top:
            return None

        # Use the detector box directly to avoid adding background or interpolation noise.
        return cv_image[top:bottom, left:right]

    def _recognize_crop_images(self, recognizer: Any, crops: list[Any]) -> list[dict[str, Any]]:
        if not crops:
            return []
        raw_results = self._run_text_recognition(recognizer, crops)
        return [self._parse_recognition_item(item) for item in raw_results]

    def _run_text_recognition(self, recognizer: Any, crops: list[Any]) -> Any:
        predict = getattr(recognizer, "predict", None)
        if not callable(predict):
            raise AttributeError("PaddleOCR recognizer exposes no 'predict' method")
        batch_size = int(getattr(self.pipeline_config, "ocr_batch_size", 1) or 1)
        try:
            return list(predict(input=crops, batch_size=batch_size))
        except TypeError:
            outputs: list[Any] = []
            for crop in crops:
                outputs.extend(list(predict(crop)))
            return outputs

    def _parse_recognition_item(self, item: Any) -> dict[str, Any]:
        payload = self._result_payload(item)
        if hasattr(payload, "get"):
            confidence = payload.get("rec_score") or payload.get("score") or payload.get("confidence")
            return {
                "text": payload.get("rec_text") or payload.get("text") or "",
                "confidence": float(confidence) if confidence is not None else None,
            }
        return {"text": str(item), "confidence": None}

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
    def _jsonable_bbox(box: Any) -> Any:
        if hasattr(box, "tolist"):
            box = box.tolist()
        if isinstance(box, list) and len(box) >= 4 and all(isinstance(item, (int, float)) for item in box[:4]):
            x1, y1, x2, y2 = (float(item) for item in box[:4])
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        return box

    @classmethod
    def _span_sort_key(cls, span: TextSpan) -> tuple[float, float]:
        rect = cls._bbox_rect(span.bbox)
        if rect is None:
            return (float("inf"), float("inf"))
        return (rect[1], rect[0])

    def _filter_and_clean_spans(self, spans: list[TextSpan], image_shape: tuple[int, int] | None = None) -> list[TextSpan]:
        min_confidence = float(getattr(self.pipeline_config, "ocr_min_confidence", 0.25) or 0.0)
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

            # Normalization is intentionally safe-only and must not rewrite OCR content.
            cleaned_text = self._clean_text(text)
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
        alnum_count = self._alnum_count(text)

        if width < min_width or height < min_height:
            return False
        if area_ratio < min_area_ratio and alnum_count < min_chars:
            return False
        if region_type == "logo" and alnum_count <= logo_max_chars:
            return False
        if region_type == "other" and not self._is_informative_other_region(width, height, area_ratio, image_width, text):
            return False
        return True

    def _is_informative_other_region(
        self,
        width: float,
        height: float,
        area_ratio: float,
        image_width: int,
        text: str,
    ) -> bool:
        alnum_count = self._alnum_count(text)
        min_chars = int(getattr(self.pipeline_config, "ocr_other_min_text_chars", 4))
        min_width_ratio = float(getattr(self.pipeline_config, "ocr_other_min_width_ratio", 0.03))
        min_area_ratio = float(getattr(self.pipeline_config, "ocr_other_min_area_ratio", 0.0003))
        # Unclassified tiny tokens are usually logo fragments, watermarks, or partial glyphs in video frames.
        if alnum_count < min_chars:
            return False
        if width / max(float(image_width), 1.0) < min_width_ratio or area_ratio < min_area_ratio:
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
        alnum_count = self._alnum_count(text)
        top_branding = center_y <= 0.12 and width_ratio <= 0.18 and area_ratio <= logo_area
        if (near_corner or top_branding) and alnum_count <= int(getattr(self.pipeline_config, "ocr_logo_max_chars", 8)):
            return "logo"
        # Subtitle detectors may split one sentence into shorter boxes; keep bottom text fragments as subtitle.
        if center_y >= 0.72 and width_ratio >= 0.12 and height_ratio <= 0.16:
            return "subtitle"
        if width_ratio >= 0.45 and (center_y <= 0.30 or center_y >= 0.70):
            return "banner"
        if area_ratio >= 0.08 or (width_ratio >= 0.35 and height_ratio >= 0.08):
            return "slide"
        # Screen text often appears as multiple medium-width rows rather than one large region.
        if 0.15 <= center_y <= 0.85 and (area_ratio >= 0.02 or (width_ratio >= 0.22 and area_ratio >= 0.008)):
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
        normalized_chars: list[str] = []
        for char in normalized:
            if char.isspace():
                normalized_chars.append(" ")
                continue
            # Keep OCR text content intact; only remove Unicode control/format characters.
            if unicodedata.category(char) in {"Cc", "Cf"}:
                continue
            normalized_chars.append(char)
        return " ".join("".join(normalized_chars).split()).strip()
