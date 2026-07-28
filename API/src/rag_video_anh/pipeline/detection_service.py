"""Object detection component."""

from __future__ import annotations

from typing import Any

from src.configuration import AppConfig
from src.log.logger import logger
from src.rag_video_anh.schemas import DetectionBox, DetectionRequest, DetectionResult, DetectionResultSet, StageStatus


class DetectionService:
    """Detect objects and salient entities in keyframes."""

    def __init__(self, config: AppConfig | None = None, model: Any | None = None) -> None:
        self.config = config or AppConfig()
        self.pipeline_config = self.config.media_pipeline
        self.model_config = self.config.media_models
        self.model = model
        self._model_loaded = model is not None
        self._load_error: str | None = None

    def detect(self, request: DetectionRequest) -> DetectionResultSet:
        if not request.keyframes.frames:
            return DetectionResultSet(
                media_id=request.media_id,
                status=StageStatus.SKIPPED,
                reason="no keyframes available for detection",
            )

        model = self._load_model()
        if model is None:
            return DetectionResultSet(
                media_id=request.media_id,
                status=StageStatus.SKIPPED,
                reason=self._load_error or "missing dependency 'ultralytics'",
            )

        results: list[DetectionResult] = []
        for frame in request.keyframes.frames:
            try:
                image = frame.image_payload if frame.image_payload is not None else frame.image_path
                detections, device = self._detect_frame(model, image)
                results.append(
                    DetectionResult(
                        frame_id=frame.frame_id,
                        detections=detections,
                        inference_meta={
                            "model": self.model_config.detection_model_name,
                            "confidence_threshold": self.model_config.detection_confidence_threshold,
                            "device": device,
                        },
                        status=StageStatus.DONE if detections else StageStatus.NOT_FOUND,
                        reason=None if detections else "no detections after filtering",
                    )
                )
            except Exception as exc:
                results.append(
                    DetectionResult(
                        frame_id=frame.frame_id,
                        status=StageStatus.ERROR,
                        reason=f"object detection failed: {self._exception_message(exc)}",
                    )
                )

        status = StageStatus.DONE if all(result.status != StageStatus.ERROR for result in results) else StageStatus.ERROR
        logger.info(f"Detection processed {len(results)} frame(s) for media_id '{request.media_id}'")
        return DetectionResultSet(media_id=request.media_id, results=results, status=status)

    def _load_model(self) -> Any | None:
        if self._model_loaded:
            return self.model
        try:
            from ultralytics import YOLO
        except ImportError:
            self._model_loaded = True
            self._load_error = "missing dependency 'ultralytics'"
            return None
        try:
            self.model = YOLO(self.model_config.detection_weight_file)
            self._model_loaded = True
            return self.model
        except Exception as exc:
            self._model_loaded = True
            self._load_error = f"detection model load failed: {exc.__class__.__name__}"
            return None

    def _detect_frame(self, model: Any, image: Any) -> tuple[list[DetectionBox], str]:
        if image is None:
            raise ValueError("keyframe image is not available")
        raw_results, device = self._run_model(model, image)
        boxes: list[DetectionBox] = []
        for result in raw_results or []:
            names = getattr(result, "names", {}) or {}
            result_boxes = getattr(result, "boxes", None)
            if result_boxes is None:
                continue
            xyxy = getattr(result_boxes, "xyxy", [])
            confidences = getattr(result_boxes, "conf", [])
            classes = getattr(result_boxes, "cls", [])
            for coords, confidence, class_id in zip(xyxy, confidences, classes):
                confidence_value = float(confidence)
                if confidence_value < self.model_config.detection_confidence_threshold:
                    continue
                label = str(names.get(int(class_id), int(class_id)))
                rounded_box = [round(float(value), 1) for value in coords]
                boxes.append(
                    DetectionBox(
                        label=label,
                        confidence=round(confidence_value, 4),
                        bbox=rounded_box,
                    )
                )
        return boxes, device

    def _run_model(self, model: Any, image: Any) -> tuple[Any, str]:
        policy = str(self.model_config.detection_device_policy or "runtime_default").strip().lower()
        kwargs: dict[str, Any] = {"verbose": self.model_config.detection_verbose}
        if policy not in {"", "auto", "default", "runtime_default"}:
            kwargs["device"] = policy
            return model(image, **kwargs), policy

        try:
            return model(image, **kwargs), "runtime_default"
        except Exception as exc:
            if not self._is_accelerator_error(exc):
                raise
            logger.warning(f"Detection accelerator failed; retrying on CPU: {self._exception_message(exc)}")
            return model(image, verbose=self.model_config.detection_verbose, device="cpu"), "cpu"

    @staticmethod
    def _is_accelerator_error(exc: Exception) -> bool:
        text = f"{exc.__class__.__name__}: {exc}".lower()
        return any(marker in text for marker in ("accelerator", "cuda", "cudnn", "mps"))

    @staticmethod
    def _exception_message(exc: Exception) -> str:
        message = str(exc).strip()
        if not message:
            return exc.__class__.__name__
        if len(message) > 300:
            message = message[:300] + "..."
        return f"{exc.__class__.__name__}: {message}"
