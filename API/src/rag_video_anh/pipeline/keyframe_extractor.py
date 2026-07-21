"""Video keyframe extraction component."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.configuration import AppConfig
from src.log.logger import logger
from src.rag_video_anh.schemas import KeyFrame, KeyFrameSet, KeyframeExtractionRequest, StageStatus


class KeyframeExtractorService:
    """Extract representative keyframes from a resolved media object."""

    def __init__(
        self,
        config: AppConfig | None = None,
        scene_detector: Any | None = None,
        embedding_model: Any | None = None,
    ) -> None:
        self.config = config or AppConfig()
        self.pipeline_config = self.config.media_pipeline
        self.model_config = self.config.media_models
        self.scene_detector = scene_detector
        self.embedding_model = embedding_model

    def extract(self, request: KeyframeExtractionRequest) -> KeyFrameSet:
        media_input = request.media_input
        route = request.route
        if route is not None and not route.requires_keyframes:
            return KeyFrameSet(
                media_id=media_input.media_id,
                status=StageStatus.SKIPPED,
                reason="keyframe extraction disabled by route",
            )

        media_path = self._resolve_media_path(media_input)
        if media_path is None:
            return KeyFrameSet(
                media_id=media_input.media_id,
                status=StageStatus.ERROR,
                reason="media path is required for keyframe extraction",
            )

        try:
            import cv2
        except ImportError:
            return KeyFrameSet(
                media_id=media_input.media_id,
                status=StageStatus.ERROR,
                reason="missing dependency 'opencv-python' for video decoding",
            )

        capture = cv2.VideoCapture(str(media_path))
        if not capture.isOpened():
            return KeyFrameSet(
                media_id=media_input.media_id,
                status=StageStatus.ERROR,
                reason="unable to decode media",
            )

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or self.pipeline_config.fps_fallback)
        if frame_count <= 0:
            capture.release()
            return KeyFrameSet(
                media_id=media_input.media_id,
                status=StageStatus.ERROR,
                reason="video contains no decodable frames",
            )

        scenes = self._detect_scenes(str(media_path), frame_count)
        candidate_indices = self._candidate_indices(scenes, frame_count)
        candidates = []
        for frame_index in candidate_indices:
            image = self._read_frame(capture, frame_index)
            if image is None:
                continue
            quality = self._quality_score(image, cv2)
            candidates.append((frame_index, image, quality))
        capture.release()

        if not candidates:
            return KeyFrameSet(
                media_id=media_input.media_id,
                status=StageStatus.ERROR,
                reason="unable to produce keyframe candidates",
            )

        filtered = [item for item in candidates if self._passes_quality(item[2])]
        if not filtered:
            filtered = [max(candidates, key=lambda item: item[2]["quality_score"])]

        selected = self._deduplicate(filtered, cv2)
        frames = [self._build_keyframe(media_input.media_id, item, fps) for item in selected]
        frames.sort(key=lambda frame: frame.frame_index)
        logger.info(f"Extracted {len(frames)} keyframe(s) for media_id '{media_input.media_id}'")
        return KeyFrameSet(
            media_id=media_input.media_id,
            frames=frames,
            status=StageStatus.DONE,
            selection_summary={
                "total_frames": frame_count,
                "candidate_count": len(candidates),
                "selected_count": len(frames),
                "scene_count": len(scenes),
            },
            coverage_summary={"fps": fps},
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

    def _detect_scenes(self, media_path: str, frame_count: int) -> list[tuple[int, int]]:
        if self.scene_detector is not None and hasattr(self.scene_detector, "detect_scenes"):
            detected = self.scene_detector.detect_scenes(
                media_path,
                window_length=self.model_config.window_length,
                window_stride=self.model_config.window_stride,
                threshold=self.model_config.shot_boundary_threshold,
            )
            scenes = [(int(start), int(end)) for start, end in detected if int(end) > int(start)]
            if scenes:
                return scenes
        return [(0, max(frame_count - 1, 0))]

    def _candidate_indices(self, scenes: list[tuple[int, int]], frame_count: int) -> list[int]:
        indices: list[int] = []
        for start, end in scenes:
            start = max(0, start)
            end = min(frame_count - 1, end)
            length = end - start + 1
            if length < self.pipeline_config.short_scene_threshold:
                indices.append(start + length // 2)
                continue
            for position in self.pipeline_config.keyframe_positions:
                indices.append(start + int(length * float(position)))
        return sorted({min(max(index, 0), frame_count - 1) for index in indices})

    @staticmethod
    def _read_frame(capture: Any, frame_index: int) -> Any | None:
        capture.set(1, frame_index)
        success, image = capture.read()
        if not success:
            return None
        return image

    def _quality_score(self, image: Any, cv2: Any) -> dict[str, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        contrast = float(gray.std())
        brightness_score = max(0.0, self.pipeline_config.brightness_midpoint - abs(brightness - self.pipeline_config.brightness_midpoint))
        return {
            "blur_score": blur_score,
            "brightness": brightness,
            "contrast": contrast,
            "quality_score": blur_score + brightness_score + contrast,
        }

    def _passes_quality(self, score: dict[str, float]) -> bool:
        return (
            score["blur_score"] >= self.pipeline_config.min_blur_score
            and self.pipeline_config.min_brightness <= score["brightness"] <= self.pipeline_config.max_brightness
            and score["contrast"] >= self.pipeline_config.min_contrast
        )

    def _deduplicate(self, candidates: list[tuple[int, Any, dict[str, float]]], cv2: Any) -> list[tuple[int, Any, dict[str, float], float]]:
        kept: list[tuple[int, Any, dict[str, float], Any, float]] = []
        for frame_index, image, quality in candidates:
            embedding = self._embedding(image, cv2)
            max_similarity = 0.0
            for _, _, _, kept_embedding, _ in kept:
                max_similarity = max(max_similarity, self._cosine_similarity(embedding, kept_embedding))
            if max_similarity < self.model_config.duplicate_threshold:
                kept.append((frame_index, image, quality, embedding, max_similarity))
        if not kept:
            frame_index, image, quality = max(candidates, key=lambda item: item[2]["quality_score"])
            kept.append((frame_index, image, quality, self._embedding(image, cv2), 0.0))
        return [(frame_index, image, quality, dedup_score) for frame_index, image, quality, _, dedup_score in kept]

    def _embedding(self, image: Any, cv2: Any) -> list[float]:
        if self.embedding_model is not None and hasattr(self.embedding_model, "encode"):
            vector = self.embedding_model.encode(image)
            return [float(value) for value in vector]
        resized = cv2.resize(image, (32, 32))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return [float(value) / 255.0 for value in rgb.reshape(-1)]

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = sum(a * a for a in left) ** 0.5
        right_norm = sum(b * b for b in right) ** 0.5
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

    def _build_keyframe(self, media_id: str, item: tuple[int, Any, dict[str, float], float], fps: float) -> KeyFrame:
        frame_index, image, quality, dedup_score = item
        timestamp_sec = frame_index / fps if fps else 0.0
        return KeyFrame(
            frame_id=self.pipeline_config.frame_id_pattern.format(media_id=media_id, index=frame_index),
            media_id=media_id,
            frame_index=frame_index,
            timestamp_ms=int(timestamp_sec * 1000),
            timestamp_sec=timestamp_sec,
            quality_score=quality["quality_score"],
            dedup_score=dedup_score,
            selection_reason="scene_candidate_quality_dedup",
            image_payload=image,
            metadata={
                "blur_score": quality["blur_score"],
                "brightness": quality["brightness"],
                "contrast": quality["contrast"],
            },
        )
