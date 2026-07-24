"""Video keyframe extraction component."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.configuration import AppConfig
from src.log.logger import logger
from src.rag_video_anh.schemas import KeyFrame, KeyFrameSet, KeyframeExtractionRequest, StageStatus


class DefaultTransNetV2SceneDetector:
    """Lazy TransNetV2 scene detector with whole-video fallback handled by the caller."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model_config = config.media_models
        self._model: Any | None = None
        self._device: Any | None = None
        self._load_error: str | None = None

    def detect_scenes(self, media_path: str, *, window_length: int, window_stride: int, threshold: float) -> list[tuple[int, int]]:
        model = self._load_model()
        frames = self._read_model_frames(media_path)
        scores = self._predict_scores(model, frames, window_length=window_length, window_stride=window_stride)
        return self._scores_to_scenes(scores, threshold)

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not getattr(self.model_config, "temporal_scene_detector_enabled", True):
            raise RuntimeError("TransNetV2 scene detector is disabled by config")

        checkpoint_path = self._checkpoint_path()
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"TransNetV2 checkpoint not found: {getattr(self.model_config, 'temporal_checkpoint_name', None)}"
            )

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("missing dependency 'torch' for TransNetV2 scene detection") from exc

        device = self._select_device(torch)
        try:
            try:
                model = torch.jit.load(str(checkpoint_path), map_location=device)
            except Exception:
                try:
                    loaded = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
                except TypeError:
                    loaded = torch.load(str(checkpoint_path), map_location=device)
                model = self._model_from_checkpoint(loaded)
            if hasattr(model, "to"):
                model = model.to(device)
            if hasattr(model, "eval"):
                model.eval()
        except Exception as exc:
            self._load_error = str(exc)
            raise RuntimeError(f"failed to load TransNetV2 checkpoint '{checkpoint_path}': {exc}") from exc

        self._model = model
        self._device = device
        return model

    def _checkpoint_path(self) -> Path | None:
        checkpoint_name = str(getattr(self.model_config, "temporal_checkpoint_name", "") or "")
        checkpoint_dir = str(getattr(self.model_config, "temporal_checkpoint_dir", "") or "")
        if not checkpoint_name:
            return None

        candidates: list[Path] = []
        name_path = Path(checkpoint_name)
        if name_path.is_absolute():
            candidates.append(name_path)
        else:
            if checkpoint_dir:
                candidates.append(Path(checkpoint_dir) / checkpoint_name)
            candidates.extend(
                [
                    Path.cwd() / checkpoint_name,
                    Path.cwd() / "API" / "Resources" / checkpoint_name,
                    Path.cwd() / "data" / "models" / checkpoint_name,
                ]
            )
        return next((path for path in candidates if path.is_file()), None)

    @staticmethod
    def _model_from_checkpoint(loaded: Any) -> Any:
        if hasattr(loaded, "eval") and callable(getattr(loaded, "__call__", None)):
            return loaded
        if isinstance(loaded, dict):
            for key in ("model", "net", "module"):
                candidate = loaded.get(key)
                if hasattr(candidate, "eval") and callable(getattr(candidate, "__call__", None)):
                    return candidate
        raise RuntimeError("checkpoint does not contain an executable TransNetV2 model")

    def _select_device(self, torch: Any) -> Any:
        policy = str(getattr(self.model_config, "temporal_device_policy", "auto") or "auto").lower()
        if policy in {"cuda", "gpu"}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if policy == "cpu":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _read_model_frames(self, media_path: str) -> Any:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("missing dependency 'opencv-python' for TransNetV2 preprocessing") from exc

        capture = cv2.VideoCapture(str(media_path))
        if not capture.isOpened():
            raise RuntimeError("unable to decode media for TransNetV2 scene detection")

        width = int(getattr(self.model_config, "frame_resize_width", 48) or 48)
        height = int(getattr(self.model_config, "frame_resize_height", 27) or 27)
        frames: list[Any] = []
        try:
            while True:
                success, image = capture.read()
                if not success:
                    break
                resized = cv2.resize(image, (width, height))
                frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()

        if not frames:
            raise RuntimeError("video contains no frames for TransNetV2 scene detection")
        return np.stack(frames, axis=0)

    def _predict_scores(self, model: Any, frames: Any, *, window_length: int, window_stride: int) -> list[float]:
        import numpy as np
        import torch

        total_frames = int(frames.shape[0])
        window = max(1, int(window_length or total_frames))
        stride = max(1, int(window_stride or window))
        starts = list(range(0, total_frames, stride))
        tail_start = max(0, total_frames - window)
        if tail_start not in starts:
            starts.append(tail_start)
        starts = sorted(set(starts))

        score_sum = np.zeros(total_frames, dtype=np.float32)
        score_count = np.zeros(total_frames, dtype=np.float32)
        for start in starts:
            end = min(total_frames, start + window)
            chunk = frames[start:end]
            if len(chunk) < window:
                pad_count = window - len(chunk)
                pad = np.repeat(chunk[-1:], pad_count, axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)
            scores = self._predict_window(model, chunk, torch)
            usable = min(len(scores), total_frames - start)
            if usable <= 0:
                continue
            score_sum[start : start + usable] += np.asarray(scores[:usable], dtype=np.float32)
            score_count[start : start + usable] += 1.0

        score_count[score_count == 0] = 1.0
        return (score_sum / score_count).tolist()

    def _predict_window(self, model: Any, window: Any, torch: Any) -> list[float]:
        import numpy as np

        device = self._device or self._select_device(torch)
        uint8_tensor = torch.from_numpy(window).unsqueeze(0).to(device)
        float_tensor = uint8_tensor.float() / 255.0
        input_variants = [uint8_tensor, float_tensor, float_tensor.permute(0, 1, 4, 2, 3)]
        last_error: Exception | None = None
        with torch.no_grad():
            for tensor in input_variants:
                try:
                    output = model(tensor)
                    return self._flatten_scores(output, np)
                except Exception as exc:
                    last_error = exc
        raise RuntimeError(f"TransNetV2 inference failed for all input layouts: {last_error}")

    @staticmethod
    def _flatten_scores(output: Any, np: Any) -> list[float]:
        if isinstance(output, dict):
            for key in ("many_hot", "single_frame_pred", "predictions", "logits", "output"):
                if key in output:
                    output = output[key]
                    break
        elif isinstance(output, (list, tuple)):
            output = output[-1] if len(output) > 1 else output[0]
        if hasattr(output, "detach"):
            output = output.detach().float().cpu().numpy()
        values = np.asarray(output, dtype=np.float32).reshape(-1)
        if values.size == 0:
            raise RuntimeError("TransNetV2 returned empty predictions")
        if float(values.min()) < 0.0 or float(values.max()) > 1.0:
            values = 1.0 / (1.0 + np.exp(-values))
        return [float(value) for value in values]

    @staticmethod
    def _scores_to_scenes(scores: list[float], threshold: float) -> list[tuple[int, int]]:
        frame_count = len(scores)
        if frame_count == 0:
            return []
        cuts = [index for index, score in enumerate(scores) if float(score) >= float(threshold)]
        scenes: list[tuple[int, int]] = []
        start = 0
        for cut in cuts:
            end = min(max(cut, start), frame_count - 1)
            if end > start:
                scenes.append((start, end))
            start = min(end + 1, frame_count - 1)
        if start < frame_count - 1:
            scenes.append((start, frame_count - 1))
        return scenes or [(0, frame_count - 1)]


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
        # Prefer an injected detector for tests/custom deployments; otherwise initialize TransNetV2 by default.
        self.scene_detector = scene_detector or DefaultTransNetV2SceneDetector(self.config)
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
        candidate_indices = self._candidate_indices(scenes, frame_count, fps)
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
            filtered = [self._quality_fallback_candidate(candidates)]

        selected = self._deduplicate(filtered, cv2)
        frames = [self._build_keyframe(media_input.media_id, item, fps) for item in selected]
        frames.sort(key=self._frame_sort_key)
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
            try:
                detected = self.scene_detector.detect_scenes(
                    media_path,
                    window_length=self.model_config.window_length,
                    window_stride=self.model_config.window_stride,
                    threshold=self.model_config.shot_boundary_threshold,
                )
                scenes = [(int(start), int(end)) for start, end in detected if int(end) > int(start)]
                if scenes:
                    return scenes
            except Exception as exc:
                logger.warning(f"Scene detection failed; falling back to the full video as one scene: {exc}")
        return [(0, max(frame_count - 1, 0))]

    def _candidate_indices(self, scenes: list[tuple[int, int]], frame_count: int, fps: float) -> list[int]:
        indices: list[int] = []
        for start, end in scenes:
            start = max(0, start)
            end = min(frame_count - 1, end)
            length = end - start + 1
            if length < self.pipeline_config.short_scene_threshold:
                indices.append(start + length // 2)
                continue
            scene_indices = self._adaptive_scene_indices(start, end, length, fps)
            indices.extend(scene_indices)
        return sorted({min(max(index, 0), frame_count - 1) for index in indices})

    def _adaptive_scene_indices(self, start: int, end: int, length: int, fps: float) -> list[int]:
        positions = getattr(self.pipeline_config, "keyframe_base_positions", [0.0, 0.5, 1.0]) or [0.0, 0.5, 1.0]
        candidates = [start + round((length - 1) * float(position)) for position in positions]

        interval_seconds = float(getattr(self.pipeline_config, "keyframe_long_scene_interval_sec", 2.5) or 0.0)
        interval_frames = int(round(interval_seconds * max(float(fps or 0.0), 1.0)))
        if interval_frames > 0 and length > interval_frames:
            candidates.extend(range(start, end + 1, interval_frames))
            candidates.append(end)

        bounded = sorted({min(max(index, start), end) for index in candidates})
        return self._limit_scene_candidates(bounded, start, end, length)

    def _limit_scene_candidates(self, candidates: list[int], start: int, end: int, length: int) -> list[int]:
        max_candidates = max(3, int(getattr(self.pipeline_config, "keyframe_max_candidates_per_scene", 8) or 8))
        if len(candidates) <= max_candidates:
            return candidates

        protected = sorted({start, start + round((length - 1) * 0.5), end})
        remaining = max_candidates - len(protected)
        extras = [index for index in candidates if index not in protected]
        if remaining <= 0:
            return protected
        if len(extras) <= remaining:
            return sorted(protected + extras)

        step = (len(extras) - 1) / max(float(remaining - 1), 1.0)
        sampled = [extras[round(step * offset)] for offset in range(remaining)]
        return sorted(set(protected + sampled))

    def _quality_fallback_candidate(self, candidates: list[tuple[int, Any, dict[str, float]]]) -> tuple[int, Any, dict[str, float]]:
        strategy = str(getattr(self.pipeline_config, "quality_fallback_selection", "best_quality_candidate") or "").lower()
        if strategy == "middle_candidate":
            return sorted(candidates, key=lambda item: item[0])[len(candidates) // 2]
        return max(candidates, key=lambda item: item[2]["quality_score"])

    def _frame_sort_key(self, frame: KeyFrame) -> tuple[float, int]:
        order = str(getattr(self.pipeline_config, "keyframe_output_sort_order", "frame_index") or "frame_index").lower()
        if order == "timestamp":
            return (float(frame.timestamp_sec), int(frame.frame_index))
        return (float(frame.frame_index), int(frame.frame_index))

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
        quality_score = self._weighted_quality_score(blur_score, brightness, contrast)
        return {
            "blur_score": blur_score,
            "brightness": brightness,
            "contrast": contrast,
            "quality_score": quality_score,
        }

    def _weighted_quality_score(self, blur_score: float, brightness: float, contrast: float) -> float:
        blur_reference = max(float(getattr(self.pipeline_config, "quality_blur_reference", 400.0) or 400.0), 1.0)
        contrast_reference = max(float(getattr(self.pipeline_config, "quality_contrast_reference", 64.0) or 64.0), 1.0)
        brightness_midpoint = float(getattr(self.pipeline_config, "brightness_midpoint", 127.5) or 127.5)
        brightness_range = max(brightness_midpoint, 255.0 - brightness_midpoint, 1.0)

        blur_norm = self._clamp01(blur_score / blur_reference)
        brightness_norm = self._clamp01(1.0 - abs(brightness - brightness_midpoint) / brightness_range)
        contrast_norm = self._clamp01(contrast / contrast_reference)

        blur_weight = float(getattr(self.pipeline_config, "quality_blur_weight", 0.45) or 0.0)
        brightness_weight = float(getattr(self.pipeline_config, "quality_brightness_weight", 0.25) or 0.0)
        contrast_weight = float(getattr(self.pipeline_config, "quality_contrast_weight", 0.30) or 0.0)
        total_weight = max(blur_weight + brightness_weight + contrast_weight, 1e-6)
        return (
            blur_norm * blur_weight
            + brightness_norm * brightness_weight
            + contrast_norm * contrast_weight
        ) / total_weight

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(float(value), 1.0))

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
