"""Offline smoke checks for the rag_video_anh package.

The script is intentionally dependency-light: it validates syntax, imports every
module, and exercises PipelineService with injected stage stubs so the Docker
test image does not need GPU libraries, model weights, or external API keys.
"""

from __future__ import annotations

import compileall
import importlib
import os
import pkgutil
import sys
from pathlib import Path
from typing import Any


API_ROOT = Path(__file__).resolve().parents[1]


def configure_environment() -> None:
    defaults = {
        "CONFIG_PATH": "API/Resources/dev.yaml",
        "MODELS_PATH": "API/Resources/model.yaml",
        "PROMPTS_PATH": "API/Resources/prompt.yaml",
        "LOG_DIR": "/tmp/hit-mira-api-test-logs",
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_API_KEY": "",
        "QDRANT_COLLECTION_NAME": "rag_documents",
        "QDRANT_DENSE_VECTOR_SIZE": "1536",
        "EMBEDDING_API_KEY": "",
        "EMBEDDING_BASE_URL": "https://openrouter.ai/api/v1",
        "EMBEDDING_DIMENSIONS": "1024",
        "EMBEDDING_MODEL": "baai/bge-m3",
        "RETRIEVAL_TOP_K": "5",
        "RETRIEVAL_KEYWORD_THRESHOLD": "0.5",
        "MINIO_ENDPOINT": "localhost:9000",
        "MINIO_ACCESS_KEY": "minioadmin",
        "MINIO_SECRET_KEY": "minioadmin",
        "MINIO_BUCKET": "hit-mira-media",
        "MINIO_SECURE": "false",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    sys.path.insert(0, str(API_ROOT))


def compile_package() -> None:
    package_dir = API_ROOT / "src" / "rag_video_anh"
    if not compileall.compile_dir(str(package_dir), quiet=1, force=True):
        raise SystemExit("compileall failed for src.rag_video_anh")


def import_package_modules() -> None:
    package = importlib.import_module("src.rag_video_anh")
    failures: list[tuple[str, Exception]] = []
    for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            importlib.import_module(module_info.name)
        except Exception as exc:
            failures.append((module_info.name, exc))
    if failures:
        details = "\n".join(f"- {name}: {exc.__class__.__name__}: {exc}" for name, exc in failures)
        raise SystemExit(f"module import check failed:\n{details}")


class StaticKeyframeExtractor:
    def extract(self, request: Any) -> Any:
        from src.rag_video_anh.schemas import KeyFrame, KeyFrameSet, StageStatus

        media_id = request.media_input.media_id
        return KeyFrameSet(
            media_id=media_id,
            frames=[
                KeyFrame(
                    frame_id=f"{media_id}_f000001",
                    media_id=media_id,
                    frame_index=1,
                    timestamp_ms=1000,
                    timestamp_sec=1.0,
                    quality_score=0.95,
                    dedup_score=0.0,
                    selection_reason="smoke_test_stub",
                    metadata={"source": "smoke_test"},
                )
            ],
            status=StageStatus.DONE,
        )


class StaticVisionService:
    def analyze(self, *, media_id: str, keyframes: Any, ocr_policy: dict[str, Any], caption_policy: dict[str, Any]) -> Any:
        from src.rag_video_anh.schemas import CaptionResult, CaptionResultSet, OCRResult, OCRResultSet, StageStatus

        frame_id = keyframes.frames[0].frame_id
        return (
            OCRResultSet(
                media_id=media_id,
                results=[OCRResult(frame_id=frame_id, full_text="HIT MIRA", confidence=0.99)],
                status=StageStatus.DONE,
            ),
            CaptionResultSet(
                media_id=media_id,
                results=[CaptionResult(frame_id=frame_id, caption_text="Khung hinh smoke test.")],
                status=StageStatus.DONE,
            ),
        )


class StaticDetectionService:
    def detect(self, request: Any) -> Any:
        from src.rag_video_anh.schemas import DetectionBox, DetectionResult, DetectionResultSet, StageStatus

        frame_id = request.keyframes.frames[0].frame_id
        return DetectionResultSet(
            media_id=request.media_id,
            results=[
                DetectionResult(
                    frame_id=frame_id,
                    detections=[DetectionBox(label="screen", confidence=0.9, bbox=[0.0, 0.0, 10.0, 10.0])],
                )
            ],
            status=StageStatus.DONE,
        )


class StaticAsrService:
    def transcribe(self, request: Any) -> Any:
        from src.rag_video_anh.schemas import StageStatus, TranscriptSegment, TranscriptSet

        media_id = request.media_input.media_id
        return TranscriptSet(
            media_id=media_id,
            segments=[
                TranscriptSegment(
                    segment_id="seg-1",
                    start_ms=0,
                    end_ms=2000,
                    start_sec=0.0,
                    end_sec=2.0,
                    text="Day la transcript smoke test.",
                    confidence=0.98,
                    language="vi",
                )
            ],
            language="vi",
            status=StageStatus.DONE,
        )


def run_pipeline_smoke_test() -> None:
    from src.configuration import AppConfig
    from src.rag_video_anh.pipeline.pipeline_service import PipelineService
    from src.rag_video_anh.schemas import MediaInput, PipelineStatus

    service = PipelineService(
        config=AppConfig(),
        keyframe_extractor=StaticKeyframeExtractor(),
        vision_service=StaticVisionService(),
        detection_service=StaticDetectionService(),
        asr_service=StaticAsrService(),
    )
    result = service.process(
        MediaInput(
            media_id="smoke-video",
            media_type="video",
            source_ref="smoke.mp4",
            mime_type="video/mp4",
            duration=2.0,
            size_bytes=1024,
            language_hint="vi",
            metadata={"audio_present": True, "frame_rate": 25.0, "width": 1280, "height": 720},
        )
    )
    if result.status != PipelineStatus.SUCCESS:
        raise SystemExit(f"pipeline smoke test failed with status={result.status!r}, errors={result.errors!r}")
    if result.normalized_metadata is None or not result.normalized_metadata.frames:
        raise SystemExit("pipeline smoke test did not produce normalized frame metadata")


def main() -> None:
    configure_environment()
    compile_package()
    import_package_modules()
    run_pipeline_smoke_test()
    print("rag_video_anh smoke checks passed")


if __name__ == "__main__":
    main()
