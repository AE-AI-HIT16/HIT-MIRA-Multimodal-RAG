"""Run the local video pipeline and export portable artifacts.

This script is intentionally storage-agnostic: it does not talk to PostgreSQL
or MinIO. Use it on a GPU server to process one local video into:

    outputs/<media_id>/
      manifest.json
      frames/*.jpg
      ocr.json
      captions.json
      detections.json
      transcript.json
      normalized_metadata.json
      run_log.txt

Then run scripts/import_media_outputs.py on the server to persist the
artifact into MinIO and PostgreSQL.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "API"
sys.path.insert(0, str(API_ROOT))

from src.configuration import AppConfig  # noqa: E402
from src.rag_video_anh.pipeline.pipeline_service import PipelineService  # noqa: E402
from src.rag_video_anh.schemas import MediaInput, PipelineResult  # noqa: E402


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def stage_status(stage: Any | None) -> str:
    if stage is None:
        return "skipped"
    status = getattr(stage, "status", None)
    return getattr(status, "value", status) or "skipped"


def stage_reason(stage: Any | None) -> str | None:
    return getattr(stage, "reason", None) if stage is not None else None


def write_frame_image(frame: Any, frames_dir: Path) -> str:
    frames_dir.mkdir(parents=True, exist_ok=True)
    filename = f"frame_{int(frame.frame_index):06d}.jpg"
    target = frames_dir / filename

    if frame.image_payload is not None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required to write keyframe images") from exc
        success = cv2.imwrite(str(target), frame.image_payload)
        if not success:
            raise RuntimeError(f"failed to write keyframe image: {target}")
    elif frame.image_path:
        shutil.copy2(frame.image_path, target)
    else:
        raise ValueError(f"keyframe {frame.frame_id} has no image payload or path")

    return f"frames/{filename}"


def result_by_frame_id(results: list[Any]) -> dict[str, Any]:
    return {str(result.frame_id): result for result in results}


def export_manifest(
    result: PipelineResult,
    run_dir: Path,
    video_path: Path,
    bucket_name: str,
    source_object_key: str,
) -> None:
    keyframes = result.keyframes
    frames = []
    for frame in keyframes.frames if keyframes else []:
        relative_file = write_frame_image(frame, run_dir / "frames")
        frames.append(
            {
                "frame_id": frame.frame_id,
                "frame_index": frame.frame_index,
                "timestamp_sec": frame.timestamp_sec,
                "timestamp_ms": frame.timestamp_ms,
                "quality_score": frame.quality_score,
                "dedup_score": frame.dedup_score,
                "selection_reason": frame.selection_reason,
                "file": relative_file,
            }
        )

    selection = keyframes.selection_summary if keyframes else {}
    coverage = keyframes.coverage_summary if keyframes else {}
    raw_frames = selection.get("total_frames")
    fps = coverage.get("fps")
    duration = (float(raw_frames) / float(fps)) if raw_frames is not None and fps else None
    manifest = {
        "media_id": result.media_input.media_id,
        "bucket_name": bucket_name,
        "source_object_key": source_object_key,
        "video_file": video_path.name,
        "status": result.status.value,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "video": {
            "frame_count": raw_frames,
            "fps": fps,
            "duration_sec": duration,
            "width": result.media_input.metadata.get("width"),
            "height": result.media_input.metadata.get("height"),
        },
        "frames": frames,
        "outputs": {
            "ocr": "ocr.json",
            "captions": "captions.json",
            "detections": "detections.json",
            "transcript": "transcript.json",
            "normalized_metadata": "normalized_metadata.json",
        },
    }
    json_dump(run_dir / "manifest.json", manifest)


def export_ocr(result: PipelineResult, run_dir: Path, config: AppConfig) -> None:
    ocr = result.ocr_results
    payload = {
        "media_id": result.media_input.media_id,
        "status": stage_status(ocr),
        "reason": stage_reason(ocr),
        "model": config.media_models.vision_model_name,
        "results": [],
    }
    for item in ocr.results if ocr else []:
        payload["results"].append(
            {
                "frame_id": item.frame_id,
                "status": stage_status(item),
                "reason": item.reason,
                "full_text": item.full_text,
            }
        )
    json_dump(run_dir / "ocr.json", payload)


def export_captions(result: PipelineResult, run_dir: Path, config: AppConfig) -> None:
    captions = result.caption_results
    payload = {
        "media_id": result.media_input.media_id,
        "status": stage_status(captions),
        "reason": stage_reason(captions),
        "caption_model": config.media_models.vision_model_name,
        "results": [],
    }
    for item in captions.results if captions else []:
        payload["results"].append(
            {
                "frame_id": item.frame_id,
                "status": stage_status(item),
                "reason": item.reason,
                "caption_text": item.caption_text,
                "generation_meta": item.generation_meta,
                "vision_metadata": {
                    key: item.generation_meta.get(key)
                    for key in ("ocr_blocks", "scene", "objects", "activities", "keywords")
                    if item.generation_meta.get(key) not in (None, "", [])
                },
                "prompt_version": item.prompt_version,
            }
        )
    json_dump(run_dir / "captions.json", payload)


def export_detections(result: PipelineResult, run_dir: Path, config: AppConfig) -> None:
    detections = result.detection_results
    payload = {
        "media_id": result.media_input.media_id,
        "status": stage_status(detections),
        "reason": stage_reason(detections),
        "model": config.media_models.detection_weight_file,
        "results": [],
    }
    for item in detections.results if detections else []:
        payload["results"].append(
            {
                "frame_id": item.frame_id,
                "status": stage_status(item),
                "reason": item.reason,
                "inference_meta": item.inference_meta,
                "detections": [
                    {
                        "label": detected.label,
                        "confidence": detected.confidence,
                        "bbox": detected.bbox,
                        "metadata": detected.metadata,
                    }
                    for detected in item.detections
                ],
            }
        )
    json_dump(run_dir / "detections.json", payload)


def export_transcript(result: PipelineResult, run_dir: Path, config: AppConfig) -> None:
    transcript = result.transcript_set
    full_text = " ".join(segment.text for segment in transcript.segments if segment.text).strip() if transcript else ""
    payload = {
        "media_id": result.media_input.media_id,
        "status": stage_status(transcript),
        "reason": stage_reason(transcript),
        "language": getattr(transcript, "language", None),
        "model": (transcript.transcription_meta.get("model") if transcript else None) or config.media_models.asr_model_name,
        "full_text": full_text,
        "transcription_meta": getattr(transcript, "transcription_meta", {}),
        "segments": [],
    }
    for segment in transcript.segments if transcript else []:
        payload["segments"].append(
            {
                "segment_id": segment.segment_id,
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "start_sec": segment.start_sec,
                "end_sec": segment.end_sec,
                "text": segment.text,
                "confidence": segment.confidence,
                "language": segment.language,
            }
        )
    json_dump(run_dir / "transcript.json", payload)


def export_normalized_metadata(result: PipelineResult, run_dir: Path) -> None:
    if result.normalized_metadata is None:
        json_dump(
            run_dir / "normalized_metadata.json",
            {
                "media_id": result.media_input.media_id,
                "status": "skipped",
                "reason": "normalization did not produce metadata",
            },
        )
        return
    json_dump(run_dir / "normalized_metadata.json", result.normalized_metadata.model_dump(mode="json"))


def export_run_log(result: PipelineResult, run_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        f"media_id={result.media_input.media_id}",
        f"video_path={args.video_path}",
        f"status={result.status.value}",
        f"keyframes={stage_status(result.keyframes)} count={len(result.keyframes.frames) if result.keyframes else 0}",
        f"ocr={stage_status(result.ocr_results)} reason={stage_reason(result.ocr_results)}",
        f"caption={stage_status(result.caption_results)} reason={stage_reason(result.caption_results)}",
        f"detection={stage_status(result.detection_results)} reason={stage_reason(result.detection_results)}",
        f"asr={stage_status(result.transcript_set)} reason={stage_reason(result.transcript_set)}",
    ]
    for error in result.errors:
        lines.append(f"error stage={error.stage} code={error.error_code} message={error.message}")
    (run_dir / "run_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export one video pipeline run as portable artifacts.")
    parser.add_argument("--video-path", required=True, help="Local path to the video file.")
    parser.add_argument("--media-id", required=True, help="PostgreSQL media.media_id for this video.")
    parser.add_argument("--output-root", default="outputs", help="Directory that will contain <media_id>/ artifacts.")
    parser.add_argument("--bucket-name", default="hit-mira-media", help="Target MinIO bucket name for manifest metadata.")
    parser.add_argument("--source-object-key", help="Original MinIO object key for the video. Defaults to the video filename.")
    parser.add_argument("--language", default="vi", help="ASR language hint; use empty string for auto-detect.")
    parser.add_argument("--disable-ocr", action="store_true", help="Skip OCR.")
    parser.add_argument("--disable-caption", action="store_true", help="Skip frame captioning.")
    parser.add_argument("--disable-detection", action="store_true", help="Skip object detection.")
    parser.add_argument("--disable-asr", action="store_true", help="Skip ASR.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    video_path = Path(args.video_path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"video file does not exist: {video_path}")

    config = AppConfig()
    config.media_pipeline.enable_ocr = not args.disable_ocr
    config.media_pipeline.enable_caption = not args.disable_caption
    config.media_pipeline.enable_detection = not args.disable_detection
    config.media_pipeline.enable_asr = not args.disable_asr
    config.media_models.asr_language = args.language.strip() or None

    run_dir = Path(args.output_root).expanduser().resolve() / str(args.media_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        pipeline = PipelineService(config=config)
        result = pipeline.process(
            MediaInput(
                media_id=str(args.media_id),
                media_type="video",
                source_ref=args.source_object_key or video_path.name,
                media_path=str(video_path),
                language_hint=config.media_models.asr_language,
                metadata={
                    "path": str(video_path),
                    "bucket_name": args.bucket_name,
                    "object_key": args.source_object_key or video_path.name,
                },
            )
        )
        export_manifest(result, run_dir, video_path, args.bucket_name, args.source_object_key or video_path.name)
        export_ocr(result, run_dir, config)
        export_captions(result, run_dir, config)
        export_detections(result, run_dir, config)
        export_transcript(result, run_dir, config)
        export_normalized_metadata(result, run_dir)
        export_run_log(result, run_dir, args)
        print(f"Exported artifacts to {run_dir}")
        print(f"Pipeline status: {result.status.value}")
    except Exception:
        (run_dir / "run_log.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
