"""Run the current OCR pipeline against keyframe images stored in MinIO.

Examples:
    python scripts/test_minio_keyframe_ocr.py --prefix frames --limit 10
    python scripts/test_minio_keyframe_ocr.py --object-key frames/<media_id>/video_frame_000001.jpg
    python scripts/test_minio_keyframe_ocr.py --prefix frames/<media_id> --output /tmp/ocr_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "API"
sys.path.insert(0, str(API_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.rag_video_anh.pipeline.minio_storage import MinioStorage  # noqa: E402
from src.rag_video_anh.pipeline.ocr_service import OCRService  # noqa: E402
from src.rag_video_anh.schemas import KeyFrame, KeyFrameSet, OCRRequest, StageStatus  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def list_image_keys(storage: MinioStorage, prefix: str, limit: int | None) -> list[str]:
    keys: list[str] = []
    for obj in storage.client.list_objects(storage.bucket_name, prefix=prefix, recursive=True):
        object_name = obj.object_name
        if Path(object_name).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        keys.append(object_name)
        if limit is not None and len(keys) >= limit:
            break
    return keys


def frame_index_from_key(object_key: str, fallback: int) -> int:
    stem = Path(object_key).stem
    tail = stem.rsplit("_", 1)[-1]
    if tail.isdigit():
        return int(tail)
    return fallback


def configure_ocr_service(args: argparse.Namespace) -> OCRService:
    service = OCRService()
    if args.cpu:
        service.model_config.ocr_gpu = False
    if args.gpu:
        service.model_config.ocr_gpu = True
    if args.disable_preprocessing:
        service.pipeline_config.ocr_enable_preprocessing = False
    if args.upscale_factors:
        service.pipeline_config.ocr_upscale_factors = parse_float_list(args.upscale_factors)
    if args.min_confidence is not None:
        service.pipeline_config.ocr_min_confidence = args.min_confidence
    if args.disable_cleanup:
        service.pipeline_config.ocr_text_cleanup = False
    return service


def result_to_dict(result: Any, object_keys: list[str]) -> dict[str, Any]:
    items = []
    key_by_frame_id = {f"minio_keyframe_f{index:06d}": key for index, key in enumerate(object_keys)}
    for item in result.results:
        spans = [
            {
                "text": span.text,
                "confidence": span.confidence,
                "bbox": span.bbox,
                "metadata": span.metadata,
            }
            for span in item.text_spans
        ]
        items.append(
            {
                "object_key": key_by_frame_id.get(item.frame_id),
                "frame_id": item.frame_id,
                "status": item.status.value,
                "reason": item.reason,
                "confidence": item.confidence,
                "full_text": item.full_text,
                "spans": spans,
            }
        )
    return {
        "media_id": result.media_id,
        "status": result.status.value,
        "reason": result.reason,
        "count": len(items),
        "results": items,
    }


def print_human_summary(payload: dict[str, Any]) -> None:
    print(f"OCR status: {payload['status']} | frames: {payload['count']}")
    for item in payload["results"]:
        print("=" * 100)
        print(item["object_key"])
        print(f"status={item['status']} confidence={item['confidence']} reason={item['reason']}")
        print(item["full_text"] or "<no text>")
        for span in item["spans"]:
            print(f"  - {span['text']} | conf={span['confidence']} | meta={span['metadata']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test OCR on keyframe images stored in MinIO.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prefix", help="MinIO prefix to scan, for example: frames or frames/<media_id>.")
    source.add_argument("--object-key", action="append", help="Specific MinIO object key. Repeat for multiple frames.")
    parser.add_argument("--bucket", help="Override bucket name from config.")
    parser.add_argument("--limit", type=int, help="Maximum number of images to process when using --prefix.")
    parser.add_argument("--output", help="Write full OCR payload to this JSON file.")
    parser.add_argument("--download-dir", help="Directory for downloaded keyframes. Defaults to a temporary directory.")
    parser.add_argument("--keep-downloads", action="store_true", help="Keep temporary downloads after OCR finishes.")
    parser.add_argument("--cpu", action="store_true", help="Force OCR on CPU.")
    parser.add_argument("--gpu", action="store_true", help="Force OCR on GPU.")
    parser.add_argument("--disable-preprocessing", action="store_true", help="Use original images only.")
    parser.add_argument("--upscale-factors", help="Comma-separated upscale factors, for example: 1,2,3.")
    parser.add_argument("--min-confidence", type=float, help="Override OCR minimum confidence.")
    parser.add_argument("--disable-cleanup", action="store_true", help="Disable Vietnamese OCR text cleanup.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cpu and args.gpu:
        raise SystemExit("Choose only one of --cpu or --gpu.")

    storage = MinioStorage()
    if args.bucket:
        storage.bucket_name = args.bucket

    object_keys = args.object_key or list_image_keys(storage, args.prefix, args.limit)
    if not object_keys:
        raise SystemExit("No image objects found. Check --prefix, --bucket, or MinIO credentials.")

    download_root = Path(args.download_dir) if args.download_dir else Path(tempfile.mkdtemp(prefix="minio-ocr-"))
    download_root.mkdir(parents=True, exist_ok=True)

    local_paths: list[Path] = []
    try:
        for index, object_key in enumerate(object_keys):
            suffix = Path(object_key).suffix or ".jpg"
            local_path = download_root / f"keyframe_{index:06d}{suffix}"
            storage.download_file(object_key, local_path)
            local_paths.append(local_path)

        frames = [
            KeyFrame(
                frame_id=f"minio_keyframe_f{index:06d}",
                media_id="minio_keyframe_ocr_test",
                frame_index=frame_index_from_key(object_keys[index], index),
                timestamp_ms=0,
                timestamp_sec=0.0,
                image_path=str(path),
                metadata={"bucket_name": storage.bucket_name, "object_key": object_keys[index]},
            )
            for index, path in enumerate(local_paths)
        ]

        service = configure_ocr_service(args)
        result = service.recognize(
            OCRRequest(
                media_id="minio_keyframe_ocr_test",
                keyframes=KeyFrameSet(
                    media_id="minio_keyframe_ocr_test",
                    frames=frames,
                    status=StageStatus.DONE,
                ),
            )
        )
        payload = result_to_dict(result, object_keys)
        print_human_summary(payload)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote JSON result to {output_path}")
    finally:
        if args.download_dir is None and not args.keep_downloads:
            for path in local_paths:
                path.unlink(missing_ok=True)
            download_root.rmdir()


if __name__ == "__main__":
    main()
