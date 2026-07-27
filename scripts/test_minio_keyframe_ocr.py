"""Run Qwen vision OCR/caption analysis against keyframes stored in MinIO.

Examples:
    python scripts/test_minio_keyframe_ocr.py --prefix frames --limit 10
    python scripts/test_minio_keyframe_ocr.py --object-key frames/<media_id>/video_frame_000001.jpg
    python scripts/test_minio_keyframe_ocr.py --prefix frames/<media_id> --output /tmp/vision_results.json
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
from src.rag_video_anh.pipeline.qwen_vision_service import QwenVisionService  # noqa: E402
from src.rag_video_anh.schemas import KeyFrame, KeyFrameSet, StageStatus  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


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


def result_to_dict(ocr_result: Any, caption_result: Any, object_keys: list[str]) -> dict[str, Any]:
    key_by_frame_id = {f"minio_keyframe_f{index:06d}": key for index, key in enumerate(object_keys)}
    captions = {item.frame_id: item for item in caption_result.results}
    items = []
    for item in ocr_result.results:
        caption = captions.get(item.frame_id)
        items.append(
            {
                "object_key": key_by_frame_id.get(item.frame_id),
                "frame_id": item.frame_id,
                "ocr_status": item.status.value,
                "ocr_reason": item.reason,
                "ocr_text": item.full_text,
                "caption_status": caption.status.value if caption else None,
                "caption_reason": caption.reason if caption else None,
                "caption_text": caption.caption_text if caption else "",
                "caption_model": (caption.generation_meta.get("model") if caption else None),
            }
        )
    return {
        "media_id": ocr_result.media_id,
        "ocr_status": ocr_result.status.value,
        "caption_status": caption_result.status.value,
        "count": len(items),
        "results": items,
    }


def print_human_summary(payload: dict[str, Any]) -> None:
    print(f"Vision status: ocr={payload['ocr_status']} caption={payload['caption_status']} | frames: {payload['count']}")
    for item in payload["results"]:
        print("=" * 100)
        print(item["object_key"])
        print(f"ocr={item['ocr_status']} caption={item['caption_status']} model={item['caption_model']}")
        print(item["ocr_text"] or "<no text>")
        print(item["caption_text"] or "<no caption>")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test Qwen vision OCR/caption analysis on keyframe images stored in MinIO.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prefix", help="MinIO prefix to scan, for example: frames or frames/<media_id>.")
    source.add_argument("--object-key", action="append", help="Specific MinIO object key. Repeat for multiple frames.")
    parser.add_argument("--bucket", help="Override bucket name from config.")
    parser.add_argument("--limit", type=int, help="Maximum number of images to process when using --prefix.")
    parser.add_argument("--output", help="Write full Qwen vision payload to this JSON file.")
    parser.add_argument("--download-dir", help="Directory for downloaded keyframes. Defaults to a temporary directory.")
    parser.add_argument("--keep-downloads", action="store_true", help="Keep temporary downloads after analysis finishes.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    storage = MinioStorage()
    if args.bucket:
        storage.bucket_name = args.bucket

    object_keys = args.object_key or list_image_keys(storage, args.prefix, args.limit)
    if not object_keys:
        raise SystemExit("No image objects found. Check --prefix, --bucket, or MinIO credentials.")

    download_root = Path(args.download_dir) if args.download_dir else Path(tempfile.mkdtemp(prefix="minio-vision-"))
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
                media_id="minio_keyframe_vision_test",
                frame_index=frame_index_from_key(object_keys[index], index),
                timestamp_ms=0,
                timestamp_sec=0.0,
                image_path=str(path),
                metadata={"bucket_name": storage.bucket_name, "object_key": object_keys[index]},
            )
            for index, path in enumerate(local_paths)
        ]

        service = QwenVisionService()
        ocr_result, caption_result = service.analyze(
            media_id="minio_keyframe_vision_test",
            keyframes=KeyFrameSet(media_id="minio_keyframe_vision_test", frames=frames, status=StageStatus.DONE),
        )
        payload = result_to_dict(ocr_result, caption_result, object_keys)
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
