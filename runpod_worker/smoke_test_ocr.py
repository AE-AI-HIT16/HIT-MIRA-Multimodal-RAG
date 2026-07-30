"""Smoke test Qwen vision OCR/caption analysis inside the production RunPod image."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "API"
sys.path.insert(0, str(API_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.configuration import AppConfig  # noqa: E402
from src.rag_video_anh.pipeline.qwen_vision_service import QwenVisionService  # noqa: E402
from src.rag_video_anh.schemas import KeyFrame, KeyFrameSet, StageStatus  # noqa: E402


def _download_image(url: str, target: Path) -> Path:
    response = httpx.get(url, follow_redirects=True, timeout=120.0)
    response.raise_for_status()
    target.write_bytes(response.content)
    return target


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _create_builtin_image(target: Path) -> Path:
    image = Image.new("RGB", (1280, 720), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(64)
    subtitle_font = _font(44)
    draw.rectangle((70, 80, 1210, 250), outline="black", width=4)
    draw.text((100, 115), "RUNPOD QWEN VISION TEST", fill="black", font=title_font)
    draw.text((100, 300), "OCR and caption API check", fill="black", font=subtitle_font)
    draw.text((100, 370), "Vietnamese: Kiem tra pipeline hinh anh", fill="black", font=subtitle_font)

    draw.rectangle((760, 330, 1160, 610), fill="#f0f4f8", outline="#1f2937", width=3)
    draw.rectangle((815, 380, 1055, 510), fill="#dbeafe", outline="#2563eb", width=3)
    draw.rectangle((895, 510, 975, 545), fill="#1f2937")
    draw.rectangle((850, 545, 1020, 565), fill="#374151")
    draw.ellipse((205, 430, 275, 500), fill="#facc15", outline="#92400e", width=3)
    draw.rectangle((225, 500, 255, 610), fill="#60a5fa", outline="#1d4ed8", width=3)
    draw.line((225, 530, 160, 585), fill="#1d4ed8", width=8)
    draw.line((255, 530, 330, 585), fill="#1d4ed8", width=8)
    draw.line((240, 610, 200, 680), fill="#111827", width=8)
    draw.line((240, 610, 285, 680), fill="#111827", width=8)
    draw.rectangle((355, 560, 690, 625), fill="#9ca3af", outline="#374151", width=3)
    draw.line((675, 560, 675, 470), fill="#111827", width=6)
    draw.ellipse((655, 445, 695, 485), fill="#111827")
    image.save(target)
    return target


def _build_keyframes(image_path: Path) -> KeyFrameSet:
    return KeyFrameSet(
        media_id="runpod_qwen_vision_smoke_test",
        frames=[
            KeyFrame(
                frame_id="runpod_qwen_vision_smoke_test_f000000",
                media_id="runpod_qwen_vision_smoke_test",
                frame_index=0,
                timestamp_ms=0,
                timestamp_sec=0.0,
                image_path=str(image_path),
                metadata={"source": "qwen_vision_smoke_test"},
            )
        ],
        status=StageStatus.DONE,
    )


def _result_payload(ocr_result: Any, caption_result: Any, elapsed_ms: float) -> dict[str, Any]:
    ocr_frame = ocr_result.results[0] if ocr_result.results else None
    caption_frame = caption_result.results[0] if caption_result.results else None
    return {
        "ocr_status": getattr(ocr_frame.status, "value", str(ocr_frame.status)) if ocr_frame else getattr(ocr_result.status, "value", str(ocr_result.status)),
        "caption_status": getattr(caption_frame.status, "value", str(caption_frame.status)) if caption_frame else getattr(caption_result.status, "value", str(caption_result.status)),
        "ocr_reason": ocr_frame.reason if ocr_frame else getattr(ocr_result, "reason", None),
        "caption_reason": caption_frame.reason if caption_frame else getattr(caption_result, "reason", None),
        "ocr_text": ocr_frame.full_text if ocr_frame else "",
        "caption_text": caption_frame.caption_text if caption_frame else "",
        "caption_model": caption_frame.generation_meta.get("model") if caption_frame else None,
        "elapsed_ms": elapsed_ms,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test Qwen vision in the RunPod production image.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", help="Local image path inside the container.")
    source.add_argument("--image-url", help="HTTP(S) URL or presigned MinIO URL for one keyframe image.")
    parser.add_argument("--repeat", type=int, default=1, help="Run analysis repeatedly to catch unstable API/runtime failures.")
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = AppConfig()
    service = QwenVisionService(config=config)

    with tempfile.TemporaryDirectory(prefix="qwen-vision-smoke-") as temp_dir:
        temp_path = Path(temp_dir)
        if args.image_url:
            image_path = _download_image(args.image_url, temp_path / "input.jpg")
        elif args.image:
            image_path = Path(args.image)
        else:
            image_path = _create_builtin_image(temp_path / "builtin.png")

        payloads: list[dict[str, Any]] = []
        for _ in range(max(1, args.repeat)):
            started = time.perf_counter()
            ocr_result, caption_result = service.analyze(
                media_id="runpod_qwen_vision_smoke_test",
                keyframes=_build_keyframes(image_path),
            )
            payloads.append(_result_payload(ocr_result, caption_result, (time.perf_counter() - started) * 1000.0))

    failures = []
    for index, payload in enumerate(payloads, start=1):
        ocr_status = str(payload["ocr_status"]).lower()
        caption_status = str(payload["caption_status"]).lower()
        if ocr_status in {"error", "skipped"} or caption_status in {"error", "skipped"}:
            failures.append(
                f"run {index}: Qwen vision did not complete "
                f"(ocr_status={payload['ocr_status']}, caption_status={payload['caption_status']})"
            )
        if not payload["ocr_text"]:
            failures.append(f"run {index}: Qwen vision returned no OCR text")
        if not payload["caption_text"]:
            failures.append(f"run {index}: Qwen vision returned no caption text")
    output = {
        "ok": not failures,
        "config": {"vision_model_name": config.media_models.vision_model_name},
        "runs": payloads,
        "failures": failures,
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    return 0 if output["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
