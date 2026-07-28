"""Build debug retrieval-unit JSON for one parsed video media row."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "API"
sys.path.insert(0, str(API_ROOT))

from src.rag_video_anh.retrieval.retrieval_units import VideoRetrievalUnitBuilder  # noqa: E402


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build clean retrieval units for one parsed video. This debug script does not embed or index.",
    )
    parser.add_argument("video_media_id", help="media.media_id of the parsed video row.")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "outputs" / "retrieval_units"),
        help="Root folder for debug JSON artifacts.",
    )
    parser.add_argument(
        "--skip-minio-check",
        action="store_true",
        help="Do not stat frame images in MinIO. Intended only for local unit debugging.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    builder = VideoRetrievalUnitBuilder(check_frame_exists=not args.skip_minio_check)
    result = builder.build(args.video_media_id)

    output_dir = Path(args.output_root).expanduser().resolve() / str(args.video_media_id)
    json_dump(output_dir / "media_clip_units.json", [unit.to_dict() for unit in result.media_clip_units])
    json_dump(output_dir / "video_transcript_units.json", [unit.to_dict() for unit in result.video_transcript_units])
    json_dump(output_dir / "summary.json", result.summary.to_dict())

    print(json.dumps(result.summary.to_dict(), ensure_ascii=False, indent=2))
    print(f"Wrote retrieval-unit debug artifacts to {output_dir}")


if __name__ == "__main__":
    main()
