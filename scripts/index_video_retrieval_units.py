"""Embed and index clean video retrieval units for one parsed video media row."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "API"
sys.path.insert(0, str(API_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from src.rag_video_anh.retrieval.indexing_service import (  # noqa: E402
    DEFAULT_MEDIA_CLIP_COLLECTION,
    DEFAULT_VIDEO_TRANSCRIPT_COLLECTION,
    VideoRetrievalIndexingService,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build clean retrieval units from PostgreSQL/MinIO, embed them, "
            "and upsert media_clip/video_transcript points into Qdrant."
        ),
    )
    parser.add_argument("video_media_id", help="media.media_id of the parsed video row.")
    parser.add_argument(
        "--media-clip-collection",
        default=DEFAULT_MEDIA_CLIP_COLLECTION,
        help="Qdrant collection for keyframe image vectors.",
    )
    parser.add_argument(
        "--video-transcript-collection",
        default=DEFAULT_VIDEO_TRANSCRIPT_COLLECTION,
        help="Qdrant collection for transcript text vectors.",
    )
    parser.add_argument(
        "--payload-only",
        action="store_true",
        help=(
            "Chỉ ghi lại payload (caption/OCR/source_url), KHÔNG nhúng lại. "
            "Dùng cho video ĐÃ index khi chỉ cần bổ sung một khoá payload."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    indexer = VideoRetrievalIndexingService(
        media_clip_collection=args.media_clip_collection,
        video_transcript_collection=args.video_transcript_collection,
    )
    if args.payload_only:
        summary = indexer.refresh_video_payloads(args.video_media_id)
    else:
        summary = indexer.index_video(args.video_media_id)
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
