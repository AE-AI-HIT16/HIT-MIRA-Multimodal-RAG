"""Upload crawled event folders to MinIO.

Example:
    python scripts/upload_crawled_data_to_minio.py ./data/crawled --prefix events
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "API"
sys.path.insert(0, str(API_ROOT))

from src.rag_video_anh.pipeline.minio_storage import MinioStorage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload crawled event data to MinIO.")
    parser.add_argument(
        "data_root",
        help="Folder containing event folders, or one event folder containing post.json.",
    )
    parser.add_argument(
        "--prefix",
        default="events",
        help="Object key prefix in MinIO. Default: events",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    storage = MinioStorage()
    uploaded_keys = storage.upload_crawled_dataset(args.data_root, base_prefix=args.prefix)
    print(f"Uploaded {len(uploaded_keys)} files to bucket '{storage.bucket_name}'.")
    for key in uploaded_keys:
        print(key)


if __name__ == "__main__":
    main()
