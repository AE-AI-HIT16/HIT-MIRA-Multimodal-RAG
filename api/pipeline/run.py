"""Entrypoint pipeline OFFLINE — xử lý 1 video / 1 batch.

Điều phối: frames → asr → caption → embed → index. Chạy ở worker/CLI, KHÔNG
nằm trên request path (NFR). Ví dụ chạy:  cd api && python -m pipeline.run <path>

Hợp đồng — sinh viên implement:
  Ghép các bước ở trên cho một media_asset; lỗi 1 bước/1 mục → log, không chặn batch.
"""
from __future__ import annotations


def process_video(video_path: str, media_asset_id: int) -> None:
    raise NotImplementedError("Ghép frames+asr+caption+embed+index cho 1 video")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m pipeline.run <video_path>")
    process_video(sys.argv[1], media_asset_id=0)
