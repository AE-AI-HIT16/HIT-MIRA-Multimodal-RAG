"""Entrypoint pipeline OFFLINE — xử lý 1 video / 1 batch.

Điều phối: frames → (caption) → embed → index. Chạy ở worker/CLI, KHÔNG nằm trên
request path (NFR). Ví dụ:  cd api && python -m pipeline.run <video_path> [media_id]

Nhánh MEDIA (keyframe → caption → Jina-CLIP → Qdrant media) + nhánh TRANSCRIPT
(ASR → chunk → Vietnamese_Embedding → Qdrant transcript). caption/asr bật qua tham số.
Lỗi 1 mục → log, không chặn batch.
"""
from __future__ import annotations

import logging
from typing import Any

from pipeline.asr import chunk_transcript, extract_transcript
from pipeline.caption import generate_caption
from pipeline.embed import embed_images, embed_texts
from pipeline.frames import extract_keyframes
from pipeline.index import build_index

log = logging.getLogger(__name__)


def _index_media(video_path: str, media_asset_id: int, *, out_dir: str,
                 embedder: object | None, store: object | None,
                 collection: str, summary: dict[str, Any],
                 captioner: object | None = None) -> None:
    """Nhánh hình ảnh: keyframe → (caption) → embed ảnh → nạp collection media."""
    import os

    frames = extract_keyframes(video_path, media_asset_id, out_dir=out_dir)
    summary["n_frames"] = len(frames)
    frames = [f for f in frames if os.path.exists(f.frame_path)]  # vector↔payload khớp 1:1
    if not frames:
        log.warning("Không có keyframe hợp lệ cho %s → bỏ nhánh media", video_path)
        return
    vectors = embed_images([f.frame_path for f in frames], embedder=embedder)
    if len(vectors) != len(frames):
        log.error("Lệch frame↔vector (%d↔%d) → bỏ nhánh media %s",
                  len(frames), len(vectors), video_path)
        return
    # Caption làm giàu payload (opt-in). Ảnh không caption được → "" (gắn cờ review).
    captions = (
        [generate_caption(f.frame_path, captioner=captioner) for f in frames]
        if captioner is not None else [""] * len(frames)
    )
    summary["n_captioned"] = sum(1 for c in captions if c)
    ids = [f"{media_asset_id}-{i}" for i in range(len(frames))]
    payloads = [
        {"video_id": f.media_asset_id, "timestamp": f.timestamp_sec,
         "frame_path": f.frame_path, "caption": cap}
        for f, cap in zip(frames, captions, strict=True)
    ]
    build_index(collection, len(vectors[0]), ids, vectors, payloads, store=store)
    summary["n_indexed"] = len(ids)


def _index_transcript(video_path: str, media_asset_id: int, *, asr: object,
                      text_embedder: object | None, store: object | None,
                      collection: str, summary: dict[str, Any], out_dir: str) -> None:
    """Nhánh lời nói: ASR → chunk → embed text → nạp collection transcript."""
    segments = extract_transcript(video_path, media_asset_id, asr=asr, out_dir=out_dir)
    if not segments:
        log.info("Video %s không có transcript (câm/không audio) → bỏ nhánh transcript", video_path)
        return
    chunks = chunk_transcript(segments, media_asset_id)
    vectors = embed_texts([c["text"] for c in chunks], embedder=text_embedder)
    if len(vectors) != len(chunks):
        log.error("Lệch chunk↔vector transcript (%d↔%d) → bỏ", len(chunks), len(vectors))
        return
    ids = [f"{media_asset_id}-t{i}" for i in range(len(chunks))]
    build_index(collection, len(vectors[0]), ids, vectors, chunks, store=store)
    summary["n_transcript"] = len(ids)


def process_video(video_path: str, media_asset_id: int, *,
                  collection: str = "media_clip",
                  out_dir: str = "./data/frames",
                  embedder: object | None = None,
                  store: object | None = None,
                  captioner: object | None = None,
                  asr: object | None = None,
                  text_embedder: object | None = None,
                  transcript_store: object | None = None,
                  transcript_collection: str = "video_transcript") -> dict[str, Any]:
    """Keyframe→nhúng→Qdrant media; truyền `captioner`/`asr` để bật caption/transcript."""
    summary: dict[str, Any] = {
        "media_asset_id": media_asset_id, "collection": collection,
        "n_frames": 0, "n_indexed": 0, "n_captioned": 0, "n_transcript": 0,
    }

    _index_media(video_path, media_asset_id, out_dir=out_dir, embedder=embedder,
                 store=store, collection=collection, summary=summary, captioner=captioner)

    # Nhánh transcript độc lập với media (video lỗi frame vẫn có thể có lời nói).
    if asr is not None:
        _index_transcript(video_path, media_asset_id, asr=asr, text_embedder=text_embedder,
                          store=transcript_store, collection=transcript_collection,
                          summary=summary, out_dir=out_dir)

    log.info("Xong %s: %d frame → %d media, %d transcript", video_path,
             summary["n_frames"], summary["n_indexed"], summary["n_transcript"])
    return summary


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m pipeline.run <video_path> [media_asset_id]")
    media_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print(process_video(sys.argv[1], media_asset_id=media_id))
