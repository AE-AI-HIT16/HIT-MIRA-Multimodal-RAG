"""Worker chạy pipeline cho ảnh tĩnh và ghi kết quả vào PostgreSQL.

Khác video ở ba điểm:
  * Ảnh KHÔNG sinh bản ghi `videos` (không có duration/fps) và KHÔNG sinh bản
    ghi `frames` — chính nó đã là đơn vị hình ảnh, nên OCR/caption/detection
    gắn thẳng vào `media_id` của ảnh.
  * Không có nhánh ASR.
  * Không upload gì thêm lên MinIO: ảnh gốc đã nằm sẵn ở đó.

Phần ghi OCR/caption/detection dùng lại nguyên của `VideoProcessingWorker` vì
các hàm đó chỉ cần ánh xạ frame_id -> media_id, không phụ thuộc vào video.
"""

from __future__ import annotations

import argparse
import tempfile
import uuid
from pathlib import Path

from src.log.logger import logger
from src.rag_video_anh.pipeline.video_processing_worker import VideoProcessingWorker
from src.rag_video_anh.repository import MediaRecord, MediaType
from src.rag_video_anh.schemas import MediaInput, PipelineResult


class ImageProcessingWorker(VideoProcessingWorker):
    """Xử lý một hàng media ảnh: tải từ MinIO -> pipeline -> ghi DB."""

    def process_media(self, media_id: str | uuid.UUID) -> PipelineResult:
        with self.uow_factory() as uow:
            if uow.media is None:
                raise RuntimeError("RepositoryUnitOfWork did not expose the media repository")
            media = uow.media.get_media(media_id)
            if media is None:
                raise ValueError(f"media does not exist: {media_id}")
            if media.media_type != MediaType.IMAGE.value:
                raise ValueError(f"media_id {media_id} is not an image row (media_type={media.media_type!r})")

        suffix = Path(media.object_key).suffix or ".jpg"
        with tempfile.TemporaryDirectory(prefix="hit-mira-image-") as tmp_dir:
            local_image = Path(tmp_dir) / f"{media.media_id}{suffix}"
            self._download_object(media, local_image)

            result = self.pipeline.process(
                MediaInput(
                    media_id=str(media.media_id),
                    media_type=media.media_type,
                    source_ref=media.object_key,
                    media_path=str(local_image),
                    metadata={
                        "bucket_name": media.bucket_name,
                        "object_key": media.object_key,
                    },
                )
            )
            self._persist_image_result(media, result)
            return result

    def _persist_image_result(self, media: MediaRecord, result: PipelineResult) -> None:
        """Gắn kết quả phân tích thẳng vào chính hàng media của ảnh."""
        frame_media_ids = {frame.frame_id: media.media_id for frame in self._result_frames(result)}
        if not frame_media_ids:
            logger.warning(f"Image '{media.media_id}' produced no analysable frame; nothing to persist")
            return

        with self.uow_factory() as uow:
            if uow.results is None:
                raise RuntimeError("RepositoryUnitOfWork did not expose the results repository")
            self._persist_ocr(uow, result, frame_media_ids)
            self._persist_captions(uow, result, frame_media_ids)
            self._persist_detections(uow, result, frame_media_ids)
        logger.info(f"Persisted image analysis for media_id '{media.media_id}'")

    @staticmethod
    def _result_frames(result: PipelineResult) -> list:
        if result.keyframes is None:
            return []
        return list(result.keyframes.frames)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chạy pipeline phân tích cho một hàng media ảnh.")
    parser.add_argument("media_id", help="media_id của ảnh trong bảng media")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = ImageProcessingWorker().process_media(args.media_id)
    print(f"status={result.status.value}")
    for error in result.errors:
        print(f"  lỗi ở '{error.stage}': {error.message}")


if __name__ == "__main__":
    main()
