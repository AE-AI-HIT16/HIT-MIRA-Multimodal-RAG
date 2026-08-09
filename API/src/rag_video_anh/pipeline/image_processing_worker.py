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
from collections import Counter
from pathlib import Path

from src.log.logger import logger
from src.rag_video_anh.pipeline.video_processing_worker import VideoProcessingWorker
from src.rag_video_anh.repository import MediaRecord, MediaType
from src.rag_video_anh.schemas import MediaInput, PipelineResult

# Trần cho khối lời thoại đưa vào prompt caption. Không phải để tiết kiệm token
# mà để chặn bịa: càng nhiều lời thoại không liên quan tới khung hình thì model
# càng dễ mô tả thứ nó chỉ nghe thấy.
GIOI_HAN_KY_TU_ASR = 400


class ImageProcessingWorker(VideoProcessingWorker):
    """Xử lý một hàng media ảnh: tải từ MinIO -> pipeline -> ghi DB."""

    def process_media(self, media_id: str | uuid.UUID) -> PipelineResult:
        with self.uow_factory() as uow:
            if uow.media is None:
                raise RuntimeError("RepositoryUnitOfWork did not expose the media repository")
            media = uow.media.get_media(media_id)
            if media is None:
                raise ValueError(f"media does not exist: {media_id}")
            # Keyframe do worker GPU tách ra cũng chỉ là một ảnh trên MinIO, và
            # cần đúng OCR/caption như ảnh tĩnh — GPU không giúp gì cho hai khâu
            # đó nên chạy tại chỗ bằng model ta đang cấu hình.
            if media.media_type not in {MediaType.IMAGE.value, MediaType.FRAME.value}:
                raise ValueError(
                    f"media_id {media_id} is not an image/frame row (media_type={media.media_type!r})"
                )

        suffix = Path(media.object_key).suffix or ".jpg"
        with tempfile.TemporaryDirectory(prefix="hit-mira-image-") as tmp_dir:
            local_image = Path(tmp_dir) / f"{media.media_id}{suffix}"
            self._download_object(media, local_image)

            result = self.pipeline.process(
                MediaInput(
                    media_id=str(media.media_id),
                    # Với pipeline thì keyframe không khác gì ảnh tĩnh: cùng một
                    # file trên MinIO, cùng cần OCR/caption, không có lời thoại.
                    # Giữ nguyên 'frame' thì media_validator loại ngay từ đầu.
                    media_type=MediaType.IMAGE.value,
                    source_ref=media.object_key,
                    media_path=str(local_image),
                    metadata={
                        "bucket_name": media.bucket_name,
                        "object_key": media.object_key,
                    },
                    processing_options=self._tuy_chon_xu_ly(media),
                )
            )
            self._persist_image_result(media, result)
            return result

    def _tuy_chon_xu_ly(self, media: MediaRecord) -> dict:
        """Bối cảnh cho lượt caption thứ hai — chỉ dựng khi chế độ hai lượt bật.

        Một lượt thì lượt duy nhất đó tự nhìn ảnh và tự đọc chữ, bối cảnh không
        đi tới đâu; dựng nó chỉ tốn thêm mấy truy vấn DB cho mỗi ảnh.
        """
        if not getattr(self.pipeline.vision_service, "hai_luot", False):
            return {}

        boi_canh: dict = {}
        with self.uow_factory() as uow:
            if uow.results is None or uow.media is None:
                return {}

            # Nhãn + số lượng, KHÔNG hộp và KHÔNG độ tin cậy: toạ độ không giúp
            # model viết câu mà lại mời nó chép số vào caption.
            ket_qua_object = uow.results.get_object_result(media.media_id)
            if ket_qua_object is not None and ket_qua_object.objects:
                dem: Counter = Counter(
                    str(vat.label).strip() for vat in ket_qua_object.objects if str(vat.label or "").strip()
                )
                if dem:
                    boi_canh["object_counts"] = dict(dem)

            # Lời thoại chỉ có với keyframe; ảnh tĩnh không có video cha nên
            # nhánh này tự bỏ qua.
            frame = uow.media.get_frame_by_media_id(media.media_id)
            if frame is not None and frame.video_media_id is not None:
                loi_noi = self._loi_noi_quanh_khung_hinh(uow, frame)
                if loi_noi:
                    boi_canh["asr"] = loi_noi

        return {"caption": {"context": boi_canh}} if boi_canh else {}

    @staticmethod
    def _loi_noi_quanh_khung_hinh(uow, frame) -> str:
        """Lời nói quanh thời điểm khung hình, cắt cứng ở GIOI_HAN_KY_TU_ASR.

        Cửa sổ "một đoạn trước + đoạn hiện tại + một đoạn sau" của nhánh truy hồi
        KHÔNG dùng lại được ở đây. Đo trên kho này: 59 video chỉ có 245 đoạn, tức
        Zipformer gộp cả bài nói vào 3-4 đoạn rất dài — lấy ba đoạn là lấy trọn
        ~2.000 từ. Nhồi ngần đó lời thoại vào prompt caption thì model gần như
        chắc chắn kể lại việc chỉ NGHE thấy chứ không THẤY, đúng thứ mà kho này
        cấm tuyệt đối.

        Nên với đoạn dài thì cắt một cửa sổ quanh vị trí nội suy của khung hình
        trong đoạn: thô, nhưng đúng hướng và bị chặn trên.
        """
        transcript = uow.results.get_transcript_by_video_media_id(frame.video_media_id)
        if transcript is None or not transcript.segments:
            return ""
        # `TranscriptSegmentRecord` dùng start_time/end_time, không phải
        # start_sec/end_sec như `CleanTranscriptSegment` bên nhánh truy hồi.
        moc = float(frame.timestamp or 0.0)
        doan = sorted(transcript.segments, key=lambda seg: float(seg.start_time or 0.0))
        vi_tri = next(
            (i for i, seg in enumerate(doan) if float(seg.start_time or 0.0) <= moc <= float(seg.end_time or 0.0)),
            None,
        )
        if vi_tri is None:
            return ""

        hien_tai = doan[vi_tri]
        chu = str(hien_tai.text or "").strip()
        if len(chu) <= GIOI_HAN_KY_TU_ASR:
            lan_can = doan[max(0, vi_tri - 1) : vi_tri + 2]
            return " ".join(str(seg.text or "").strip() for seg in lan_can).strip()[:GIOI_HAN_KY_TU_ASR]

        bat_dau = float(hien_tai.start_time or 0.0)
        dai = float(hien_tai.end_time or 0.0) - bat_dau
        ty_le = 0.0 if dai <= 0 else max(0.0, min(1.0, (moc - bat_dau) / dai))
        giua = int(ty_le * len(chu))
        dau = max(0, min(len(chu) - GIOI_HAN_KY_TU_ASR, giua - GIOI_HAN_KY_TU_ASR // 2))
        return chu[dau : dau + GIOI_HAN_KY_TU_ASR].strip()

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
