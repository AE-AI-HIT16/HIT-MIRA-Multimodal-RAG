"""Cách quy đổi trạng thái và nội dung khi nạp artifact từ worker GPU.

Worker GPU có thể tắt bớt khâu (`--disable-caption`) để chạy tại chỗ bằng model
khác. Khi đó artifact vẫn có mục cho từng khung, nhưng mang status SKIPPED kèm
`reason`. Bản đầu ghi thẳng reason thành caption và đánh dấu DONE, nên 1.646
keyframe có caption là chuỗi "caption disabled by route" — chuỗi đó đi vào
payload Qdrant và hiện ra như một trích dẫn cho người đọc.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.import_media_outputs import status  # noqa: E402

from src.rag_video_anh.repository.schemas import ProcessingStatus  # noqa: E402


def test_skipped_stage_is_not_recorded_as_done() -> None:
    """Bỏ qua nghĩa là chưa chạy, phải để lượt sau nhặt lại."""
    assert status("SKIPPED") == ProcessingStatus.PENDING.value


def test_not_found_stays_done() -> None:
    """Đã phân tích nhưng không có gì để lấy vẫn là kết quả hợp lệ."""
    assert status("NOT_FOUND") == ProcessingStatus.DONE.value


def test_error_maps_to_failed() -> None:
    assert status("ERROR") == ProcessingStatus.FAILED.value
    assert status("FAILED") == ProcessingStatus.FAILED.value


def test_done_and_success_map_to_done() -> None:
    assert status("DONE") == ProcessingStatus.DONE.value
    assert status("SUCCESS") == ProcessingStatus.DONE.value


def test_caption_import_never_uses_reason_as_caption_text() -> None:
    """Chốt bằng chính mã nguồn: reason không được là nguồn của caption_text."""
    source = (PROJECT_ROOT / "scripts" / "import_media_outputs.py").read_text(encoding="utf-8")
    start = source.index("def _import_captions")
    end = source.index("def _import_detections")
    block = source[start:end]

    assert "caption_text=item.get(\"caption_text\") or item.get(\"caption\")" in block
    assert "reason" not in block.split("caption_text=")[1].split("\n")[0]
