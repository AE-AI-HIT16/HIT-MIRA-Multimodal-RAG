"""Skeleton test cho các phần LÕI HỌC (đang skip).

Sinh viên: implement hàm tương ứng rồi bỏ @pytest.mark.skip, viết assert theo
Acceptance Criteria trong docs/prd.md. Mỗi test map tới 1 TC-xxx.
"""
from __future__ import annotations

import pytest


@pytest.mark.skip(reason="US-201.1: implement pipeline.frames.extract_keyframes")
def test_frames_have_timestamp():
    ...  # TC-201: mỗi video ≥1 frame, timestamp tăng dần & trong thời lượng


@pytest.mark.skip(reason="US-208.1: implement pipeline.asr.extract_transcript")
def test_transcript_segments():
    ...  # TC-208: video có tiếng → ≥1 đoạn transcript + timestamp


@pytest.mark.skip(reason="US-301.1: implement retrieval.retrieve_media")
def test_recall_at_k():
    ...  # TC-301: kết quả đúng chủ đề nằm trong top-k


@pytest.mark.skip(reason="US-306.1: implement retrieval.rank + ngưỡng")
def test_ranking_and_threshold():
    ...  # TC-306: top-1 liên quan hơn hạng thấp; dưới ngưỡng → "không tìm thấy"


@pytest.mark.skip(reason="US-407.1: implement answer.answer_regulation")
def test_regulation_answer_has_citation_and_disclaimer():
    ...  # TC-407: neo Điều/Khoản + disclaimer; không quy định → báo rõ


@pytest.mark.skip(reason="US-602.1: implement eval.recall_at_k / mrr")
def test_recall_and_mrr_match_by_hand():
    ...  # TC-602: số khớp tính tay trên tập nhỏ
