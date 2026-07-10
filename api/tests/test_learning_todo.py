"""Skeleton test cho các phần LÕI HỌC còn lại (đang skip).

Sinh viên: implement hàm tương ứng rồi bỏ @pytest.mark.skip, viết assert theo
Acceptance Criteria trong docs/prd.md. Mỗi test map tới 1 TC-xxx.

ĐÃ implement (xem file test tương ứng, không còn skip ở đây):
  US-201.1 extract_keyframes (T-10) ....... tests/test_pipeline.py
  US-202.1/205.1 embed/build_index (T-13/14) tests/test_pipeline.py
  US-208.1 extract_transcript (T-11) ...... tests/test_pipeline.py
  US-301.1 retrieve_media (T-30) .......... tests/test_retrieval.py
  US-306.1 rank + ngưỡng (T-33) ........... tests/test_retrieval.py
  US-307.1 retrieve_regulations (T-31) .... tests/test_retrieval.py
  US-308.1 retrieve_by_transcript (T-32) .. tests/test_retrieval.py
  US-203.1 generate_caption (T-12) ........ tests/test_pipeline.py
  US-401.1 synthesize_answer (T-41) ....... tests/test_answer.py
  US-407.1 answer_regulation (T-42) ....... tests/test_answer.py
"""
from __future__ import annotations

import pytest


@pytest.mark.skip(reason="US-602.1: implement eval.recall_at_k / mrr (T-71)")
def test_recall_and_mrr_match_by_hand():
    ...  # TC-602: số khớp tính tay trên tập nhỏ
