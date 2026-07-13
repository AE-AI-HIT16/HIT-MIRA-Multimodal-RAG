"""Đánh giá chất lượng truy xuất (LÕI HỌC). [BR-602/606]

Hợp đồng — sinh viên implement:
  recall_at_k : trung bình Recall@k trên tập eval_queries
  mrr         : Mean Reciprocal Rank
  run_eval    : chạy toàn tập → báo cáo {recall@k, mrr, latency}; so mục tiêu 0.80/0.60
  Pass : tests/test_eval.py::test_recall_and_mrr_match_by_hand
"""
from __future__ import annotations

from typing import Any


def recall_at_k(retrieved_ids: list[Any], expected_ids: list[Any], k: int) -> float:
    raise NotImplementedError("US-602.1: tính Recall@k")


def mrr(rankings: list[list[Any]], expected: list[list[Any]]) -> float:
    raise NotImplementedError("US-602.1: tính MRR")


def run_eval(k_values: tuple[int, ...] = (1, 5, 20, 50, 100)) -> dict[str, Any]:
    raise NotImplementedError("US-602.1: chạy đánh giá toàn tập")
