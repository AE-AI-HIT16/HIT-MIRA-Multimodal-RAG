"""Test Router RAG — TC-507. Phần luật CHẠY XANH; classifier là TODO."""
from __future__ import annotations

import pytest

from app.routing.intent import Source, route


def test_image_routes_to_media():
    assert route(None, has_image=True) is Source.MEDIA


def test_keyword_routes_to_regulation():
    assert route("Điều 5 quy định gì về sinh hoạt", has_image=False) is Source.REGULATION


def test_plain_query_routes_to_media():
    assert route("ảnh sự kiện tân sinh viên 2023", has_image=False) is Source.MEDIA


def test_override_wins():
    assert route("bất kỳ", has_image=True, override=Source.BOTH) is Source.BOTH


@pytest.mark.skip(reason="BR-507: sinh viên implement classifier fallback khi luật không rõ")
def test_ambiguous_uses_classifier():
    ...
