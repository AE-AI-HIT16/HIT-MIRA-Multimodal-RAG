"""Cấu hình thiếu `.env` phải rơi về default thay vì giữ placeholder YAML."""

from types import SimpleNamespace

from src.configuration import _config_number


def test_config_number_dung_default_khi_placeholder_chua_duoc_thay() -> None:
    config = SimpleNamespace(RETRIEVAL=SimpleNamespace(TOP_K="${RETRIEVAL_TOP_K}"))

    assert _config_number(config, "RETRIEVAL.TOP_K", 5, int) == 5


def test_config_number_parse_gia_tri_hop_le() -> None:
    config = SimpleNamespace(
        RETRIEVAL=SimpleNamespace(TOP_K="3", KEYWORD_THRESHOLD="0.5")
    )

    assert _config_number(config, "RETRIEVAL.TOP_K", 5, int) == 3
    assert _config_number(config, "RETRIEVAL.KEYWORD_THRESHOLD", 0.1, float) == 0.5
