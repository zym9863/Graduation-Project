import pytest

from src.features.news_value_annotator import parse_news_value_response


def test_parse_news_value_response_with_english_fields() -> None:
    raw = '{"timeliness": 5, "importance": 4, "prominence": 3, "proximity": 2, "interest": 1}'

    scores = parse_news_value_response(raw)

    assert scores == [5, 4, 3, 2, 1]


def test_parse_news_value_response_with_chinese_fields() -> None:
    raw = '{"时新性": 4, "重要性": 4, "显著性": 3, "接近性": 2, "趣味性": 5}'

    scores = parse_news_value_response(raw)

    assert scores == [4, 4, 3, 2, 5]


def test_parse_news_value_response_clamps_to_valid_range() -> None:
    raw = "[0, 2, 9, 5, 7]"

    scores = parse_news_value_response(raw)

    assert scores == [1, 2, 5, 5, 5]


def test_parse_news_value_response_rejects_missing_fields() -> None:
    raw = '{"timeliness": 5, "importance": 4, "prominence": 3}'

    with pytest.raises(ValueError):
        parse_news_value_response(raw)
