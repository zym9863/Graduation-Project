import pytest

from src.features.news_value_annotator import parse_news_value_response


def test_parse_news_value_response_with_english_fields() -> None:
    raw = '{"conflict": 5, "importance": 4, "prominence": 3, "proximity": 2, "interest": 1}'

    scores = parse_news_value_response(raw)

    assert scores == [5, 4, 3, 2, 1]


def test_parse_news_value_response_with_chinese_fields() -> None:
    raw = '{"冲突性": 4, "重要性": 4, "显著性": 3, "接近性": 2, "趣味性": 5}'

    scores = parse_news_value_response(raw)

    assert scores == [4, 4, 3, 2, 5]


def test_parse_news_value_response_clamps_to_valid_range() -> None:
    raw = "[0, 2, 9, 5, 7]"

    scores = parse_news_value_response(raw)

    assert scores == [1, 2, 5, 5, 5]


def test_parse_news_value_response_rejects_missing_fields() -> None:
    raw = '{"conflict": 5, "importance": 4, "prominence": 3}'

    with pytest.raises(ValueError):
        parse_news_value_response(raw)


from src.features.news_value_annotator import heuristic_news_value_scores


def test_heuristic_conflict_default_is_two() -> None:
    article = {"title": "Company releases quarterly earnings", "abstract": "Steady growth reported.", "category": "finance"}

    scores = heuristic_news_value_scores(article)

    assert scores[0] == 2  # conflict: no conflict keywords → default 2


def test_heuristic_conflict_keyword_scores_four() -> None:
    article = {"title": "War erupts in border region", "abstract": "Military conflict escalates.", "category": "news"}

    scores = heuristic_news_value_scores(article)

    assert scores[0] == 4  # conflict: "war" and "conflict" present → 4


def test_heuristic_proximity_livelihood_keyword_scores_four() -> None:
    article = {"title": "Housing prices surge nationwide", "abstract": "Rent increases affect millions.", "category": "finance"}

    scores = heuristic_news_value_scores(article)

    assert scores[3] == 4  # proximity: "housing" and "rent" present → 4


def test_heuristic_proximity_default_is_three() -> None:
    article = {"title": "New art exhibition opens", "abstract": "Local gallery showcases paintings.", "category": "lifestyle"}

    scores = heuristic_news_value_scores(article)

    assert scores[3] == 3  # proximity: no livelihood keywords → default 3
