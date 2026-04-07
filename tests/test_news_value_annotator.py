import pytest

import src.features.news_value_annotator as annotator_module
from src.features.news_value_annotator import NewsValueAnnotator, parse_news_value_response


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


def test_annotator_rejects_unsupported_provider() -> None:
    annotator = NewsValueAnnotator(model="fake-model", provider="heuristic")

    with pytest.raises(ValueError, match="Unsupported provider"):
        annotator.annotate({"title": "a", "abstract": "b"})


def test_annotator_requires_api_key() -> None:
    annotator = NewsValueAnnotator(
        model="fake-model",
        provider="openai-compatible",
        base_url="https://example.com/v1",
        api_key=None,
    )

    with pytest.raises(ValueError, match="api_key is required"):
        annotator.annotate({"title": "a", "abstract": "b"})


def test_annotator_requires_base_url() -> None:
    annotator = NewsValueAnnotator(
        model="fake-model",
        provider="openai-compatible",
        base_url=None,
        api_key="token",
    )

    with pytest.raises(ValueError, match="base_url is required"):
        annotator.annotate({"title": "a", "abstract": "b"})


def test_annotator_calls_openai_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.last_create_kwargs: dict[str, object] | None = None

        def create(self, **kwargs: object):
            self.last_create_kwargs = kwargs
            message = type("Message", (), {"content": '{"conflict": 5, "importance": 4, "prominence": 3, "proximity": 2, "interest": 1}'})
            choice = type("Choice", (), {"message": message()})
            return type("Response", (), {"choices": [choice()]})

    class FakeOpenAI:
        last_init_kwargs: dict[str, object] | None = None
        completions = FakeCompletions()

        def __init__(self, **kwargs: object) -> None:
            FakeOpenAI.last_init_kwargs = kwargs
            self.chat = type("Chat", (), {"completions": FakeOpenAI.completions})()

    monkeypatch.setattr(annotator_module, "OpenAI", FakeOpenAI)

    annotator = NewsValueAnnotator(
        model="fake-model",
        provider="openai-compatible",
        base_url="https://example.com/v1",
        api_key="token",
        timeout=12.5,
    )

    scores = annotator.annotate(
        {
            "title": "Sample title",
            "abstract": "Sample abstract",
            "category": "news",
            "subcategory": "policy",
        }
    )

    assert scores == [5, 4, 3, 2, 1]
    assert FakeOpenAI.last_init_kwargs == {
        "base_url": "https://example.com/v1",
        "api_key": "token",
        "timeout": 12.5,
    }
    assert FakeOpenAI.completions.last_create_kwargs is not None
    assert FakeOpenAI.completions.last_create_kwargs["model"] == "fake-model"
    assert FakeOpenAI.completions.last_create_kwargs["temperature"] == 0
    assert FakeOpenAI.completions.last_create_kwargs["response_format"] == {"type": "json_object"}

