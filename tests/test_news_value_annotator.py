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


def test_annotate_batch_calls_aliyun_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    output_jsonl = "\n".join(
        [
            '{"custom_id":"N1","response":{"status_code":200,"body":{"choices":[{"message":{"content":"{\\"conflict\\": 5, \\"importance\\": 4, \\"prominence\\": 3, \\"proximity\\": 2, \\"interest\\": 1}"}}]}}}',
            '{"custom_id":"N2","response":{"status_code":200,"body":{"choices":[{"message":{"content":"{\\"conflict\\": 6, \\"importance\\": 0, \\"prominence\\": 3, \\"proximity\\": 2, \\"interest\\": 1}"}}]}}}',
        ]
    )
    error_jsonl = '{"custom_id":"N3","error":{"code":"BadRequest","message":"Invalid content"}}\n'

    class FakeFiles:
        uploaded_text: str | None = None

        def create(self, **kwargs: object):
            file_obj = kwargs["file"]
            FakeFiles.uploaded_text = file_obj.read().decode("utf-8")
            return type("UploadedFile", (), {"id": "file-input-1"})()

        def content(self, file_id: str):
            if file_id == "file-output-1":
                return type("Content", (), {"text": output_jsonl})()
            if file_id == "file-error-1":
                return type("Content", (), {"text": error_jsonl})()
            raise AssertionError(f"Unexpected file_id: {file_id}")

    class FakeBatches:
        create_kwargs: dict[str, object] | None = None
        retrieve_calls: int = 0

        def create(self, **kwargs: object):
            FakeBatches.create_kwargs = kwargs
            return type("Batch", (), {"id": "batch-123", "status": "validating"})()

        def retrieve(self, _batch_id: str):
            FakeBatches.retrieve_calls += 1
            if FakeBatches.retrieve_calls == 1:
                return type("Batch", (), {"id": "batch-123", "status": "in_progress"})()
            return type(
                "Batch",
                (),
                {
                    "id": "batch-123",
                    "status": "completed",
                    "output_file_id": "file-output-1",
                    "error_file_id": "file-error-1",
                },
            )()

    class FakeOpenAI:
        last_init_kwargs: dict[str, object] | None = None

        def __init__(self, **kwargs: object) -> None:
            FakeOpenAI.last_init_kwargs = kwargs
            self.files = FakeFiles()
            self.batches = FakeBatches()

    monkeypatch.setattr(annotator_module, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(annotator_module.time, "sleep", lambda _seconds: None)

    annotator = NewsValueAnnotator(
        model="qwen-plus",
        provider="aliyun-batch",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="token",
        poll_interval=1,
    )

    result = annotator.annotate_batch(
        [
            ("N1", {"title": "t1", "abstract": "a1", "category": "c1", "subcategory": "s1"}),
            ("N2", {"title": "t2", "abstract": "a2", "category": "c2", "subcategory": "s2"}),
            ("N3", {"title": "t3", "abstract": "a3", "category": "c3", "subcategory": "s3"}),
        ]
    )

    assert FakeOpenAI.last_init_kwargs == {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "token",
        "timeout": 60.0,
    }
    assert FakeBatches.create_kwargs is not None
    assert FakeBatches.create_kwargs["input_file_id"] == "file-input-1"
    assert FakeBatches.create_kwargs["endpoint"] == "/v1/chat/completions"
    assert result.batch_job_id == "batch-123"
    assert result.status == "completed"
    assert result.scores["N1"] == [5, 4, 3, 2, 1]
    assert result.scores["N2"] == [5, 1, 3, 2, 1]
    assert "N3" in result.failures
    assert "Invalid content" in result.failures["N3"]
    assert FakeFiles.uploaded_text is not None
    assert '"custom_id": "N1"' in FakeFiles.uploaded_text

