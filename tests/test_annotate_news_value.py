import pytest

from scripts.annotate_news_value import main, parse_args


def test_parse_args_single_mode_enabled() -> None:
    args = parse_args([
        "--single-title",
        "政策发布",
        "--single-abstract",
        "影响多个行业",
    ])

    assert args.single_mode is True


def test_parse_args_single_mode_conflict_with_limit() -> None:
    with pytest.raises(SystemExit):
        parse_args([
            "--single-title",
            "政策发布",
            "--limit",
            "1",
        ])


def test_single_case_prints_named_vector(capsys: pytest.CaptureFixture[str]) -> None:
    main([
        "--provider",
        "heuristic",
        "--single-title",
        "Breaking: New regulation released",
        "--single-abstract",
        "Authorities announced a major update today.",
        "--single-category",
        "news",
    ])

    output = capsys.readouterr().out
    assert "Single News Value Case" in output
    assert '"timeliness"' in output
    assert '"importance"' in output
    assert '"prominence"' in output
    assert '"proximity"' in output
    assert '"interest"' in output
    assert "vector:" in output
