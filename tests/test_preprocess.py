from pathlib import Path

from src.data.preprocess import build_category_maps, parse_behaviors_file, parse_news_file


def test_parse_news_returns_dict(tmp_path: Path) -> None:
    tsv = tmp_path / "news.tsv"
    tsv.write_text(
        "N1\tnews\tnewsworld\tTitle One\tAbstract one.\thttps://ex.com\t[]\t[]\n"
        "N2\tsports\tbaseball\tTitle Two\tAbstract two.\thttps://ex.com\t[]\t[]\n",
        encoding="utf-8",
    )

    news = parse_news_file(tsv)

    assert len(news) == 2
    assert news["N1"]["category"] == "news"
    assert news["N1"]["subcategory"] == "newsworld"
    assert news["N1"]["title"] == "Title One"
    assert news["N1"]["abstract"] == "Abstract one."

    cat2id, subcat2id = build_category_maps(news)
    assert cat2id["news"] >= 1
    assert subcat2id["baseball"] >= 1


def test_parse_behaviors_returns_list(tmp_path: Path) -> None:
    tsv = tmp_path / "behaviors.tsv"
    tsv.write_text(
        "1\tU1\t11/11/2019 9:00:00 AM\tN1 N2\tN3-1 N4-0 N5-0\n",
        encoding="utf-8",
    )

    behaviors = parse_behaviors_file(tsv)
    assert len(behaviors) == 1
    sample = behaviors[0]
    assert sample["user_id"] == "U1"
    assert sample["history"] == ["N1", "N2"]
    assert sample["impressions"] == [("N3", 1), ("N4", 0), ("N5", 0)]