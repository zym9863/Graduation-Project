import argparse

from scripts.annotate_news_value import main as annotate_news_value_main
from scripts.evaluate import main as evaluate_main
from scripts.extract_features import main as extract_features_main
from scripts.preprocess import main as preprocess_main
from scripts.train import main as train_main


COMMANDS = {
    "annotate-news-value": annotate_news_value_main,
    "evaluate": evaluate_main,
    "extract-features": extract_features_main,
    "preprocess": preprocess_main,
    "train": train_main,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="多模态新闻推荐系统命令行入口。",
    )
    parser.add_argument(
        "command",
        choices=sorted(COMMANDS),
        help="要执行的子命令。",
    )
    args, remaining = parser.parse_known_args()
    COMMANDS[args.command](remaining)


if __name__ == "__main__":
    main()
