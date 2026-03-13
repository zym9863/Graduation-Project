from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_project_env(project_root: Path | None = None) -> None:
    root = project_root if project_root is not None else Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env", override=False)