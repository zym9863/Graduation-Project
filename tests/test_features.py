from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gp_newsrec.features import _feature_tensor, _resolve_image_path


class FakeTensor:
    def norm(self) -> None:
        return None


class FeatureExtractionTest(unittest.TestCase):
    def test_raw_tensor_output_is_returned(self) -> None:
        tensor = FakeTensor()

        self.assertIs(_feature_tensor(tensor, "text_embeds"), tensor)

    def test_named_embedding_output_is_returned(self) -> None:
        tensor = FakeTensor()
        output = SimpleNamespace(text_embeds=tensor)

        self.assertIs(_feature_tensor(output, "text_embeds"), tensor)

    def test_pooled_model_output_is_returned(self) -> None:
        tensor = FakeTensor()
        output = SimpleNamespace(pooler_output=tensor)

        self.assertIs(_feature_tensor(output, "text_embeds"), tensor)

    def test_windows_style_cached_image_path_is_resolved_on_linux(self) -> None:
        record = {"news_id": "test_features", "image_path": r"tests\test_features.py"}

        self.assertEqual(_resolve_image_path(record, Path("unused")).resolve(), Path("tests/test_features.py").resolve())

    def test_windows_absolute_cached_image_path_falls_back_to_image_dir(self) -> None:
        record = {"news_id": "missing", "image_path": r"C:\cached\test_features.py"}

        self.assertEqual(_resolve_image_path(record, Path("tests")).resolve(), Path("tests/test_features.py").resolve())


if __name__ == "__main__":
    unittest.main()
