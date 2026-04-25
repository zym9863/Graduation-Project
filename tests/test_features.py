from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gp_newsrec.features import _feature_tensor


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


if __name__ == "__main__":
    unittest.main()
