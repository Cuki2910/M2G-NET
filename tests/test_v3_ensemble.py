"""ProbabilityEnsemble returns the exact mean of member probabilities per task."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate.paper_pipeline import ProbabilityEnsemble


class Stub(torch.nn.Module):
    ablation, fusion = "full", None

    def __init__(self, values):
        super().__init__()
        self.values = values

    def forward(self, views, use_site_intercept=True):
        return [torch.tensor(v) for v in self.values], [], torch.tensor(0.0)


class EnsembleTest(unittest.TestCase):
    def test_mean_of_members(self):
        ensemble = ProbabilityEnsemble([Stub([[0.2, 0.4], [1.0, 0.0]]), Stub([[0.6, 0.0], [0.0, 0.5]])])
        preds, _, _ = ensemble({})
        self.assertTrue(torch.allclose(preds[0], torch.tensor([0.4, 0.2])))
        self.assertTrue(torch.allclose(preds[1], torch.tensor([0.5, 0.25])))


if __name__ == "__main__":
    unittest.main()
