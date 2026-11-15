"""Synthetic-only math and checkpoint regression checks for optional v3 gates."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import config as cfg
from src.checkpoint import load_model_bundle, make_checkpoint_payload
from src.fusion import RegularizedTaskGate, ResidualGatedFusion
from src.loss import UncertaintyWeightedLoss
from src.model import TGMVMTGFNetV2


def vocab():
    columns = [c for cols in cfg.INDIVIDUAL_VIEW_COLS.values() for c in cols]
    columns += [c for cols in cfg.CONTEXTUAL_VIEW_COLS.values() for c in cols]
    return {**dict.fromkeys(columns, 3), "num_sites": 4}


def views():
    result = {name: torch.zeros(3, len(cols), dtype=torch.long)
              for name, cols in {**cfg.INDIVIDUAL_VIEW_COLS, **cfg.CONTEXTUAL_VIEW_COLS}.items()}
    result["site_id"] = torch.tensor([0, 1, 3])
    return result


class GateVariantTest(unittest.TestCase):
    def test_concrete_prior_and_nonlinear_simplex(self):
        for nonlinear in (False, True):
            gate = RegularizedTaskGate(3, 2, prior_weight=0.3, nonlinear=nonlinear)
            with torch.no_grad():
                for parameter in gate.gate.parameters():
                    parameter.zero_()
                output = gate.gate[-1] if nonlinear else gate.gate
                output.bias.copy_(torch.tensor([2., 1., 0.]))
                gate.temperature.fill_(1.)
            weights = gate([torch.zeros(2, 2) for _ in range(3)])
            torch.testing.assert_close(weights, torch.tensor([[.8, .1, .1]]).expand(2, -1))
            gate.prior_weight = 1.
            torch.testing.assert_close(gate([torch.zeros(2, 2) for _ in range(3)]), torch.full((2, 3), 1/3))

    def test_task_blend_indexes_task_not_feature_axis(self):
        # d == num_tasks catches accidental [K] broadcasting across feature coordinates.
        fusion = ResidualGatedFusion(2, 2, 2, 2, view_dim=2, dropout=0.,
                                     ablation="no_interaction", gate_variant="task_blend")
        fusion.early_mlp = torch.nn.Identity()
        with torch.no_grad():
            fusion.alpha_logit.copy_(torch.logit(torch.tensor([.25, .75])))
        x = torch.tensor([[2., 4.]])
        result, _, alpha = fusion([x], None)
        torch.testing.assert_close(alpha, torch.tensor([.25, .75]))
        for z in result:
            torch.testing.assert_close(z, x)
        # Distinct early path yields task-specific convex combinations.
        fusion.early_mlp = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            fusion.early_mlp.weight.zero_()
        result, _, _ = fusion([x], None)
        torch.testing.assert_close(result[0], .25*x)
        torch.testing.assert_close(result[1], .75*x)

    def test_variants_finite_shapes_gradients_and_invalid_mode(self):
        for variant in cfg.V3_GATE_VARIANTS:
            model = TGMVMTGFNetV2(vocab(), gate_variant=variant)
            predictions, gates, alpha = model(views())
            self.assertEqual(alpha.shape, torch.Size([cfg.NUM_TASKS]) if "task_blend" in variant else torch.Size([]))
            for weights in gates:
                self.assertTrue(torch.isfinite(weights).all())
                torch.testing.assert_close(weights.sum(-1), torch.ones(3))
                self.assertTrue((weights >= cfg.GATE_PRIOR_WEIGHT/cfg.NUM_GATE_INPUTS-1e-7).all())
            sum(p.sum() for p in predictions).backward()
            self.assertIsNotNone(model.fusion.alpha_logit.grad)
            self.assertTrue(torch.isfinite(model.fusion.alpha_logit.grad).all())
        with self.assertRaises(ValueError):
            TGMVMTGFNetV2(vocab(), gate_variant="invalid")
        with self.assertRaises(ValueError):
            TGMVMTGFNetV2(vocab(), ablation="early_fusion", gate_variant="task_blend")

    def test_checkpoint_variant_and_skip_roundtrip_without_data_access(self):
        for variant in cfg.V3_GATE_VARIANTS:
            model = TGMVMTGFNetV2(vocab(), gate_variant=variant, interaction_skip=True,
                                   gate_prior_weight=.37).eval()
            with torch.no_grad():
                model.fusion.alpha_logit.fill_(.7)
                for gate in model.fusion.gates:
                    gate.temperature.fill_(1.27)
            payload = make_checkpoint_payload(1, model, UncertaintyWeightedLoss(), .7, vocab(),
                                              data_path="synthetic-fixture-no-data.csv")
            Path(cfg.V3_DEV_ROOT).mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=cfg.V3_DEV_ROOT) as directory:
                path = Path(directory)/"model.pt"
                torch.save(payload, path)
                frames = (None,)*7 + (vocab(),)
                with patch("src.checkpoint.splits_from_checkpoint", return_value=frames):
                    loaded = load_model_bundle(path)["model"]
                self.assertEqual(loaded.gate_variant, variant)
                self.assertTrue(all(gate.prior_weight == .37 for gate in loaded.fusion.gates))
                for expected, actual in zip(model(views())[0], loaded(views())[0]):
                    torch.testing.assert_close(expected, actual, rtol=0, atol=0)


if __name__ == "__main__":
    Path(cfg.V3_DEV_ROOT).mkdir(parents=True, exist_ok=True)
    unittest.main()
