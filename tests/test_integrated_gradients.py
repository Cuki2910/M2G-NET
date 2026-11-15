import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import config as cfg
from scripts.explain.ig_explain import (
    TaskPredictionWrapper, integrated_view_attributions, compute_ig_per_view, main,
)
from src.checkpoint import make_checkpoint_payload, dataset_sha256
from src.data_pipeline import load_data, RiderDataset
from src.loss import UncertaintyWeightedLoss
from src.model import TGMVMTGFNetV2


class SiteIdentity(torch.nn.Module):
    def forward(self, features, site_ids, use_intercept=True):
        return features


class ConcatenateFusion(torch.nn.Module):
    def forward(self, representations, interaction):
        fused = torch.cat(representations, dim=-1)
        return [fused] * cfg.NUM_TASKS, [], None


class PolynomialHead(torch.nn.Module):
    def __init__(self, power=3):
        super().__init__()
        self.power = power

    def forward(self, features):
        return features[:, 0].pow(self.power) + features[:, 1:].sum(dim=1)


class PolynomialModel(torch.nn.Module):
    def __init__(self, power=3):
        super().__init__()
        self.ablation = "no_interaction"
        self.encoders = torch.nn.ModuleDict({name: torch.nn.Identity() for name in cfg.ALL_VIEW_NAMES})
        self.encoders["site"] = SiteIdentity()
        self.fusion = ConcatenateFusion()
        self.heads = torch.nn.ModuleList([PolynomialHead(power) for _ in cfg.TASK_NAMES])


def polynomial_views(count=1):
    return {"rider_role": torch.full((count, 1), 2.0), "rider_traits": torch.full((count, 1), 8.0),
            "road_context": torch.zeros(count, 1), "environment": torch.zeros(count, 1),
            "site_obs": torch.zeros(count, 1), "site_id": torch.zeros(count, dtype=torch.long)}


class IntegratedGradientsTest(unittest.TestCase):
    def test_nonlinear_attributions_match_analytic_integral(self):
        attribution, delta, difference = integrated_view_attributions(PolynomialModel(), polynomial_views(), 0, 16)
        np.testing.assert_allclose(attribution.numpy().reshape(-1), [8, 8, 0, 0, 0], atol=1e-5)
        np.testing.assert_allclose(delta.numpy(), 0, atol=1e-5)
        np.testing.assert_allclose(attribution.sum(dim=(1, 2)).numpy(), difference.numpy(), atol=1e-5)
        loader = [(polynomial_views(), torch.zeros(1, 4), torch.ones(1, 4))]
        result = compute_ig_per_view(PolynomialModel(), loader, 0, n_steps=16)
        np.testing.assert_allclose(result["normalized_importance"], [0.5, 0.5, 0, 0, 0], atol=1e-6)

    def test_quadrature_steps_control_convergence(self):
        model = PolynomialModel(power=9)
        _, coarse, _ = integrated_view_attributions(model, polynomial_views(), 0, 2)
        _, fine, _ = integrated_view_attributions(model, polynomial_views(), 0, 32)
        self.assertGreater(float(coarse.abs().max()), 1)
        self.assertLess(float(fine.abs().max()), 1e-3)
        loader = [(polynomial_views(), torch.zeros(1, 4), torch.ones(1, 4))]
        with self.assertRaisesRegex(RuntimeError, "completeness tolerance failed"):
            compute_ig_per_view(model, loader, 0, n_steps=2)

    def test_only_observed_examples_count_toward_sample_limit(self):
        views = polynomial_views(5)
        masks = torch.ones(5, 4)
        masks[[0, 2], 0] = 0
        targets = torch.zeros(5, 4)
        for batch_size in [2, 3]:
            loader = [({name: values[start:start + batch_size] for name, values in views.items()},
                       targets[start:start + batch_size], masks[start:start + batch_size])
                      for start in range(0, 5, batch_size)]
            result = compute_ig_per_view(PolynomialModel(), loader, 0, n_steps=16, n_samples=2)
            np.testing.assert_array_equal(result["sample_indices"], [1, 3])
            np.testing.assert_allclose(result["normalized_importance"], [0.5, 0.5, 0, 0, 0], atol=1e-6)

    def test_zero_attribution_normalization_is_finite(self):
        views = {name: torch.zeros_like(value) for name, value in polynomial_views().items()}
        result = compute_ig_per_view(PolynomialModel(), [(views, torch.zeros(1, 4), torch.ones(1, 4))], 0, n_steps=4)
        np.testing.assert_array_equal(result["normalized_importance"], np.zeros(5))

    def test_population_importance_uses_absolute_signed_view_sum(self):
        attribution = torch.zeros(1, 5, 2)
        attribution[0, 0] = torch.tensor([1.0, -1.0])
        attribution[0, 1] = torch.tensor([2.0, 0.0])
        loader = [(polynomial_views(), torch.zeros(1, 4), torch.ones(1, 4))]
        with patch("scripts.explain.ig_explain.integrated_view_attributions",
                   return_value=(attribution, torch.zeros(1), torch.tensor([2.0]))):
            result = compute_ig_per_view(PolynomialModel(), loader, 0)
        np.testing.assert_array_equal(result["normalized_importance"], [0, 1, 0, 0, 0])

    def test_invalid_inputs_and_unobserved_tasks_rejected(self):
        loader = [(polynomial_views(), torch.zeros(1, 4), torch.ones(1, 4))]
        for arguments in [{"n_samples": 0}, {"atol": float("nan")}, {"rtol": -1}, {"task_idx": -1}]:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                compute_ig_per_view(PolynomialModel(), loader, **{"task_idx": 0, **arguments})
        with self.assertRaisesRegex(ValueError, "No observed"):
            compute_ig_per_view(PolynomialModel(), [(polynomial_views(), torch.zeros(1, 4), torch.zeros(1, 4))], 0)
        for steps in [0, 1, 2.5, True]:
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                integrated_view_attributions(PolynomialModel(), polynomial_views(), 0, steps)
        views = polynomial_views()
        views["rider_role"].fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "finite"):
            integrated_view_attributions(PolynomialModel(), views, 0)

    def test_failure_restores_mixed_training_states(self):
        model = PolynomialModel().train()
        model.encoders["rider_role"].eval()
        before = {module: module.training for module in model.modules()}
        with patch("scripts.explain.ig_explain.IntegratedGradients.attribute", side_effect=RuntimeError("test")):
            with self.assertRaisesRegex(RuntimeError, "test"):
                integrated_view_attributions(model, polynomial_views(), 0)
        self.assertEqual(before, {module: module.training for module in model.modules()})


class RealArchitectureAttributionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, _, test, _, cls.vocab = load_data(label_policy="mask")
        cls.views = RiderDataset(test.head(3), cls.vocab).views

    def test_latent_forward_matches_all_architecture_variants(self):
        for ablation in ["full", "no_interaction", "no_site", "shared_gate", "early_fusion"]:
            for intercept in [False, True]:
                with self.subTest(ablation=ablation, intercept=intercept):
                    model = TGMVMTGFNetV2(self.vocab, ablation=ablation).eval()
                    representations = tuple(model.encoders[name](self.views["site_obs"], self.views["site_id"], use_intercept=intercept)
                                            if name == "site" else model.encoders[name](self.views[name])
                                            for name in cfg.ALL_VIEW_NAMES)
                    expected, _, _ = model(self.views, use_site_intercept=intercept)
                    for task_idx in range(cfg.NUM_TASKS):
                        actual = TaskPredictionWrapper(model, task_idx)(*representations)
                        torch.testing.assert_close(actual, expected[task_idx], rtol=0, atol=0)
                    if ablation == "no_site":
                        attribution, _, _ = integrated_view_attributions(model, self.views, 0, n_steps=4, use_site_intercept=intercept)
                        torch.testing.assert_close(attribution[:, -1], torch.zeros_like(attribution[:, -1]))

    def test_site_out_is_invariant_to_intercept_weights_and_preserves_gradients(self):
        model = TGMVMTGFNetV2(self.vocab).train()
        parameter = next(model.parameters())
        parameter.grad = torch.ones_like(parameter)
        before = parameter.grad.clone()
        first, _, _ = integrated_view_attributions(model, self.views, 0, n_steps=4, use_site_intercept=False)
        with torch.no_grad():
            model.encoders["site"].site_intercept.weight.fill_(100)
        second, _, _ = integrated_view_attributions(model, self.views, 0, n_steps=4, use_site_intercept=False)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(parameter.grad, before, rtol=0, atol=0)
        self.assertTrue(model.training)


class AttributionArtifactsTest(unittest.TestCase):
    def test_cli_artifacts_are_auditable_and_never_overwritten(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.csv"
            pd.read_csv(cfg.DATA_PATH).head(500).to_csv(data, index=False)
            train, validation, test, encoders, vocab = load_data(data_path=data, label_policy="mask")
            torch.manual_seed(0)
            model = TGMVMTGFNetV2(vocab)
            payload = make_checkpoint_payload(
                0, model, UncertaintyWeightedLoss(), 0.5, vocab, encoders=encoders,
                data_path=data, label_policy="mask",
                split_rows={name: frame["_row_id"].tolist() for name, frame in
                            zip(["train", "validation", "test"], [train, validation, test])})
            checkpoint = root / "model.pt"
            torch.save(payload, checkpoint)
            output = root / "ig"
            arguments = ["--checkpoint", str(checkpoint), "--output", str(output), "--samples", "2", "--n-steps", "64"]
            main(arguments)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["prediction_scale"], "raw_model_probability")
            self.assertEqual(manifest["baseline"], "zero_encoder_outputs")
            self.assertFalse(manifest["feature_level_attribution"])
            self.assertEqual(manifest["checkpoint_sha256"], dataset_sha256(checkpoint))
            for filename, digest in manifest["artifact_sha256"].items():
                self.assertEqual(digest, dataset_sha256(output / filename))
            for source, digest in manifest["source_sha256"].items():
                self.assertEqual(digest, dataset_sha256(output / "source" / source))
            for task in cfg.TASK_NAMES:
                artifact = np.load(output / f"{task}.npz")
                expected_rows = test.loc[test[cfg.TASK_MASK_COLS[task]].eq(1), "_row_id"].head(2)
                np.testing.assert_array_equal(artifact["row_ids"], expected_rows)
                np.testing.assert_allclose(artifact["signed_attributions"].sum(axis=(1, 2)) - artifact["output_difference"],
                                           artifact["completeness_delta"], atol=1e-6)
            before = (output / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                main(arguments)
            self.assertEqual(before, (output / "manifest.json").read_bytes())
            failed_output = root / "failed"
            arguments[arguments.index(str(output))] = str(failed_output)
            with patch("scripts.explain.ig_explain.compute_ig_per_view", side_effect=RuntimeError("nonconverged")):
                with self.assertRaisesRegex(RuntimeError, "nonconverged"):
                    main(arguments)
            failed = json.loads((failed_output / "manifest.json").read_text())
            self.assertEqual(failed["status"], "failed")
            self.assertNotIn("artifact_sha256", failed)


if __name__ == "__main__":
    unittest.main()
