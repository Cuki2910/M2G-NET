"""Synthetic-only checks; never train or inspect development/held-out data.

Run: .venv/bin/python -m unittest tests.test_v3_weight_decay_dev -v
"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_v3_weight_decay_dev as wrapper

runner = wrapper.runner


class WeightDecayDevTests(unittest.TestCase):
    def test_fixed_full_grid_and_score_default(self):
        with patch.object(runner, "main", return_value="report") as shared:
            self.assertEqual(wrapper.main([]), "report")
        args = runner.parse_args(shared.call_args.args[0])
        self.assertEqual(args.phase, "score")
        self.assertEqual(args.variants, list(runner.cfg.V3_DECAY_VARIANTS))
        self.assertEqual(args.generator_seeds, list(runner.cfg.V3_DEV_GENERATOR_SEEDS))
        self.assertEqual(args.training_seeds, [42, 101])
        self.assertEqual(args.members, 5)
        fits = (len(runner.cfg.V3_DEV_MECHANISMS) * len(args.generator_seeds)
                * len(args.training_seeds) * len(args.variants) * args.members)
        self.assertEqual(fits, 600)
        self.assertFalse(args.worker)
        self.assertIsNone(args.mechanism)

    def test_fit_only_delegates_to_shared_runner(self):
        with patch.object(runner, "main") as shared:
            wrapper.main(["--phase", "fit"])
        shared.assert_called_once_with([
            "--phase", "fit", "--variants", *runner.cfg.V3_DECAY_VARIANTS])

    def test_rejects_subset_override_or_locked_options_before_delegation(self):
        with patch.object(runner, "main") as shared, patch("sys.stderr"):
            for arguments in (["--variants", "wd_base"], ["--variants", "unknown"],
                              ["--members", "1"], ["--members", "-1"],
                              ["--generator-seeds", "1711"], ["--training-seeds", "42"],
                              ["--worker"], ["--mechanism", "additive"],
                              ["--phase", "test"], ["--ph", "fit"]):
                with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                    wrapper.main(arguments)
        shared.assert_not_called()

    def test_shared_decay_options_and_invalid_values(self):
        for variant, decay in (("wd_base", 1e-4), ("wd_strong", 1e-3)):
            with self.subTest(variant=variant):
                self.assertEqual(runner.training_options(variant),
                                 {"gate_variant": "legacy", "weight_decay": decay})
        for decay in (-1, float("nan"), float("inf"), True, "1e-3", None):
            with self.subTest(decay=decay), patch.dict(runner.cfg.V3_DECAY_VARIANTS, wd_strong=decay):
                with self.assertRaises(ValueError):
                    runner.training_options("wd_strong")
        with self.assertRaises(ValueError):
            runner.training_options("not-a-variant")
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            runner.parse_args(["--variants", "not-a-variant"])

    def test_fit_unit_forwards_m5_options_without_training(self):
        import pandas as pd
        from scripts import train
        from src import data_pipeline

        labels = np.tile([[0], [1]], (1, len(runner.cfg.TASK_NAMES)))
        arrays = dict(row_ids=np.arange(2), targets=labels, masks=np.ones_like(labels),
                      probabilities=labels * 0.6 + 0.2)
        validation = pd.DataFrame({"_row_id": arrays["row_ids"]})
        for variant, decay in runner.cfg.V3_DECAY_VARIANTS.items():
            job = (runner.cfg.V3_DEV_MECHANISMS[0], runner.cfg.V3_DEV_GENERATOR_SEEDS[0], 42, variant)
            with (self.subTest(variant=variant),
                  tempfile.TemporaryDirectory(dir=runner.ROOT / "tests") as temporary,
                  patch.object(runner, "unit_path", return_value=Path(temporary)),
                  patch.object(runner, "identity_for", return_value={"synthetic": True}),
                  patch.object(runner, "validation_frames", return_value=(validation, validation)),
                  patch.object(runner, "reference_arrays", return_value=arrays),
                  patch.object(data_pipeline, "encode_splits", return_value=(validation, validation, None, {}, {})),
                  patch.object(data_pipeline, "get_loaders", return_value=(None, None, None)),
                  patch.object(train, "fit_model_for_split", return_value=(None, None, None, None)) as fit,
                  patch.object(train, "collect_predictions", return_value=(
                      arrays["targets"], arrays["probabilities"], arrays["masks"])),
                  patch.object(runner, "write_cache") as publish,
                  patch("builtins.print")):
                runner.fit_unit(job, 1)
                self.assertEqual(fit.call_args.kwargs["weight_decay"], decay)
                self.assertEqual(fit.call_args.kwargs["gate_variant"], "legacy")
                self.assertFalse(fit.call_args.kwargs["use_focal"])
                self.assertFalse(fit.call_args.kwargs["interaction_skip"])
                publish.assert_called_once()

    def test_optimizer_receives_decay_and_invalid_decay_stops_before_model(self):
        from scripts import train

        for decay in (1e-4, 1e-3):
            with (self.subTest(decay=decay), patch.object(train, "TGMVMTGFNetV2"),
                  patch.object(train, "compute_pos_weights"),
                  patch.object(train, "UncertaintyWeightedLoss"),
                  patch.object(train.optim, "Adam", side_effect=RuntimeError("optimizer sentinel")) as optimizer):
                with self.assertRaisesRegex(RuntimeError, "optimizer sentinel"):
                    train.fit_model_for_split({}, None, None, weight_decay=decay)
                self.assertEqual(optimizer.call_args.kwargs["weight_decay"], decay)
        for decay in (-1, float("nan"), float("inf"), True, "1e-3", None):
            with self.subTest(decay=decay), patch.object(train, "TGMVMTGFNetV2") as model:
                with self.assertRaises(ValueError):
                    train.fit_model_for_split({}, None, None, weight_decay=decay)
                model.assert_not_called()

    def test_identity_binds_decay_and_paths_are_distinct(self):
        prefix = (runner.cfg.V3_DEV_MECHANISMS[0], runner.cfg.V3_DEV_GENERATOR_SEEDS[0],
                  runner.cfg.V3_DEV_TRAINING_SEEDS[0])
        with (patch.object(runner, "sha256", return_value="synthetic"),
              patch.object(runner, "version", return_value="synthetic")):
            base = runner.identity_for((*prefix, "wd_base"))
            strong = runner.identity_for((*prefix, "wd_strong"))
            with patch.dict(runner.cfg.V3_DECAY_VARIANTS, wd_strong=2e-3):
                changed = runner.identity_for((*prefix, "wd_strong"))
        self.assertEqual(base["training_options"]["gate_variant"], "legacy")
        self.assertEqual(base["training_options"]["weight_decay"], 1e-4)
        self.assertEqual(strong["training_options"]["weight_decay"], 1e-3)
        self.assertEqual(changed["training_options"]["weight_decay"], 2e-3)
        self.assertNotEqual(strong, changed)
        self.assertNotEqual(runner.unit_path((*prefix, "wd_base")),
                            runner.unit_path((*prefix, "wd_strong")))
        self.assertEqual(runner.parse_args([]).variants, list(runner.cfg.V3_GATE_VARIANTS))


if __name__ == "__main__":
    unittest.main()
