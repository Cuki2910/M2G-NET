"""Synthetic-only gate runner checks; no training or real prediction artifacts.

Run: .venv/bin/python -m unittest tests.test_v3_gate_dev -v
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_v3_gate_dev as runner


class GateDevTests(unittest.TestCase):
    def arrays(self):
        labels = np.tile([[0], [1], [0], [1]], (1, len(runner.cfg.TASK_NAMES)))
        return {"row_ids": np.arange(4), "targets": labels,
                "probabilities": labels * 0.6 + 0.2, "masks": np.ones_like(labels)}

    def summaries(self, gain=0.006):
        return [dict(mechanism=mechanism, opponent=opponent, metric=metric,
                     **runner.delta_summary([gain] * 10))
                for mechanism in runner.cfg.V3_DEV_MECHANISMS
                for opponent in runner.cfg.V3_DEV_OPPONENTS
                for metric in ("auc_gain", "brier_gain")] + [
                    dict(mechanism="aggregate", opponent=opponent, metric="brier_gain",
                         **runner.delta_summary([gain] * 10))
                    for opponent in runner.cfg.V3_DEV_OPPONENTS]

    def test_cache_tamper_and_identity(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT / "tests") as temporary:
            directory = Path(temporary) / "member-0"
            identity = {"synthetic": True, "member": 0}
            runner.write_cache(directory, identity, self.arrays())
            runner.align_validation(runner.read_cache(directory, identity), self.arrays())
            with self.assertRaisesRegex(ValueError, "identity"):
                runner.read_cache(directory, dict(identity, member=1))
            with (directory / "validation_predictions.npz").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "changed"):
                runner.read_cache(directory, identity)

    def test_row_labels_masks_and_coverage_are_exact(self):
        original = self.arrays()
        for key in ("row_ids", "targets", "masks"):
            changed = deepcopy(original)
            changed[key] = changed[key][::-1].copy() if key != "masks" else changed[key].copy()
            if key == "masks":
                changed[key][0, 0] = 0
            with self.assertRaisesRegex(ValueError, "alignment"):
                runner.align_validation(changed, original)
        for key in ("targets", "masks", "probabilities"):
            changed = deepcopy(original)
            changed[key] = changed[key].astype(float)
            changed[key][0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "finite"):
                runner.validate_arrays(changed)
        missing = deepcopy(original)
        missing["masks"][:, 0] = 0
        with self.assertRaisesRegex(ValueError, "coverage"):
            runner.validate_arrays(missing)
        duplicate = deepcopy(original)
        duplicate["row_ids"][0] = duplicate["row_ids"][1]
        with self.assertRaisesRegex(ValueError, "unique"):
            runner.validate_arrays(duplicate)

    def test_frozen_validation_reads_only_validation_contract(self):
        arrays = self.arrays()
        with tempfile.TemporaryDirectory(dir=runner.ROOT / "tests") as temporary:
            root = Path(temporary)
            directory = root / "units/additive-g1701-t42-catboost/random_seed_42/catboost"
            directory.mkdir(parents=True)
            metadata = {"seed": 42, "model": "catboost", "fold": "random",
                        "dataset_sha256": "synthetic", "split_rows": {"validation": arrays["row_ids"].tolist()}}
            np.savez_compressed(directory / "validation_predictions.npz",
                                **{key: value for key, value in arrays.items() if key != "row_ids"})
            evaluation = directory / "evaluation.json"
            evaluation.write_text(json.dumps(metadata))
            completion = directory / "complete.json"

            def complete():
                completion.write_text(json.dumps(dict(model="catboost", seed=42, fold="random",
                    artifact_sha256={path.name: runner.sha256(path) for path in
                                     (evaluation, directory / "validation_predictions.npz")})))

            complete()
            with patch.object(runner, "data_root", return_value=root):
                scores, hashes = runner.frozen_validation("additive", 1701, 42, "catboost", arrays, "synthetic")
                self.assertAlmostEqual(scores["auc"], 1)
                self.assertAlmostEqual(scores["brier"], 0.04)
                self.assertEqual(len(hashes), 3)
                metadata["split_rows"]["validation"].reverse()
                evaluation.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    runner.frozen_validation("additive", 1701, 42, "catboost", arrays, "synthetic")
                complete()
                with self.assertRaisesRegex(ValueError, "row_ids"):
                    runner.frozen_validation("additive", 1701, 42, "catboost", arrays, "synthetic")

    def test_sample_mcse_and_threshold_boundaries(self):
        summary = runner.delta_summary([1, 2, 3])
        self.assertAlmostEqual(summary["mcse"], 1 / np.sqrt(3))
        self.assertEqual((summary["mean"], summary["min"], summary["max"], summary["positive"]), (2, 1, 3, 3))
        self.assertIsNone(runner.delta_summary([1])["mcse"])
        good = self.summaries()
        self.assertTrue(all(row["passed"] for row in runner.screening(good)))
        boundary = deepcopy(good)
        target = next(row for row in boundary if row["mechanism"] == "interaction" and row["metric"] == "auc_gain")
        target["mean"] = runner.cfg.V3_DEV_AUC_MIN_DELTA
        self.assertFalse(all(row["passed"] for row in runner.screening(boundary)))
        target["mean"], target["positive"] = 0.006, 8
        self.assertFalse(all(row["passed"] for row in runner.screening(boundary)))
        target["positive"] = 9
        self.assertTrue(all(row["passed"] for row in runner.screening(boundary)))
        brier = next(row for row in boundary if row["mechanism"] == "aggregate" and
                     row["opponent"] == "catboost" and row["metric"] == "brier_gain")
        brier["mean"] = -0.0001
        self.assertFalse(all(row["passed"] for row in runner.screening(boundary)))
        with self.assertRaises(KeyError):
            runner.screening(good[:-2])

    def test_score_averages_training_seeds_and_subset_cannot_pass(self):
        args = runner.parse_args(["--variants", "legacy", "--generator-seeds", "1701", "1702",
                                  "--members", "1"])
        arrays = self.arrays()
        with tempfile.TemporaryDirectory(dir=runner.ROOT / "tests") as temporary:
            root = Path(temporary)

            def opponent(mechanism, generator, training, name, expected, data_hash):
                return {"auc": 0.8 if training == 42 else 0.9, "brier": 0.1}, {}

            with (patch.object(runner.cfg, "V3_DEV_REPORT_ROOT", str(root)),
                  patch.object(runner, "identity_for", return_value={"sources": {}}),
                  patch.object(runner, "validation_frames", return_value=(None, None)),
                  patch.object(runner, "reference_arrays", return_value=arrays),
                  patch.object(runner, "dataset_path", return_value=root / "synthetic.csv"),
                  patch.object(runner, "unit_path", return_value=root / "cache"),
                  patch.object(runner, "read_cache", return_value=arrays),
                  patch.object(runner, "frozen_validation", side_effect=opponent),
                  patch.object(runner, "sha256", return_value="synthetic"),
                  patch("builtins.print")):
                runner.score([("additive", 1701, 42, "legacy")], args)
            report = next(root.glob("score-*"))
            screening = json.loads((report / "screening.json").read_text())
            self.assertTrue(screening["exploratory_subset"])
            self.assertTrue(screening["criteria"][0]["passes_selected_subset"])
            self.assertFalse(screening["criteria"][0]["passes_full_dev_screen"])
            import pandas as pd
            deltas = pd.read_csv(report / "paired-deltas.csv")
            np.testing.assert_allclose(deltas["auc_gain"], 0.15)
            np.testing.assert_allclose(deltas["brier_gain"], 0.06)
            summary = pd.read_csv(report / "summary.csv")
            self.assertTrue((summary["n"] == 2).all())
            self.assertTrue((summary["mcse"] == 0).all())

    def test_cli_subsets_no_caps_and_no_holdout(self):
        args = runner.parse_args(["--variants", "task_blend", "--generator-seeds", "1701",
                                  "--training-seeds", "101", "--members", "3"])
        self.assertEqual(args.members, 3)
        self.assertEqual(args.phase, "score")
        with patch("sys.stderr"):
            for arguments in (["--generator-seeds", "1711"], ["--members", "0"],
                              ["--members", str(runner.cfg.V3_ENSEMBLE_SIZE + 1)],
                              ["--training-seeds", "42", "42"]):
                with self.assertRaises(SystemExit):
                    runner.parse_args(arguments)
        with self.assertRaisesRegex(ValueError, "DEVELOPMENT"):
            runner.data_root(1711)


if __name__ == "__main__":
    unittest.main()
