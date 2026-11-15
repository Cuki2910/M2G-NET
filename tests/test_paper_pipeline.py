import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import config as cfg
from src.data_pipeline import (
    prepare_labels, load_data, encode_splits, extract_targets, extract_task_masks,
)
from src.calibration import TemperatureScaler, fit_validation_rule
from src.checkpoint import make_checkpoint_payload, load_model_bundle, splits_from_checkpoint
from src.loss import UncertaintyWeightedLoss
from src.model import TGMVMTGFNetV2
from src.metrics import tune_task_thresholds
from scripts.evaluate.paper_pipeline import split_frames, cluster_intervals, save_json, main
from baselines.run_all_baselines import fit_tuned_baseline, predict_baseline


def records(count=120):
    frame = pd.read_csv(cfg.DATA_PATH).head(count).copy()
    frame["_row_id"] = np.arange(count)
    return frame


class DataContractTest(unittest.TestCase):
    def test_strict_rejects_contradictory_positive(self):
        frame = records(1)
        frame.loc[0, ["has_signal", "red_light_running"]] = ["No", 1]
        with self.assertRaisesRegex(ValueError, "red-light positives"):
            prepare_labels(frame)

    def test_explicit_mask_policy_preserves_source(self):
        frame = records(2)
        frame.loc[:, "has_signal"] = "No"
        frame.loc[:, "red_light_running"] = [0, 1]
        original = frame.copy(deep=True)
        prepared = prepare_labels(frame, "mask")
        self.assertTrue(prepared["red_light_running_observed"].eq(0).all())
        pd.testing.assert_frame_equal(original, frame)

    def test_missing_target_is_unobserved_not_negative(self):
        frame = records(1).astype({"mobile_phone_use": float})
        frame.loc[0, "mobile_phone_use"] = np.nan
        prepared = prepare_labels(frame, "mask")
        self.assertEqual(prepared.loc[0, "mobile_phone_use_observed"], 0)
        self.assertTrue(torch.isfinite(extract_targets(prepared)).all())

    def test_explicit_observed_missing_target_rejected(self):
        frame = records(1).astype({"mobile_phone_use": float})
        frame.loc[0, "mobile_phone_use"] = np.nan
        frame["mobile_phone_use_observed"] = 1
        with self.assertRaisesRegex(ValueError, "observed labels cannot be missing"):
            prepare_labels(frame, "mask")

    def test_real_data_requires_masks(self):
        with self.assertRaisesRegex(ValueError, "Missing explicit observation mask"):
            prepare_labels(records(), "mask", require_masks=True)

    def test_invalid_masks_and_targets_fail(self):
        for value in [np.nan, -1, 0.5, 2]:
            with self.subTest(mask=value):
                frame = records(1)
                frame["helmet_nonuse_observed"] = value
                with self.assertRaises(ValueError):
                    prepare_labels(frame, "mask")
        frame = records(1)
        frame["helmet_nonuse"] = 2
        with self.assertRaises(ValueError):
            prepare_labels(frame, "mask")

    def test_encoders_do_not_learn_validation_categories(self):
        frame = prepare_labels(records(), "mask")
        train, validation, test = frame.iloc[:80].copy(), frame.iloc[80:100].copy(), frame.iloc[100:].copy()
        validation["weather"] = "validation_only"
        train, validation, test, encoders, _ = encode_splits(train, validation, test)
        self.assertNotIn("validation_only", encoders["weather"].classes_)
        self.assertTrue(validation["weather"].eq(encoders["weather"].transform(["__UNK__"])[0]).all())

    def test_site_splits_are_disjoint(self):
        frame = prepare_labels(records(500), "mask")
        site = frame[cfg.SITE_ID_COL].iloc[0]
        train, validation, test = split_frames(frame, 42, site)
        sites = [set(part[cfg.SITE_ID_COL]) for part in (train, validation, test)]
        self.assertFalse(sites[0] & sites[1] or sites[0] & sites[2] or sites[1] & sites[2])


class ModelProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.validation, cls.test, cls.encoders, cls.vocab = load_data(label_policy="mask")

    def views(self):
        from src.data_pipeline import RiderDataset
        dataset = RiderDataset(self.test.head(8), self.vocab)
        return dataset.views

    def test_architecture_ablations_have_expected_gate_shapes(self):
        for ablation, width in [("full", 6), ("no_interaction", 5), ("no_site", 5), ("shared_gate", 6), ("early_fusion", None)]:
            with self.subTest(ablation=ablation):
                model = TGMVMTGFNetV2(self.vocab, ablation=ablation).eval()
                predictions, gates, _ = model(self.views())
                self.assertEqual(len(predictions), 4)
                if width is None:
                    self.assertEqual(gates, [])
                else:
                    self.assertEqual(gates[0].shape, (8, width))
                    self.assertTrue(torch.allclose(gates[0].sum(dim=1), torch.ones(8), atol=1e-6))
                if ablation == "shared_gate":
                    self.assertEqual(len(model.fusion.gates), 1)
                    self.assertTrue(torch.equal(gates[0], gates[1]))

    def test_no_site_variant_is_invariant_to_site_inputs(self):
        model = TGMVMTGFNetV2(self.vocab, ablation="no_site").eval()
        views = self.views()
        original = model(views)[0]
        altered = {key: value.clone() for key, value in views.items()}
        altered["site_id"].fill_(self.vocab["unknown_site_id"])
        altered["site_obs"].fill_(0)
        changed = model(altered)[0]
        for first, second in zip(original, changed):
            self.assertTrue(torch.equal(first, second))

    def test_unknown_site_has_zero_intercept_even_with_legacy_weights(self):
        model = TGMVMTGFNetV2(self.vocab).eval()
        views = self.views()
        views["site_id"].fill_(self.vocab["unknown_site_id"])
        with torch.no_grad():
            model.encoders["site"].site_intercept.weight[-1].fill_(100)
        for first, second in zip(model(views, use_site_intercept=True)[0], model(views, use_site_intercept=False)[0]):
            self.assertTrue(torch.equal(first, second))

    def test_no_uncertainty_and_missing_nan_loss(self):
        loss_fn = UncertaintyWeightedLoss(use_focal=False, use_uncertainty=False)
        probabilities = [torch.tensor([0.8, 0.2], requires_grad=True) for _ in cfg.TASK_NAMES]
        targets = torch.tensor([[1.0] * 4, [float("nan")] * 4])
        masks = torch.tensor([[1.0] * 4, [0.0] * 4])
        loss, _ = loss_fn(probabilities, targets, masks)
        self.assertAlmostEqual(float(loss.detach()), -4 * np.log(0.8), places=5)
        loss.backward()
        self.assertEqual(float(probabilities[0].grad[1]), 0.0)
        self.assertIsNone(loss_fn.log_tau_sq.grad)

    def test_fully_unobserved_batch_is_finite_zero(self):
        predictions = [torch.full((2,), 0.5, requires_grad=True) for _ in cfg.TASK_NAMES]
        loss, _ = UncertaintyWeightedLoss()(predictions, torch.full((2, 4), float("nan")), torch.zeros(2, 4))
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()

    def test_checkpoint_roundtrip_uses_exact_rows_and_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "records.csv"
            records(500).to_csv(source, index=False)
            train, validation, test, encoders, vocab = load_data(data_path=source, label_policy="mask")
            model = TGMVMTGFNetV2(vocab, ablation="shared_gate", gate_prior_weight=0.2)
            loss_fn = UncertaintyWeightedLoss(use_focal=False, use_uncertainty=False)
            membership = {name: part["_row_id"].tolist() for name, part in
                          zip(("train", "validation", "test"), (train, validation, test))}
            payload = make_checkpoint_payload(1, model, loss_fn, 0.7, vocab, encoders,
                                              thresholds={task: 0.3 for task in cfg.TASK_NAMES},
                                              data_path=source, temperature=0.4,
                                              split_rows=membership, label_policy="mask")
            checkpoint = Path(directory) / "model.pt"
            torch.save(payload, checkpoint)
            with patch.object(cfg, "TEST_SIZE", 0.4), patch.object(cfg, "VAL_SIZE", 0.2):
                bundle = load_model_bundle(checkpoint)
            self.assertEqual(bundle["test_df"]["_row_id"].tolist(), membership["test"])
            self.assertEqual(bundle["temperature"], 0.4)
            self.assertEqual(bundle["model"].ablation, "shared_gate")
            self.assertFalse(bundle["loss_fn"].use_focal)
            self.assertFalse(bundle["loss_fn"].use_uncertainty)
            self.assertEqual(bundle["model"].fusion.gates[0].prior_weight, 0.2)
            source.write_text(source.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "Dataset hash differs"):
                splits_from_checkpoint(payload)


class CalibrationAndBaselineTest(unittest.TestCase):
    def bootstrap_predictions(self):
        rows = []
        for model in ["m2g_net", "baseline"]:
            for site in [1, 2]:
                for target in [0, 1]:
                    for task in cfg.TASK_NAMES:
                        rows.append({"model": model, "seed": 42, "row_id": 2 * site + target,
                                     "site_id": site, "task": task, "observed": 1, "target": target,
                                     "probability": float(target if model == "m2g_net" else 1 - target)})
        return pd.DataFrame(rows)

    def test_bootstrap_point_estimates_belong_to_each_model(self):
        rows = {row["model"]: row for row in cluster_intervals(self.bootstrap_predictions(), draws=30)}
        self.assertEqual(rows["m2g_net"]["pooled_seed_ensemble_macro_auc"], 1.0)
        self.assertEqual(rows["baseline"]["pooled_seed_ensemble_macro_auc"], 0.0)
        self.assertEqual(rows["baseline"]["m2g_minus_model_ci_lower"], 1.0)

    def test_bootstrap_rejects_unpaired_predictions(self):
        frame = self.bootstrap_predictions()
        with self.assertRaisesRegex(ValueError, "identical rows"):
            cluster_intervals(frame.iloc[1:])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            cluster_intervals(pd.concat([frame, frame.iloc[:1]]))

    def test_bootstrap_rejects_unequal_training_seed_sets(self):
        frame = self.bootstrap_predictions()
        extra_seed = frame.loc[frame["model"].eq("m2g_net")].assign(seed=99)
        different_seed = frame.copy()
        different_seed.loc[different_seed["model"].eq("m2g_net"), "seed"] = 99
        for invalid in [pd.concat([frame, extra_seed], ignore_index=True), different_seed]:
            with self.subTest(count=len(invalid)), self.assertRaisesRegex(ValueError, "identical training seed sets"):
                cluster_intervals(invalid, draws=0)

    def test_bootstrap_averages_predictions_before_scoring(self):
        frame = self.bootstrap_predictions()
        extra_seed = frame.assign(seed=99, probability=0.4 + 0.2 * (1 - frame["probability"]))
        rows = cluster_intervals(pd.concat([frame, extra_seed], ignore_index=True), draws=30)
        for row in rows:
            expected = 1.0 if row["model"] == "m2g_net" else 0.0
            self.assertEqual(row["pooled_seed_ensemble_macro_auc"], expected)
            self.assertEqual(row["auc_ci_lower"], expected)
            self.assertEqual(row["auc_ci_upper"], expected)

    def test_bootstrap_matches_explicit_paired_site_resampling(self):
        from sklearn.metrics import roc_auc_score
        frame = self.bootstrap_predictions()
        frame.loc[frame["site_id"].eq(2), "probability"] = 1 - frame.loc[frame["site_id"].eq(2), "probability"]
        actual = {row["model"]: row for row in cluster_intervals(frame, draws=40, seed=7)}
        scores = {name: [] for name in frame["model"].unique()}
        sites = np.sort(frame["site_id"].unique())
        for sampled in np.random.default_rng(7).integers(len(sites), size=(40, len(sites))):
            resampled = pd.concat([frame.loc[frame["site_id"].eq(sites[index])] for index in sampled])
            for name, model in resampled.groupby("model"):
                scores[name].append(np.mean([roc_auc_score(task["target"], task["probability"])
                                            for _, task in model.groupby("task")]))
        for name, values in scores.items():
            np.testing.assert_allclose([actual[name]["auc_ci_lower"], actual[name]["auc_ci_upper"]],
                                       np.quantile(values, [0.025, 0.975]))
        delta = np.asarray(scores["m2g_net"]) - np.asarray(scores["baseline"])
        np.testing.assert_allclose([actual["baseline"]["m2g_minus_model_ci_lower"],
                                    actual["baseline"]["m2g_minus_model_ci_upper"]],
                                   np.quantile(delta, [0.025, 0.975]))

    def test_bootstrap_omits_intervals_with_insufficient_evidence(self):
        frame = self.bootstrap_predictions()
        missing_task = frame.copy()
        missing_task.loc[missing_task["task"].eq(cfg.TASK_NAMES[0]), "observed"] = 0
        for predictions, draws in [(frame, 0), (frame, 19), (frame.loc[frame["site_id"].eq(1)], 30),
                                   (missing_task, 30)]:
            with self.subTest(rows=len(predictions), draws=draws):
                for row in cluster_intervals(predictions, draws=draws):
                    self.assertNotIn("auc_ci_lower", row)
                    self.assertNotIn("m2g_minus_model_ci_lower", row)

    def test_calibration_rejects_empty_and_invalid_inputs(self):
        for probs, labels in [(np.array([np.nan]), np.array([np.nan])),
                              (np.array([0.5]), np.array([2])),
                              (np.array([np.nan]), np.array([1]))]:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                TemperatureScaler().fit(probs, labels)

    def test_thresholds_are_selected_after_temperature_scaling(self):
        probabilities = np.array([[0.2, 0.3, 0.4, 0.2], [0.4, 0.45, 0.3, 0.4],
                                  [0.25, 0.2, 0.1, 0.3], [0.45, 0.4, 0.45, 0.4]])
        targets = np.array([[0] * 4, [1] * 4, [0] * 4, [1] * 4])
        masks = np.ones_like(targets)
        scaler, thresholds = fit_validation_rule(targets, probabilities, masks)
        self.assertEqual(thresholds, tune_task_thresholds(targets, scaler.transform(probabilities), masks))

    def test_baseline_selection_uses_train_only_categories(self):
        frame = prepare_labels(records(1000), "mask")
        train, validation, test, encoders, _ = encode_splits(*split_frames(frame, 42))
        models, selections = fit_tuned_baseline("decision_tree", train, validation, quick=True)
        self.assertEqual(len(selections), 4)
        probabilities = predict_baseline(models, test)
        self.assertEqual(probabilities.shape, (len(test), 4))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertEqual(models[0].named_steps["decisiontreeclassifier"].tree_.n_node_samples[0],
                         int(extract_task_masks(train)[:, 0].sum()))

    def test_json_outputs_do_not_contain_nonstandard_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "metrics.json"
            save_json(output, {"score": float("nan"), "value": np.int64(2)})
            self.assertIn('"score": null', output.read_text())

    def test_run_resume_reuses_and_verifies_completed_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "records.csv"
            output = Path(directory) / "run"
            records(1000).to_csv(data, index=False)
            arguments = ["--data", str(data), "--label-policy", "mask", "--models", "decision_tree",
                         "--seeds", "42", "--smoke", "--gate-priors", "0.1", "--bootstrap", "0"]
            main([*arguments, "--output", str(output)])
            with patch("scripts.evaluate.paper_pipeline.fit_tuned_baseline", side_effect=AssertionError("Must not retrain")):
                main([*arguments, "--resume", str(output)])
            predictions = output / "random_seed_42" / "decision_tree" / "predictions.csv"
            predictions.write_text(predictions.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "completed artifact changed"):
                main([*arguments, "--resume", str(output)])


    def test_resume_rejects_corrupted_or_missing_source_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "records.csv"
            output = Path(directory) / "run"
            records(1000).to_csv(data, index=False)
            arguments = ["--data", str(data), "--label-policy", "mask", "--models", "decision_tree",
                         "--seeds", "42", "--smoke", "--gate-priors", "0.1", "--bootstrap", "0"]
            main([*arguments, "--output", str(output)])
            manifest_bytes = (output / "manifest.json").read_bytes()
            source = output / "source" / "config.py"
            source.write_text(source.read_text() + "\nCHANGED = True\n")
            for missing in [False, True]:
                with self.subTest(missing=missing):
                    if missing:
                        source.unlink()
                    with self.assertRaisesRegex(ValueError, "source snapshot changed or missing"):
                        main([*arguments, "--resume", str(output)])
                    self.assertEqual((output / "manifest.json").read_bytes(), manifest_bytes)

    def test_resume_requires_all_expected_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "records.csv"
            output = Path(directory) / "run"
            records(1000).to_csv(data, index=False)
            arguments = ["--data", str(data), "--label-policy", "mask", "--models", "decision_tree",
                         "--seeds", "42", "--smoke", "--gate-priors", "0.1", "--bootstrap", "0"]
            main([*arguments, "--output", str(output)])
            model_output = output / "random_seed_42" / "decision_tree"
            complete = json.loads((model_output / "complete.json").read_text())
            for artifact in complete["artifact_sha256"]:
                with self.subTest(artifact=artifact):
                    incomplete = {**complete, "artifact_sha256": {
                        name: digest for name, digest in complete["artifact_sha256"].items() if name != artifact}}
                    save_json(model_output / "complete.json", incomplete)
                    with self.assertRaisesRegex(ValueError, "artifact hash set"):
                        main([*arguments, "--resume", str(output)])
            (model_output / "model.joblib").unlink()
            save_json(model_output / "complete.json", complete)
            with self.assertRaisesRegex(ValueError, "completed artifact changed or missing"):
                main([*arguments, "--resume", str(output)])

    def test_resume_rejects_mismatched_completion_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "records.csv"
            output = Path(directory) / "run"
            records(1000).to_csv(data, index=False)
            arguments = ["--data", str(data), "--label-policy", "mask", "--models", "decision_tree",
                         "--seeds", "42", "--smoke", "--gate-priors", "0.1", "--bootstrap", "0"]
            main([*arguments, "--output", str(output)])
            complete_path = output / "random_seed_42" / "decision_tree" / "complete.json"
            complete = json.loads(complete_path.read_text())
            for field, value in [("model", "catboost"), ("seed", 99), ("fold", "site_1")]:
                with self.subTest(field=field):
                    save_json(complete_path, {**complete, field: value})
                    with self.assertRaisesRegex(ValueError, "completion identity"):
                        main([*arguments, "--resume", str(output)])

if __name__ == "__main__":
    unittest.main()
