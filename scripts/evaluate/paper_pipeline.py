"""Validation-only selection, matched model comparisons, auditable artifacts."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

import config as cfg
from baselines.run_all_baselines import (
    fit_tuned_baseline, predict_baseline, fit_validation_mlp, predict_mlp,
)
from scripts.train import collect_predictions, fit_model_for_split
from src.calibration import fit_validation_rule, prediction_metrics
from src.checkpoint import dataset_sha256, make_checkpoint_payload
from src.data_pipeline import (
    prepare_labels, stratified_split, encode_splits, get_loaders,
    extract_targets, extract_task_masks,
    encoders_to_state,
)
from src.metrics import compute_all_metrics

NETWORKS = ["m2g_net", "m2g_net_v3", "no_interaction", "no_site", "shared_gate", "early_fusion", "no_uncertainty", "complete_case"]
BASELINES = ["logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm", "catboost", "early_fusion_mlp", "single_task_mlp"]

def package_versions():
    versions = {}
    for name in ["torch", "numpy", "pandas", "scikit-learn", "scipy", "xgboost", "lightgbm", "catboost", "joblib"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def save_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {str(key): clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        if isinstance(item, np.generic):
            return clean(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def data_audit(raw):
    return {
        "n_records": len(raw), "n_sites": int(raw[cfg.SITE_ID_COL].nunique()),
        "missing_mask_columns": [col for col in cfg.TASK_MASK_COLS.values() if col not in raw],
        "red_light_positives_without_signal": int((raw["has_signal"].astype(str).str.lower().eq("no") & raw["red_light_running"].eq(1)).sum()),
    }


def observation_tables(frame, output):
    rows = []
    labels = frame[cfg.TASK_NAMES].where(frame[list(cfg.TASK_MASK_COLS.values())].to_numpy(dtype=bool))
    for task in cfg.TASK_NAMES:
        observed = labels[task].dropna()
        rows.append({"task": task, "n_total": len(frame), "n_observed": len(observed),
                     "n_positive": int(observed.sum()), "prevalence": float(observed.mean())})
    pd.DataFrame(rows).to_csv(output / "task_observation.csv", index=False)
    labels.corr().to_csv(output / "label_correlation.csv")
    cooccurrence = pd.DataFrame(index=cfg.TASK_NAMES, columns=cfg.TASK_NAMES, dtype=float)
    for first in cfg.TASK_NAMES:
        for second in cfg.TASK_NAMES:
            jointly_observed = labels[first].notna() & labels[second].notna()
            cooccurrence.loc[first, second] = (labels.loc[jointly_observed, first].eq(1) & labels.loc[jointly_observed, second].eq(1)).mean()
    cooccurrence.to_csv(output / "label_cooccurrence.csv")


def split_frames(frame, split_seed, held_out_site=None):
    if held_out_site is None:
        return stratified_split(frame, seed=split_seed)
    development = frame.loc[frame[cfg.SITE_ID_COL] != held_out_site]
    test = frame.loc[frame[cfg.SITE_ID_COL] == held_out_site]
    if development[cfg.SITE_ID_COL].nunique() < 2 or test.empty:
        raise ValueError("Site-out evaluation needs a test site and at least two other sites.")
    train_indices, val_indices = next(GroupShuffleSplit(
        n_splits=1, test_size=cfg.VAL_SIZE / (1 - cfg.TEST_SIZE), random_state=split_seed,
    ).split(development, groups=development[cfg.SITE_ID_COL]))
    return tuple(part.reset_index(drop=True) for part in
                 (development.iloc[train_indices], development.iloc[val_indices], test))


class ProbabilityEnsemble(torch.nn.Module):
    """Mean of member probabilities; mimics TGMVMTGFNetV2.forward for collect_predictions.

    ponytail: checkpoint is a plain state_dict of all members (not loadable by load_model_bundle);
    add an ensemble loader if v3 weights must be reused outside the paper pipeline.
    """

    def __init__(self, members):
        super().__init__()
        self.members = torch.nn.ModuleList(members)
        self.ablation = members[0].ablation
        self.gate_variant = getattr(members[0], "gate_variant", "legacy")
        self.interaction_skip = getattr(members[0], "interaction_skip", False)
        self.fusion = members[0].fusion

    def forward(self, views, use_site_intercept=True):
        outputs = [member(views, use_site_intercept=use_site_intercept) for member in self.members]
        mean = [torch.stack([out[0][k] for out in outputs]).mean(0) for k in range(len(outputs[0][0]))]
        return mean, outputs[0][1], outputs[0][2]


def select_ensemble(train, validation, vocab, encoders, seed, args):
    members, losses, selections = [], [], []
    for index in range(cfg.V3_ENSEMBLE_SIZE):
        model, loss_fn, score, epoch = fit_model_for_split(
            vocab, train, validation, encoders=encoders,
            split_seed=seed + index * cfg.V3_MEMBER_SEED_OFFSET,
            max_epochs=args.max_epochs, patience=args.patience, ablation="full",
            gate_prior_weight=cfg.GATE_PRIOR_WEIGHT, use_focal=cfg.V3_USE_FOCAL_LOSS,
            interaction_skip=cfg.V3_INTERACTION_SKIP, gate_variant=cfg.V3_GATE_VARIANT,
            validation_use_site_intercept=args.split == "random")
        members.append(model)
        losses.append(loss_fn)
        selections.append({"member": index, "member_seed": seed + index * cfg.V3_MEMBER_SEED_OFFSET,
                           "gate_prior_weight": cfg.GATE_PRIOR_WEIGHT, "validation_roc_auc": score, "best_epoch": epoch})
    ensemble = ProbabilityEnsemble(members)
    _, val_loader, _ = get_loaders(train, validation, validation, vocab)
    y, probs, masks = collect_predictions(ensemble, val_loader, args.split == "random")
    score = compute_all_metrics(y, probs, masks)["macro"]["roc_auc"]
    selections.append({"ensemble_validation_roc_auc": float(score)})
    return (ensemble, losses[0], float(score), selections[0]["best_epoch"]), selections, train, validation


def select_network(name, train, validation, vocab, encoders, seed, args):
    if name == "complete_case":
        columns = list(cfg.TASK_MASK_COLS.values())
        train = train.loc[train[columns].eq(1).all(axis=1)].copy()
        validation = validation.loc[validation[columns].eq(1).all(axis=1)].copy()
        if train.empty or validation.empty:
            raise ValueError("Complete-case ablation has no fully observed train/validation records.")
    if name == "m2g_net_v3":
        return select_ensemble(train, validation, vocab, encoders, seed, args)
    architecture = name if name in {"no_interaction", "no_site", "shared_gate", "early_fusion"} else "full"
    candidates = [None] if architecture == "early_fusion" else args.gate_priors
    best, selections = None, []
    for prior in candidates:
        model, loss_fn, score, epoch = fit_model_for_split(
            vocab, train, validation, encoders=encoders, split_seed=seed,
            max_epochs=args.max_epochs, patience=args.patience,
            ablation=architecture, gate_prior_weight=prior,
            use_uncertainty=name != "no_uncertainty",
            validation_use_site_intercept=args.split == "random",
        )
        selections.append({"gate_prior_weight": prior, "validation_roc_auc": score, "best_epoch": epoch})
        if best is None or score > best[2]:
            best = (model, loss_fn, score, epoch)
    return best, selections, train, validation


def evaluate_model(name, seed, fold, raw_frames, args, output):
    train, validation, test, encoders, vocab = encode_splits(*raw_frames)
    model_output = output / f"{fold}_seed_{seed}" / name
    artifacts = ["metrics.csv", "predictions.csv", "subgroups.csv", "evaluation.json", "validation_predictions.npz",
                 "model.pt" if name in NETWORKS else "model.joblib"]
    if args.resume and (model_output / "complete.json").is_file():
        complete = json.loads((model_output / "complete.json").read_text())
        identity = {"model": name, "seed": seed, "fold": fold}
        if not isinstance(complete, dict) or any(complete.get(key) != value for key, value in identity.items()):
            raise ValueError(f"Cannot resume: completion identity differs: {model_output}")
        hashes = complete.get("artifact_sha256")
        if not isinstance(hashes, dict) or set(hashes) != set(artifacts):
            raise ValueError(f"Cannot resume: completed artifact hash set differs: {model_output}")
        for filename, expected in complete["artifact_sha256"].items():
            artifact = model_output / filename
            if not artifact.is_file() or dataset_sha256(artifact) != expected:
                raise ValueError(f"Cannot resume: completed artifact changed or missing: {artifact}")
        return (pd.read_csv(model_output / "metrics.csv").to_dict("records"),
                pd.read_csv(model_output / "predictions.csv"),
                pd.read_csv(model_output / "subgroups.csv").to_dict("records"))
    if args.resume and model_output.exists():
        model_output.rename(model_output.with_name(f"{name}.partial-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"))
    model_output.mkdir(parents=True)
    unseen = args.split == "site"
    if name in NETWORKS:
        (model, loss_fn, score, epoch), selections, fit_train, fit_val = select_network(
            name, train, validation, vocab, encoders, seed, args)
        _, val_loader, test_loader = get_loaders(fit_train, fit_val, test, vocab)
        val_y, val_probs, val_masks = collect_predictions(model, val_loader, not unseen)
        test_y, test_probs, test_masks = collect_predictions(model, test_loader, not unseen)
    else:
        fit_train, fit_val = train, validation
        val_y, test_y = extract_targets(validation).numpy(), extract_targets(test).numpy()
        val_masks, test_masks = extract_task_masks(validation).numpy(), extract_task_masks(test).numpy()
        if name.endswith("_mlp"):
            model, encoder, selections = fit_validation_mlp(
                train, validation, seed, args.max_epochs, args.patience, name == "single_task_mlp")
            val_probs = predict_mlp(model, encoder, validation, name == "single_task_mlp")
            test_probs = predict_mlp(model, encoder, test, name == "single_task_mlp")
            joblib.dump({"model": model, "encoder": encoder}, model_output / "model.joblib")
        else:
            models, selections = fit_tuned_baseline(name, train, validation, seed, args.smoke)
            val_probs, test_probs = predict_baseline(models, validation), predict_baseline(models, test)
            joblib.dump(models, model_output / "model.joblib")
    scaler, thresholds = fit_validation_rule(val_y, val_probs, val_masks)
    membership = {key: frame["_row_id"].tolist() for key, frame in
                  zip(("train", "validation", "test"), (fit_train, fit_val, test))}
    if name in NETWORKS:
        payload = make_checkpoint_payload(
            epoch, model, loss_fn, score, vocab, encoders, thresholds,
            split_seed=args.split_seed, data_path=args.data, temperature=scaler.temperature,
            split_rows=membership, label_policy=args.label_policy,
        )
        payload["validation_use_site_intercept"] = not unseen
        payload["test_use_site_intercept"] = not unseen
        torch.save(payload, model_output / "model.pt")
    reports = prediction_metrics(test_y, test_probs, test_masks, scaler, thresholds)
    save_json(model_output / "evaluation.json", {
        "model": name, "seed": seed, "fold": fold, "selection": selections,
        "temperature": scaler.temperature, "thresholds": thresholds,
        "split_rows": membership, "metrics": reports,
        "dataset_sha256": dataset_sha256(args.data),
        "preprocessing": {"encoder_classes": encoders_to_state(encoders), "site_mapping": vocab["site_mapping"]},
        "validation_use_site_intercept": not unseen, "test_use_site_intercept": not unseen,
    })
    calibrated = scaler.transform(test_probs)
    np.savez_compressed(model_output / "validation_predictions.npz", targets=val_y, probabilities=val_probs, masks=val_masks)
    rows, predictions, subgroups = [], [], []
    for rule, results in reports.items():
        for task, metrics in results.items():
            rows.append({"model": name, "seed": seed, "fold": fold, "rule": rule, "task": task, **metrics})
    for task_idx, task in enumerate(cfg.TASK_NAMES):
        predictions.append(pd.DataFrame({
            "model": name, "seed": seed, "fold": fold, "row_id": test["_row_id"],
            "site_id": raw_frames[2][cfg.SITE_ID_COL], "task": task,
            "target": test_y[:, task_idx], "observed": test_masks[:, task_idx].astype(int),
            "raw_probability": test_probs[:, task_idx], "probability": calibrated[:, task_idx],
            "threshold": thresholds[task],
        }))
    for column in ["gender", "age_group"]:
        for value in raw_frames[2][column].unique():
            group = raw_frames[2][column].eq(value).to_numpy()
            metrics = compute_all_metrics(test_y[group], calibrated[group], test_masks[group], thresholds)
            for task_idx, task in enumerate(cfg.TASK_NAMES):
                observed = test_masks[group, task_idx].astype(bool)
                subgroups.append({"model": name, "seed": seed, "fold": fold, "group_variable": column,
                                  "group_value": value, "task": task, "n": int(group.sum()),
                                  "n_observed": int(observed.sum()),
                                  "n_positive": int(test_y[group, task_idx][observed].sum()), **metrics[task]})
    prediction_frame = pd.concat(predictions, ignore_index=True)
    prediction_frame.to_csv(model_output / "predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(model_output / "metrics.csv", index=False)
    pd.DataFrame(subgroups).to_csv(model_output / "subgroups.csv", index=False)
    save_json(model_output / "complete.json", {"model": name, "seed": seed, "fold": fold,
              "artifact_sha256": {filename: dataset_sha256(model_output / filename) for filename in artifacts}})
    return rows, prediction_frame, subgroups


def cluster_intervals(predictions, draws=1000, seed=42):
    from sklearn.metrics import roc_auc_score
    if draws < 0 or predictions.empty:
        raise ValueError("Bootstrap needs predictions and a nonnegative draw count.")
    if predictions.duplicated(["model", "seed", "row_id", "task"]).any():
        raise ValueError("Duplicate per-model/seed/row/task predictions.")
    if len({frozenset(frame["seed"]) for _, frame in predictions.groupby("model")}) != 1:
        raise ValueError("Paired bootstrap requires identical training seed sets for every model.")
    reference = None
    for _, frame in predictions.groupby(["model", "seed"]):
        identity = frame.sort_values(["row_id", "task"])[["row_id", "site_id", "task", "target", "observed"]].reset_index(drop=True)
        if reference is not None and not reference.equals(identity):
            raise ValueError("Paired bootstrap requires identical rows, sites, labels, and masks for every model/seed.")
        reference = identity
    ensemble = predictions.groupby(["model", "row_id", "site_id", "task"], as_index=False).agg(
        target=("target", "first"), observed=("observed", "first"), probability=("probability", "mean"))
    sites = np.sort(ensemble["site_id"].unique())
    rng = np.random.default_rng(seed)
    sampled_sites = rng.integers(len(sites), size=(draws, len(sites)))
    scores_by_model, points_by_model = {}, {}
    for name, frame in ensemble.groupby("model"):
        scores = []
        task_arrays = []
        for task in cfg.TASK_NAMES:
            observed = frame.loc[frame["task"].eq(task) & frame["observed"].eq(1)]
            task_arrays.append((observed["target"].to_numpy(), observed["probability"].to_numpy(),
                                np.searchsorted(sites, observed["site_id"].to_numpy())))
        for sampled in sampled_sites:
            counts = np.bincount(sampled, minlength=len(sites))
            task_scores = []
            for targets, probabilities, site_indices in task_arrays:
                weights = counts[site_indices]
                if len(np.unique(targets[weights > 0])) == 2:
                    task_scores.append(roc_auc_score(targets, probabilities, sample_weight=weights))
            scores.append(float(np.mean(task_scores)) if len(task_scores) == cfg.NUM_TASKS else np.nan)
        scores_by_model[name] = np.asarray(scores)
        point_scores = [roc_auc_score(targets, probabilities) for targets, probabilities, _ in task_arrays
                        if len(np.unique(targets)) == 2]
        points_by_model[name] = float(np.mean(point_scores)) if len(point_scores) == cfg.NUM_TASKS else np.nan
    rows = []
    comparisons = max(len(scores_by_model) - 1, 1)
    for name, scores in scores_by_model.items():
        finite = scores[np.isfinite(scores)]
        row = {"model": name, "valid_bootstraps": len(finite), "n_sites": len(sites),
               "pooled_seed_ensemble_macro_auc": points_by_model[name]}
        if len(sites) >= 2 and len(finite) >= max(20, int(0.8 * draws)):
            row.update({"auc_ci_lower": np.quantile(finite, 0.025), "auc_ci_upper": np.quantile(finite, 0.975)})
            if name != "m2g_net" and "m2g_net" in scores_by_model:
                delta = scores_by_model["m2g_net"] - scores
                delta = delta[np.isfinite(delta)]
                if len(delta) >= max(20, int(0.8 * draws)):
                    lower, upper = np.quantile(delta, [0.025 / comparisons, 1 - 0.025 / comparisons])
                    row.update({"m2g_minus_model_ci_lower": lower, "m2g_minus_model_ci_upper": upper})
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=cfg.DATA_PATH)
    parser.add_argument("--label-policy", choices=["strict", "mask"], default=cfg.LABEL_POLICY)
    parser.add_argument("--require-masks", action="store_true")
    parser.add_argument("--models", nargs="+", choices=NETWORKS + BASELINES, default=NETWORKS + BASELINES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 2023, 777, 999])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split", choices=["random", "site"], default="random")
    parser.add_argument("--sites", nargs="+", type=int)
    parser.add_argument("--gate-priors", nargs="+", type=float, default=[0.0, 0.1, 0.2])
    parser.add_argument("--max-epochs", type=int, default=cfg.MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=cfg.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output")
    parser.add_argument("--resume", help="Resume an existing run only when data, code, config, and arguments match.")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.max_epochs < 1 or args.patience < 1 or args.bootstrap < 0 or any(not 0 <= prior <= 1 for prior in args.gate_priors):
        parser.error("Epochs/patience must be positive, bootstrap nonnegative, and gate priors in [0, 1].")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.models)) != len(args.models):
        parser.error("Models and seeds must be unique.")
    if args.sites and args.split != "site":
        parser.error("--sites requires --split site.")
    if args.resume and args.output:
        parser.error("Use --resume or --output, not both.")
    if args.smoke:
        args.max_epochs, args.patience = min(args.max_epochs, 2), 1
    torch.set_num_threads(1)
    for library in ["xgboost", "lightgbm", "catboost"]:
        if library in args.models:
            importlib.import_module(library)
    raw = pd.read_csv(args.data)
    raw["_row_id"] = np.arange(len(raw))
    frame = prepare_labels(raw, args.label_policy, args.require_masks)
    audit = data_audit(raw)
    folds = (args.sites or sorted(frame[cfg.SITE_ID_COL].unique())) if args.split == "site" else [None]
    if args.split == "site" and not set(folds).issubset(set(frame[cfg.SITE_ID_COL])):
        parser.error("Unknown requested held-out site.")
    output = Path(args.resume or args.output or f"outputs/paper_pipeline/{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}")
    manifest = {"status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
                "arguments": vars(args), "dataset_sha256": dataset_sha256(args.data), "data_audit": audit,
                "result_kind": "smoke_not_for_publication" if args.smoke else "development_not_external_validation",
                "python": platform.python_version(), "packages": package_versions(),
                "known_development_test": "data/test/independent_test_set.csv was used in previous model selection",
                "config": json.loads(json.dumps({key: value for key, value in vars(cfg).items() if key.isupper()})),
                "source_sha256": {str(path): dataset_sha256(path) for path in
                                  [*(path for path in sorted(Path("src").glob("*.py")) if not path.name.startswith("._")), Path(__file__).relative_to(Path.cwd()),
                                   Path("config.py"), Path("scripts/train.py"), Path("baselines/run_all_baselines.py")]},
                "split_seed_fixed_across_training_seeds": True}
    if args.resume:
        previous = json.loads((output / "manifest.json").read_text())
        excluded_arguments = {"resume", "output"}
        for key in ["dataset_sha256", "config", "source_sha256", "packages"]:
            if previous.get(key) != manifest[key]:
                raise ValueError(f"Cannot resume: {key} changed.")
        old_args = {key: value for key, value in previous["arguments"].items() if key not in excluded_arguments}
        new_args = {key: value for key, value in vars(args).items() if key not in excluded_arguments}
        if old_args != new_args:
            raise ValueError("Cannot resume: experiment arguments changed; supply the original arguments.")
        for source, expected in previous["source_sha256"].items():
            snapshot = output / "source" / source
            if not snapshot.is_file() or dataset_sha256(snapshot) != expected:
                raise ValueError(f"Cannot resume: source snapshot changed or missing: {snapshot}")
        manifest["created_at"] = previous["created_at"]
    else:
        output.mkdir(parents=True, exist_ok=False)
        for source in manifest["source_sha256"]:
            destination = output / "source" / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    save_json(output / "manifest.json", manifest)
    observation_tables(frame, output)
    all_rows, all_predictions, all_subgroups = [], [], []
    try:
        for site in folds:
            raw_frames = split_frames(frame, args.split_seed, site)
            for seed in args.seeds:
                for name in args.models:
                    fold = "random" if site is None else f"site_{site}"
                    print(f"{fold} seed={seed} model={name}", flush=True)
                    rows, predictions, subgroups = evaluate_model(name, seed, fold, raw_frames, args, output)
                    all_rows.extend(rows)
                    all_predictions.append(predictions)
                    all_subgroups.extend(subgroups)
                    manifest["last_completed"] = {"fold": fold, "seed": seed, "model": name}
                    save_json(output / "manifest.json", manifest)
        metrics = pd.DataFrame(all_rows)
        predictions = pd.concat(all_predictions, ignore_index=True)
        metrics.to_csv(output / "metrics.csv", index=False)
        predictions.to_csv(output / "predictions.csv", index=False)
        pd.DataFrame(all_subgroups).to_csv(output / "subgroups.csv", index=False)
        summary = metrics.loc[metrics["task"].eq("macro") & metrics["rule"].eq("calibrated_tuned")].groupby("model")[
            ["roc_auc", "pr_auc", "f1", "balanced_acc", "brier", "ece"]].agg(["mean", "std"])
        summary.to_csv(output / "summary.csv")
        save_json(output / "cluster_bootstrap.json", cluster_intervals(predictions, args.bootstrap, args.split_seed))
        report = ["# Pipeline Results", "", f"Result kind: {manifest['result_kind']}", "",
                  "These are development results, not untouched external-validation evidence.",
                  "All selection uses validation only. Repeated seeds keep test rows fixed.",
                  "Cluster intervals resample sites from pooled, seed-averaged predictions; models are not refitted.",
                  "Paired delta intervals use Bonferroni-adjusted quantiles across comparisons.",
                  "No population significance claim follows from fold/seed standard deviations.", "",
                  "| Model | Mean macro AUC | Mean macro F1 | Mean macro ECE |", "|---|---:|---:|---:|"]
        for name, row in summary.iterrows():
            report.append(f"| {name} | {row[('roc_auc', 'mean')]:.4f} | {row[('f1', 'mean')]:.4f} | {row[('ece', 'mean')]:.4f} |")
        (output / "report.md").write_text("\n".join(report) + "\n")
        manifest["status"] = "complete"
        save_json(output / "manifest.json", manifest)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(output / "manifest.json", manifest)
        raise
    print(f"Artifacts: {output}")
    return output


if __name__ == "__main__":
    main()
