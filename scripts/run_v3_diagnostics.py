"""Bounded legacy M5 DEVELOPMENT train/validation diagnosis; score-only by default.

Run --phase fit only after the separate full project preflight. No held-out,
locked-generator, oracle or frozen-opponent predictions are read or produced.
Train/validation gaps are descriptive diagnostics, not proof of overfitting.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import fcntl
import itertools
import json
import math
import multiprocessing
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as cfg
from scripts.run_v3_gate_dev import (
    align_validation, dataset_path, identity_for, project_path, reference_arrays,
    sha256, validation_frames,
)


SPLITS = ("train", "validation")
ARRAY_KEYS = ("row_ids", "targets", "probabilities", "masks")
ARTIFACTS = ("predictions.npz", "telemetry.json")


def diagnostic_identity(job):
    identity = identity_for((*job, "legacy"))
    identity["sources"][str(Path(__file__).relative_to(ROOT))] = sha256(__file__)
    return dict(identity, diagnostic_schema=1, ensemble_size=cfg.V3_ENSEMBLE_SIZE,
                scope="DEVELOPMENT train/validation only; legacy M5")


def member_identity(identity, index):
    return dict(identity, member=index,
                member_seed=identity["job"][2] + index * cfg.V3_MEMBER_SEED_OFFSET)


def unit_path(job):
    mechanism, generator, training = job
    dataset_path(mechanism, generator)
    if training not in cfg.V3_DEV_TRAINING_SEEDS:
        raise ValueError("Unconfigured development training seed")
    return project_path(cfg.V3_DIAGNOSTIC_ROOT) / f"{mechanism}-g{generator}-t{training}-legacy"


def split_arrays(arrays, split):
    return {key: arrays[f"{split}_{key}"] for key in ARRAY_KEYS}


def align_member(arrays, references):
    for split in SPLITS:
        align_validation(split_arrays(arrays, split), references[split])
    if np.intersect1d(arrays["train_row_ids"], arrays["validation_row_ids"]).size:
        raise ValueError("Train and validation rows overlap")


def metric_pair(metrics, task):
    values = {key: float(metrics[task][key]) for key in ("roc_auc", "brier")}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"Nonfinite diagnostic metrics for {task}")
    return values


def validate_telemetry(telemetry):
    history, best = telemetry["history"], telemetry["best"]
    if not history or [row["epoch"] for row in history] != list(range(1, len(history) + 1)):
        raise ValueError("Incomplete diagnostic epoch history")
    if not 1 <= best["epoch"] <= len(history):
        raise ValueError("Selected checkpoint epoch is outside history")
    for record in history:
        for key in ("training_metrics_dropout", "validation_metrics"):
            for task in (*cfg.TASK_NAMES, "macro"):
                metric_pair(record[key], task)
    selected = max(history, key=lambda row: row["validation_metrics"]["macro"]["roc_auc"])
    if best["epoch"] != selected["epoch"]:
        raise ValueError("Selected epoch differs from first maximum validation macro AUC")
    for key in ("train_metrics", "validation_metrics"):
        for task in (*cfg.TASK_NAMES, "macro"):
            metric_pair(best[key], task)


def write_cache(directory, identity, arrays, telemetry, references):
    align_member(arrays, references)
    validate_telemetry(telemetry)
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".member-", dir=directory.parent))
    try:
        np.savez_compressed(staging / "predictions.npz", **arrays)
        (staging / "telemetry.json").write_text(json.dumps(
            telemetry, indent=2, allow_nan=False,
            default=lambda value: value.item() if isinstance(value, np.generic) else value))
        (staging / "complete.json").write_text(json.dumps({
            "identity": identity,
            "artifact_sha256": {name: sha256(staging / name) for name in ARTIFACTS},
        }, indent=2, sort_keys=True, allow_nan=False))
        staging.rename(directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def read_cache(directory, expected_identity, references):
    manifest = json.loads((directory / "complete.json").read_text())
    if manifest.get("identity") != expected_identity:
        raise ValueError(f"Stale diagnostic cache identity: {directory}")
    if set(manifest.get("artifact_sha256", {})) != set(ARTIFACTS):
        raise ValueError(f"Incomplete diagnostic artifact manifest: {directory}")
    for name in ARTIFACTS:
        if manifest["artifact_sha256"][name] != sha256(directory / name):
            raise ValueError(f"Diagnostic cache artifact changed: {directory / name}")
    with np.load(directory / "predictions.npz", allow_pickle=False) as data:
        arrays = {f"{split}_{key}": data[f"{split}_{key}"] for split in SPLITS for key in ARRAY_KEYS}
    telemetry = json.loads((directory / "telemetry.json").read_text())
    align_member(arrays, references)
    validate_telemetry(telemetry)
    return arrays, telemetry


def fit_unit(job):
    import torch
    from scripts.train import collect_predictions, fit_model_for_split
    from src.data_pipeline import encode_splits, get_loaders

    torch.set_num_threads(1)
    directory = unit_path(job)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".fit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = diagnostic_identity(job)
        raw_train, raw_validation = validation_frames(*job[:2])
        references = {"train": reference_arrays(raw_train), "validation": reference_arrays(raw_validation)}
        train, validation, _, encoders, vocab = encode_splits(raw_train, raw_validation, raw_validation)
        del _
        # get_loaders' training iterator shuffles: use its validation iterator for BOTH eval splits.
        _, train_eval, _ = get_loaders(train, train, train, vocab)
        _, val_eval, _ = get_loaders(train, validation, validation, vocab)
        for index in range(cfg.V3_ENSEMBLE_SIZE):
            expected = member_identity(identity, index)
            member_dir = directory / f"member-{index}"
            if member_dir.exists():
                read_cache(member_dir, expected, references)
                print(f"cached diagnostic {job} member={index}", flush=True)
                continue
            telemetry = {}
            model, _, _, selected_epoch = fit_model_for_split(
                vocab, train, validation, encoders=encoders,
                split_seed=expected["member_seed"], data_path=str(dataset_path(*job[:2])),
                gate_variant="legacy", gate_prior_weight=cfg.GATE_PRIOR_WEIGHT,
                use_focal=False, interaction_skip=False, validation_use_site_intercept=True,
                diagnostics=telemetry)
            if telemetry.get("best", {}).get("epoch") != selected_epoch:
                raise ValueError("Training API did not return selected-checkpoint diagnostics")
            arrays = {}
            numpy_state = np.random.get_state()
            try:
                with torch.random.fork_rng(devices=[]):
                    for split, loader in (("train", train_eval), ("validation", val_eval)):
                        y, probabilities, masks = collect_predictions(model, loader, use_site_intercept=True)
                        values = dict(row_ids=references[split]["row_ids"], targets=y,
                                      probabilities=probabilities, masks=masks)
                        arrays.update({f"{split}_{key}": value for key, value in values.items()})
            finally:
                np.random.set_state(numpy_state)
            if diagnostic_identity(job) != identity:
                raise ValueError("Source/config/data changed during fit; refusing diagnostic publication")
            write_cache(member_dir, expected, arrays, telemetry, references)
            print(f"fitted diagnostic {job} member={index} best_epoch={selected_epoch}", flush=True)
            del model


def run_fits(jobs):
    with ProcessPoolExecutor(max_workers=cfg.V3_DEV_WORKERS,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = iter(jobs)
        futures = {pool.submit(fit_unit, job) for job in itertools.islice(pending, cfg.V3_DEV_WORKERS)}
        try:
            while futures:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    future.result()
                futures -= completed
                for job in itertools.islice(pending, len(completed)):
                    futures.add(pool.submit(fit_unit, job))
        except BaseException:
            for future in futures:
                future.cancel()
            raise  # No queued units; at most the already-running bounded workers finish.


def ensemble_arrays(members, references):
    if not members:
        raise ValueError("Cannot average an empty diagnostic ensemble")
    for arrays in members:
        align_member(arrays, references)
    return {f"{split}_{key}": (
        np.mean([split_arrays(arrays, split)[key] for arrays in members], axis=0)
        if key == "probabilities" else references[split][key])
        for split in SPLITS for key in ARRAY_KEYS}


def metric_rows(arrays):
    from src.metrics import compute_all_metrics

    metrics = {split: compute_all_metrics(values["targets"], values["probabilities"], values["masks"])
               for split in SPLITS for values in (split_arrays(arrays, split),)}
    rows = []
    for task in (*cfg.TASK_NAMES, "macro"):
        train, validation = (metric_pair(metrics[split], task) for split in SPLITS)
        rows.append(dict(task=task, train_auc=train["roc_auc"], validation_auc=validation["roc_auc"],
                         auc_gap=train["roc_auc"] - validation["roc_auc"],
                         train_brier=train["brier"], validation_brier=validation["brier"],
                         brier_gap=validation["brier"] - train["brier"]))
    return rows, metrics


def group_metrics(frame):
    columns = ["train_auc", "validation_auc", "auc_gap", "train_brier", "validation_brier", "brier_gap"]
    # First average members within units, then metrics over training seeds, NOT predictions.
    units = frame.groupby(["mechanism", "generator_seed", "training_seed", "estimator", "task"])[columns].mean()
    generators = units.groupby(["mechanism", "generator_seed", "estimator", "task"])[columns].mean().reset_index()
    summary = []
    for keys, group in generators.groupby(["mechanism", "estimator", "task"]):
        for metric in columns:
            values = group[metric].to_numpy()
            summary.append(dict(zip(("mechanism", "estimator", "task"), keys), metric=metric,
                                n_generators=len(values), mean=float(values.mean()),
                                sd=float(values.std(ddof=1)) if len(values) > 1 else None,
                                mcse=float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
                                min=float(values.min()), max=float(values.max())))
    return generators, summary


def score(jobs):
    import pandas as pd

    if not jobs:
        raise ValueError("No configured diagnostic jobs")
    initial_sources = diagnostic_identity(jobs[0])["sources"]
    rows, curves, epochs, provenance = [], [], [], dict(initial_sources)
    for job in jobs:
        mechanism, generator, training = job
        identity = diagnostic_identity(job)
        if identity["sources"] != initial_sources:
            raise ValueError("Source/config changed between diagnostic units")
        data = dataset_path(mechanism, generator)
        provenance[str(data.relative_to(ROOT))] = identity["dataset_sha256"]
        raw_train, raw_validation = validation_frames(mechanism, generator)
        references = {"train": reference_arrays(raw_train), "validation": reference_arrays(raw_validation)}
        base = dict(mechanism=mechanism, generator_seed=generator, training_seed=training)
        members = []
        for index in range(cfg.V3_ENSEMBLE_SIZE):
            directory = unit_path(job) / f"member-{index}"
            arrays, telemetry = read_cache(directory, member_identity(identity, index), references)
            members.append(arrays)
            member_rows, metrics = metric_rows(arrays)
            for split in SPLITS:
                key = "train_metrics" if split == "train" else "validation_metrics"
                for task in (*cfg.TASK_NAMES, "macro"):
                    for metric, value in metric_pair(metrics[split], task).items():
                        if not np.isclose(value, telemetry["best"][key][task][metric], rtol=0, atol=1e-10):
                            raise ValueError("Selected-checkpoint telemetry differs from raw prediction metrics")
            rows.extend(dict(base, estimator="member", member=index, **row) for row in member_rows)
            best = telemetry["best"]["epoch"]
            history = telemetry["history"]
            epochs.append(dict(base, member=index, selected_epoch=best, epochs_run=len(history),
                               epochs_after_selected=len(history) - best,
                               selected_validation_auc=telemetry["best"]["validation_metrics"]["macro"]["roc_auc"],
                               final_validation_auc=history[-1]["validation_metrics"]["macro"]["roc_auc"]))
            for record in history:
                for task in (*cfg.TASK_NAMES, "macro"):
                    dropout = metric_pair(record["training_metrics_dropout"], task)
                    validation = metric_pair(record["validation_metrics"], task)
                    curves.append(dict(base, member=index, epoch=record["epoch"], task=task,
                                       selected=record["epoch"] == best,
                                       training_auc_dropout=dropout["roc_auc"], training_brier_dropout=dropout["brier"],
                                       validation_auc=validation["roc_auc"], validation_brier=validation["brier"]))
            for name in (*ARTIFACTS, "complete.json"):
                path = directory / name
                provenance[str(path.relative_to(ROOT))] = sha256(path)
        ensemble_rows, _ = metric_rows(ensemble_arrays(members, references))
        rows.extend(dict(base, estimator="ensemble", member="", **row) for row in ensemble_rows)
        if diagnostic_identity(job) != identity:
            raise ValueError("Source/config/data changed during diagnostic scoring")
    frame = pd.DataFrame(rows)
    generators, summary = group_metrics(frame)
    epoch_frame = pd.DataFrame(epochs)
    epoch_generators = epoch_frame.groupby(["mechanism", "generator_seed"])[
        ["selected_epoch", "epochs_run", "epochs_after_selected"]].mean()
    epoch_summary = epoch_generators.groupby("mechanism").agg(["mean", "std", "min", "max"])
    epoch_summary.columns = ["_".join(pair) for pair in epoch_summary.columns]
    if any(sha256(project_path(path)) != digest for path, digest in provenance.items()):
        raise ValueError("Diagnostic input changed during scoring")
    root = project_path(cfg.V3_DIAGNOSTIC_REPORT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".score-", dir=root))
    try:
        frame.to_csv(staging / "unit-metrics.csv", index=False)
        generators.to_csv(staging / "generator-metrics.csv", index=False)
        pd.DataFrame(summary).to_csv(staging / "group-metrics.csv", index=False)
        pd.DataFrame(curves).to_csv(staging / "epoch-curves.csv", index=False)
        epoch_frame.to_csv(staging / "member-epochs.csv", index=False)
        epoch_summary.to_csv(staging / "group-epochs.csv")
        outputs = {path.name: sha256(path) for path in staging.iterdir()}
        (staging / "provenance.json").write_text(json.dumps({
            "scope": "DEVELOPMENT train/validation only; bounded legacy M5 diagnosis",
            "warning": "Gaps alone do not establish overfitting or justify a regularization change.",
            "jobs": [list(job) for job in jobs], "members": cfg.V3_ENSEMBLE_SIZE,
            "gap_convention": "AUC=train-validation; Brier=validation-train",
            "member_metrics": "eval-mode selected validation-macro-AUC checkpoint, raw probabilities",
            "ensemble": "mean raw probabilities of five individually selected member checkpoints",
            "curves": "training_*_dropout is online training-mode, not eval-mode train metrics",
            "aggregation": "mean member metrics within unit; mean unit metrics over training seeds; summarize generators",
            "mcse": "sample SD(ddof=1)/sqrt(n_generators); null for n=1",
            "runtime": identity["runtime"], "python": identity["python"],
            "input_sha256": provenance, "report_sha256": outputs,
        }, indent=2, allow_nan=False))
        destination = root / str(time.time_ns())
        staging.rename(destination)
        print(destination, flush=True)
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fit", "score"), default="score")
    args = parser.parse_args(argv)
    if cfg.V3_ENSEMBLE_SIZE != 5 or cfg.V3_DEV_WORKERS < 1:
        parser.error("Diagnosis requires configured M5 and positive bounded workers")
    jobs = list(itertools.product(cfg.V3_DEV_MECHANISMS, cfg.V3_DEV_GENERATOR_SEEDS,
                                  cfg.V3_DEV_TRAINING_SEEDS))
    if args.phase == "fit":
        run_fits(jobs)
    else:
        score(jobs)


if __name__ == "__main__":
    main()
