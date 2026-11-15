"""DEVELOPMENT validation only; never score held-out rows or locked generators.

Default is score-only. Fit requires a separately successful project preflight.
Examples: --phase fit --variants legacy nonlinear; --phase score --members 3.
All subsets are exploratory; no development result establishes held-out success.
"""
import argparse
from collections import deque
import hashlib
import fcntl
from importlib.metadata import version
import itertools
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as cfg


def project_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path outside project: {path}")
    return path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_root(generator_seed):
    if generator_seed not in cfg.V3_DEV_GENERATOR_SEEDS:
        raise ValueError("Only configured DEVELOPMENT generator seeds are allowed")
    return project_path(cfg.V3_DEV_FINAL_DATA_ROOT if generator_seed <= cfg.V3_DEV_FINAL_LAST_SEED
                        else cfg.V3_DEV_EXTENSION_DATA_ROOT)


def dataset_path(mechanism, generator_seed):
    if mechanism not in cfg.V3_DEV_MECHANISMS:
        raise ValueError("Unknown development mechanism")
    return data_root(generator_seed) / "data" / f"{mechanism}-{generator_seed}" / "data.csv"


def training_options(variant):
    if variant in cfg.V3_DECAY_VARIANTS:
        decay = cfg.V3_DECAY_VARIANTS[variant]
        if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not math.isfinite(decay) or decay < 0:
            raise ValueError("Configured weight decay must be finite and nonnegative")
        return {"gate_variant": "legacy", "weight_decay": decay}
    if variant not in cfg.V3_GATE_VARIANTS:
        raise ValueError("Unconfigured development variant")
    return {"gate_variant": variant, "weight_decay": cfg.WEIGHT_DECAY}


def identity_for(job):
    mechanism, generator_seed, training_seed, variant = job
    options = training_options(variant)
    if training_seed not in cfg.V3_DEV_TRAINING_SEEDS:
        raise ValueError("Unconfigured training seed")
    sources = sorted((ROOT / "src").glob("*.py")) + [
        ROOT / "config.py", ROOT / "scripts/train.py", Path(__file__), ROOT / "requirements.txt"]
    return {
        "job": list(job), "split_seed": cfg.V3_DEV_SPLIT_SEED,
        "dataset_sha256": sha256(dataset_path(mechanism, generator_seed)),
        "sources": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "runtime": {name: version(name) for name in ("numpy", "pandas", "torch", "scikit-learn")},
        "python": sys.version, "use_focal": False, "interaction_skip": False,
        "member_seed_offset": cfg.V3_MEMBER_SEED_OFFSET, "training_options": options,
    }


def unit_path(job):
    mechanism, generator_seed, training_seed, variant = job
    dataset_path(mechanism, generator_seed)  # Validate before forming output paths.
    training_options(variant)
    if training_seed not in cfg.V3_DEV_TRAINING_SEEDS:
        raise ValueError("Unconfigured training seed")
    return project_path(cfg.V3_DEV_ROOT) / f"{mechanism}-g{generator_seed}-t{training_seed}-{variant}"


def validate_arrays(arrays):
    rows, y, p, masks = (arrays[key] for key in ("row_ids", "targets", "probabilities", "masks"))
    if rows.ndim != 1 or rows.dtype.kind not in "iu" or len(np.unique(rows)) != len(rows):
        raise ValueError("Validation row IDs must be unique integers")
    if y.shape != (len(rows), len(cfg.TASK_NAMES)) or not len(rows) or p.shape != y.shape or masks.shape != y.shape:
        raise ValueError("Validation prediction dimensions differ")
    if not all(np.isfinite(value).all() for value in (y, p, masks)):
        raise ValueError("Validation arrays must be finite")
    if not np.isin(y, [0, 1]).all() or not np.isin(masks, [0, 1]).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid labels, observed masks, or raw probabilities")
    for index, task in enumerate(cfg.TASK_NAMES):
        if len(np.unique(y[masks[:, index].astype(bool), index])) != 2:
            raise ValueError(f"Missing observed binary coverage for {task}")


def align_validation(candidate, reference):
    """Require exact ordered IDs, labels and masks; never silently intersect rows."""
    validate_arrays(candidate)
    validate_arrays(reference)
    for key in ("row_ids", "targets", "masks"):
        if not np.array_equal(candidate[key], reference[key]):
            raise ValueError(f"Validation alignment mismatch: {key}")


def read_cache(directory, expected_identity):
    manifest = json.loads((directory / "complete.json").read_text())
    if manifest.get("identity") != expected_identity:
        raise ValueError(f"Stale cache identity: {directory}")
    artifact = directory / "validation_predictions.npz"
    if manifest.get("prediction_sha256") != sha256(artifact):
        raise ValueError(f"Cache artifact changed: {directory}")
    with np.load(artifact, allow_pickle=False) as data:
        arrays = {key: data[key] for key in ("row_ids", "targets", "probabilities", "masks")}
    validate_arrays(arrays)
    return arrays


def write_cache(directory, identity, arrays):
    validate_arrays(arrays)
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".member-", dir=directory.parent))
    try:
        artifact = staging / "validation_predictions.npz"
        np.savez_compressed(artifact, **arrays)
        (staging / "complete.json").write_text(json.dumps({
            "identity": identity, "prediction_sha256": sha256(artifact)}, sort_keys=True, indent=2))
        staging.rename(directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def validation_frames(mechanism, generator_seed):
    import pandas as pd
    from src.data_pipeline import prepare_labels, stratified_split

    raw = pd.read_csv(dataset_path(mechanism, generator_seed))
    raw["_row_id"] = np.arange(len(raw))
    train, validation, _ = stratified_split(
        prepare_labels(raw, cfg.LABEL_POLICY, require_masks=True), seed=cfg.V3_DEV_SPLIT_SEED)
    del _  # Discard holdout without inspecting or encoding it.
    return train, validation


def reference_arrays(validation):
    from src.data_pipeline import extract_targets, extract_task_masks

    y = extract_targets(validation).numpy()
    return {"row_ids": validation["_row_id"].to_numpy(), "targets": y,
            "masks": extract_task_masks(validation).numpy(), "probabilities": np.zeros_like(y)}


def fit_unit(job, members):
    import torch
    from scripts.train import collect_predictions, fit_model_for_split
    from src.data_pipeline import encode_splits, get_loaders

    torch.set_num_threads(1)
    directory = unit_path(job)
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / ".fit.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = identity_for(job)
        train_raw, val_raw = validation_frames(*job[:2])
        expected = reference_arrays(val_raw)
        train, validation, _, encoders, vocab = encode_splits(train_raw, val_raw, val_raw)
        del _
        _, loader, _ = get_loaders(train, validation, validation, vocab)
        for index in range(members):
            member_identity = dict(identity, member=index,
                                   member_seed=job[2] + index * cfg.V3_MEMBER_SEED_OFFSET)
            member_dir = directory / f"member-{index}"
            if member_dir.exists():
                align_validation(read_cache(member_dir, member_identity), expected)
                print(f"cached {job} member={index}", flush=True)
                continue
            model, _, _, _ = fit_model_for_split(
                vocab, train, validation, encoders=encoders,
                split_seed=member_identity["member_seed"], data_path=str(dataset_path(*job[:2])),
                gate_prior_weight=cfg.GATE_PRIOR_WEIGHT,
                use_focal=False, interaction_skip=False, validation_use_site_intercept=True,
                **training_options(job[3]))
            y, probabilities, masks = collect_predictions(model, loader, use_site_intercept=True)
            arrays = dict(row_ids=validation["_row_id"].to_numpy(), targets=y,
                          probabilities=probabilities, masks=masks)
            align_validation(arrays, expected)
            if identity_for(job) != identity:
                raise ValueError("Source/config/dataset changed during fitting; refusing cache publication")
            write_cache(member_dir, member_identity, arrays)
            print(f"fitted {job} member={index}", flush=True)
            del model
    finally:
        lock.close()  # OS releases lock even after an interrupted worker.


def run_fits(jobs, args):
    """Bounded OS subprocesses; no shared torch/RNG state or training threads."""
    pending, active = deque(jobs), []
    try:
        while pending or active:
            while pending and len(active) < cfg.V3_DEV_WORKERS:
                mechanism, seed, training, variant = pending.popleft()
                command = [sys.executable, str(Path(__file__)), "--phase", "fit", "--worker",
                           "--mechanism", mechanism, "--generator-seeds", str(seed),
                           "--training-seeds", str(training), "--variants", variant,
                           "--members", str(args.members)]
                active.append((command, subprocess.Popen(command, cwd=ROOT)))
            for command, process in list(active):
                code = process.poll()
                if code is None:
                    continue
                active.remove((command, process))
                if code:
                    raise RuntimeError(f"Fit subprocess failed ({code}): {command}")
            if active:
                time.sleep(0.1)
    finally:
        for _, process in active:
            if process.poll() is None:
                process.terminate()
        for _, process in active:
            process.wait()


def macro_scores(arrays):
    from src.metrics import compute_all_metrics

    validate_arrays(arrays)
    macro = compute_all_metrics(arrays["targets"], arrays["probabilities"], arrays["masks"])["macro"]
    scores = {"auc": float(macro["roc_auc"]), "brier": float(macro["brier"])}
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("Nonfinite validation score")
    return scores


def frozen_validation(mechanism, generator_seed, training_seed, opponent, expected, data_hash):
    directory = (data_root(generator_seed) / "units" /
                 f"{mechanism}-g{generator_seed}-t{training_seed}-{opponent}" /
                 f"random_seed_{training_seed}" / opponent)
    evaluation = directory / "evaluation.json"
    artifact = directory / "validation_predictions.npz"
    completion = directory / "complete.json"
    complete = json.loads(completion.read_text())
    if any(complete.get(key) != value for key, value in
           (("model", opponent), ("seed", training_seed), ("fold", "random"))):
        raise ValueError(f"Frozen completion identity mismatch: {directory}")
    hashes = complete.get("artifact_sha256", {})
    for path in (artifact, evaluation):
        if hashes.get(path.name) != sha256(path):
            raise ValueError(f"Frozen validation completion hash mismatch: {path}")
    metadata = json.loads(evaluation.read_text())
    if metadata.get("dataset_sha256") != data_hash:
        raise ValueError(f"Frozen validation dataset hash mismatch: {directory}")
    if (metadata.get("seed") != training_seed or metadata.get("model") != opponent
            or metadata.get("fold") != "random"):
        raise ValueError(f"Frozen validation identity mismatch: {directory}")
    with np.load(artifact, allow_pickle=False) as data:
        arrays = {key: data[key] for key in ("targets", "probabilities", "masks")}
        arrays["row_ids"] = np.asarray(metadata["split_rows"]["validation"])
        if "row_ids" in data and not np.array_equal(data["row_ids"], arrays["row_ids"]):
            raise ValueError(f"Frozen artifact row IDs disagree: {directory}")
    align_validation(arrays, expected)
    return macro_scores(arrays), {str(path.relative_to(ROOT)): sha256(path)
                                 for path in (artifact, evaluation, completion)}


def delta_summary(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Incomplete paired delta coverage")
    return {"n": len(values), "mean": float(values.mean()),
            "mcse": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
            "min": float(values.min()), "max": float(values.max()),
            "positive": int((values > 0).sum())}


def screening(summary):
    """Positive deltas always favor candidate (AUC candidate-old; Brier old-candidate)."""
    lookup = {(row["mechanism"], row["opponent"], row["metric"]): row for row in summary}
    checks = []
    for mechanism in cfg.V3_DEV_MECHANISMS:
        for opponent in cfg.V3_DEV_OPPONENTS:
            auc = lookup[mechanism, opponent, "auc_gain"]
            if mechanism != "additive":
                passed = (auc["positive"] >= math.ceil(cfg.V3_DEV_AUC_POSITIVE_FRACTION * auc["n"])
                          and auc["mean"] > cfg.V3_DEV_AUC_MIN_DELTA)
                checks.append(dict(mechanism=mechanism, opponent=opponent, rule="auc_superiority", passed=passed))
            elif opponent in ("catboost", "m2g_net"):
                checks.append(dict(mechanism=mechanism, opponent=opponent,
                                   rule="auc_noninferiority", passed=auc["mean"] >= 0))
    for opponent in ("catboost", "early_fusion_mlp"):
        brier = lookup["aggregate", opponent, "brier_gain"]
        checks.append(dict(mechanism="aggregate", opponent=opponent,
                           rule="brier_noninferiority", passed=brier["mean"] >= 0))
    return checks


def score(jobs, args):
    import pandas as pd

    initial_sources = identity_for(jobs[0])["sources"]
    rows, provenance = [], dict(initial_sources)
    for mechanism, seed, training in itertools.product(
            cfg.V3_DEV_MECHANISMS, args.generator_seeds, args.training_seeds):
        _, validation = validation_frames(mechanism, seed)
        expected = reference_arrays(validation)
        data_path = dataset_path(mechanism, seed)
        data_hash = sha256(data_path)
        provenance[str(data_path.relative_to(ROOT))] = data_hash
        for opponent in cfg.V3_DEV_OPPONENTS:
            scores, hashes = frozen_validation(mechanism, seed, training, opponent, expected, data_hash)
            provenance.update(hashes)
            rows.append(dict(mechanism=mechanism, generator_seed=seed, training_seed=training,
                             model=opponent, **scores))
        for variant in args.variants:
            job = (mechanism, seed, training, variant)
            identity = identity_for(job)
            members = []
            for index in range(args.members):
                directory = unit_path(job) / f"member-{index}"
                member_identity = dict(identity, member=index,
                                       member_seed=training + index * cfg.V3_MEMBER_SEED_OFFSET)
                arrays = read_cache(directory, member_identity)
                align_validation(arrays, expected)
                members.append(arrays["probabilities"])
                for name in ("complete.json", "validation_predictions.npz"):
                    path = directory / name
                    provenance[str(path.relative_to(ROOT))] = sha256(path)
            ensemble = dict(expected, probabilities=np.mean(members, axis=0))
            rows.append(dict(mechanism=mechanism, generator_seed=seed, training_seed=training,
                             model=f"gate:{variant}", **macro_scores(ensemble)))
            if identity_for(job) != identity:
                raise ValueError("Source/config/dataset changed during scoring")
    frame = pd.DataFrame(rows)
    # Average metrics over training seeds within each generated set, not probabilities.
    means = frame.groupby(["mechanism", "generator_seed", "model"])[["auc", "brier"]].mean()
    paired, summaries, criteria = [], [], []
    for variant, mechanism, opponent in itertools.product(args.variants, cfg.V3_DEV_MECHANISMS, cfg.V3_DEV_OPPONENTS):
        deltas = {"auc_gain": [], "brier_gain": []}
        for seed in args.generator_seeds:
            candidate, baseline = means.loc[mechanism, seed, f"gate:{variant}"], means.loc[mechanism, seed, opponent]
            values = {"auc_gain": float(candidate.auc - baseline.auc),
                      "brier_gain": float(baseline.brier - candidate.brier)}
            paired.append(dict(variant=variant, mechanism=mechanism, generator_seed=seed, opponent=opponent, **values))
            for metric, value in values.items():
                deltas[metric].append(value)
        for metric, values in deltas.items():
            summaries.append(dict(variant=variant, mechanism=mechanism, opponent=opponent,
                                  metric=metric, **delta_summary(values)))
    # Aggregate Brier across mechanisms within each generator, preserving paired sets.
    for variant, opponent in itertools.product(args.variants, cfg.V3_DEV_OPPONENTS):
        values = []
        for seed in args.generator_seeds:
            values.append(float(np.mean([
                row["brier_gain"] for row in paired
                if row["variant"] == variant and row["opponent"] == opponent
                and row["generator_seed"] == seed])))
        summaries.append(dict(variant=variant, mechanism="aggregate", opponent=opponent,
                              metric="brier_gain", **delta_summary(values)))
    full = (set(args.generator_seeds) == set(cfg.V3_DEV_GENERATOR_SEEDS)
            and set(args.training_seeds) == set(cfg.V3_DEV_TRAINING_SEEDS)
            and args.members == cfg.V3_ENSEMBLE_SIZE)
    for variant in args.variants:
        checks = screening([row for row in summaries if row["variant"] == variant])
        criteria.append(dict(variant=variant, checks=checks,
                             passes_selected_subset=all(row["passed"] for row in checks),
                             passes_full_dev_screen=full and all(row["passed"] for row in checks)))
    # Check inputs again before publishing one complete, atomic report directory.
    if any(sha256(project_path(path)) != digest for path, digest in provenance.items()):
        raise ValueError("Prediction artifacts changed during scoring")
    report_root = project_path(cfg.V3_DEV_REPORT_ROOT)
    report_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".score-", dir=report_root))
    try:
        frame.to_csv(staging / "validation-units.csv", index=False)
        pd.DataFrame(paired).to_csv(staging / "paired-deltas.csv", index=False)
        pd.DataFrame(summaries).to_csv(staging / "summary.csv", index=False)
        (staging / "screening.json").write_text(json.dumps({
            "scope": "DEVELOPMENT VALIDATION ONLY; not held-out confirmation",
            "exploratory_subset": not full, "generator_seeds": args.generator_seeds,
            "training_seeds": args.training_seeds, "variants": args.variants, "members": args.members,
            "delta_convention": "positive favors candidate for both auc_gain and brier_gain",
            "mcse": "sample SD(ddof=1)/sqrt(n); undefined (null) for one generator",
            "criteria": criteria, "input_sha256": provenance,
        }, indent=2, allow_nan=False))
        destination = report_root / f"score-{time.time_ns()}"
        staging.rename(destination)
        print(destination, flush=True)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fit", "score"), default="score")
    parser.add_argument("--variants", nargs="+", choices=(*cfg.V3_GATE_VARIANTS, *cfg.V3_DECAY_VARIANTS),
                        default=list(cfg.V3_GATE_VARIANTS))
    parser.add_argument("--generator-seeds", nargs="+", type=int, choices=cfg.V3_DEV_GENERATOR_SEEDS,
                        default=list(cfg.V3_DEV_GENERATOR_SEEDS))
    parser.add_argument("--training-seeds", nargs="+", type=int, choices=cfg.V3_DEV_TRAINING_SEEDS,
                        default=list(cfg.V3_DEV_TRAINING_SEEDS))
    parser.add_argument("--members", type=int, default=cfg.V3_ENSEMBLE_SIZE)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mechanism", choices=cfg.V3_DEV_MECHANISMS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for key in ("variants", "generator_seeds", "training_seeds"):
        if len(getattr(args, key)) != len(set(getattr(args, key))):
            parser.error(f"Duplicate --{key.replace('_', '-')} values")
    if not 1 <= args.members <= cfg.V3_ENSEMBLE_SIZE:
        parser.error("--members must be a positive prefix no larger than configured ensemble size")
    if cfg.V3_DEV_WORKERS < 1:
        parser.error("V3_DEV_WORKERS must be positive")
    if args.worker and (args.phase != "fit" or args.mechanism is None or
                        any(len(getattr(args, key)) != 1 for key in ("variants", "generator_seeds", "training_seeds"))):
        parser.error("Worker requires exactly one fit unit")
    if args.mechanism is not None and not args.worker:
        parser.error("--mechanism is internal to subprocess workers")
    return args


def main(argv=None):
    args = parse_args(argv)
    mechanisms = [args.mechanism] if args.worker else cfg.V3_DEV_MECHANISMS
    jobs = list(itertools.product(mechanisms, args.generator_seeds, args.training_seeds, args.variants))
    if args.worker:
        fit_unit(jobs[0], args.members)
    elif args.phase == "fit":
        run_fits(jobs, args)
    else:
        score(jobs, args)


if __name__ == "__main__":
    main()
