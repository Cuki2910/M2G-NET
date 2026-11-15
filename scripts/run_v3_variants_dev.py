"""m2g_net_v3 variant screening on DEVELOPMENT seeds 1701-1710, VALIDATION split only.

Fits single BCE members once (cached as npz of validation probabilities) and builds ensemble variants by averaging.
Variants declared before running: M3, M5 (current v3), M10 (size), MIX5 (gate prior cycled 0.0/0.1/0.2/0.0/0.1).
Never reads the test split or LOCKED seeds. Writes docs/paper/extension-v3/dev-variants.csv.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
from scripts.run_v3_dev import MECHS, SEEDS, COMPARE, data_dir, frozen_unit, validation_scores
from src.metrics import compute_all_metrics

CACHE = Path("outputs/paper-v3-dev-variants")
TRAIN_SEED = 42
MIX_PRIORS = (0.0, 0.1, 0.2, 0.0, 0.1)
VARIANTS = {  # name -> [(member index, gate prior)]; prior "s" tag = interaction skip on
    "M3": [(i, 0.1) for i in range(3)],
    "M5": [(i, 0.1) for i in range(5)],
    "M10": [(i, 0.1) for i in range(10)],
    "MIX5": list(enumerate(MIX_PRIORS)),
}
if len(sys.argv) > 1 and sys.argv[1] == "skip":
    VARIANTS = {"SKIP5": [(i, "skip") for i in range(5)], "M5": VARIANTS["M5"]}


def member_path(mech, seed, index, prior):
    return CACHE / f"{mech}-g{seed}-m{index}-p{prior}.npz"


def fit_member(job):
    import torch
    from scripts.train import collect_predictions, fit_model_for_split
    from src.data_pipeline import encode_splits, get_loaders, prepare_labels, stratified_split
    mech, seed, index, prior = job
    skip = prior == "skip"
    prior = 0.1 if skip else prior
    path = member_path(mech, seed, index, "skip" if skip else prior)
    if path.exists():
        return job
    torch.set_num_threads(1)
    raw = pd.read_csv(data_dir(mech, seed) / "data.csv")
    raw["_row_id"] = np.arange(len(raw))
    train, validation, test, encoders, vocab = encode_splits(*stratified_split(prepare_labels(raw, cfg.LABEL_POLICY, True), seed=42))
    model, _, _, _ = fit_model_for_split(
        vocab, train, validation, encoders=encoders, split_seed=TRAIN_SEED + index * cfg.V3_MEMBER_SEED_OFFSET,
        ablation="full", gate_prior_weight=prior, use_focal=cfg.V3_USE_FOCAL_LOSS, interaction_skip=skip, validation_use_site_intercept=True)
    _, val_loader, _ = get_loaders(train, validation, validation, vocab)
    y, probs, masks = collect_predictions(model, val_loader, True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, targets=y, probabilities=probs, masks=masks)
    return job


def main():
    jobs = sorted({(m, s, i, p) for m in MECHS for s in SEEDS for v in VARIANTS.values() for i, p in v}, key=str)
    with ProcessPoolExecutor(6) as pool:
        for n, job in enumerate(pool.map(fit_member, jobs), 1):
            print(n, len(jobs), job, flush=True)
    rows = []
    for mech in MECHS:
        for seed in SEEDS:
            for name, members in VARIANTS.items():
                zs = [np.load(member_path(mech, seed, i, p)) for i, p in members]
                macro = compute_all_metrics(zs[0]["targets"], np.mean([z["probabilities"] for z in zs], 0), zs[0]["masks"])["macro"]
                rows.append({"mechanism": mech, "generator_seed": seed, "model": name, "val_auc": macro["roc_auc"], "val_brier": macro["brier"]})
            for model in COMPARE:
                auc, brier = validation_scores(frozen_unit(mech, seed, model) / "random_seed_42" / model)
                rows.append({"mechanism": mech, "generator_seed": seed, "model": model, "val_auc": auc, "val_brier": brier})
    frame = pd.DataFrame(rows)
    frame.to_csv("docs/paper/extension-v3/dev-variants-skip.csv" if "SKIP5" in VARIANTS else "docs/paper/extension-v3/dev-variants.csv", index=False)
    wide = frame.pivot_table(index=["mechanism", "generator_seed"], columns="model", values="val_auc")
    for name in VARIANTS:
        for opp in COMPARE + (("M5",) if name == "SKIP5" else ()):
            d = (wide[name] - wide[opp]).groupby("mechanism")
            print(f"{name} - {opp}: " + "  ".join(f"{m}: {g.mean():+.4f} ({(g > 0).sum()}/10)" for m, g in d))
    print(frame.groupby(["mechanism", "model"])[["val_auc", "val_brier"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
