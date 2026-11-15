"""m2g_net_v3 development check on generator seeds 1701-1710, VALIDATION split only (observed labels).

Fits v3 (training seed 42) and compares with already-frozen units' validation_predictions.npz.
Never reads predictions.csv/oracle.csv of the test split. Writes outputs/paper-v3-dev and docs/paper/extension-v3/dev-validation.csv.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import compute_all_metrics

OUT = Path("outputs/paper-v3-dev")
MECHS = ("additive", "interaction", "partial")
SEEDS = range(1701, 1711)
COMPARE = ("m2g_net", "early_fusion_mlp", "catboost")


def data_dir(mech, seed):
    return Path("outputs/paper-final" if seed <= 1703 else "outputs/paper-extension") / "data" / f"{mech}-{seed}"


def frozen_unit(mech, seed, model):
    return Path("outputs/paper-final" if seed <= 1703 else "outputs/paper-extension") / "units" / f"{mech}-g{seed}-t42-{model}"


def fit(item):
    mech, seed = item
    out = OUT / f"{mech}-g{seed}-t42-m2g_net_v3"
    if (out / "random_seed_42/m2g_net_v3/complete.json").exists():
        return item, 0
    cmd = [".venv/bin/python", "scripts/evaluate/paper_pipeline.py", "--data", str(data_dir(mech, seed) / "data.csv"),
           "--require-masks", "--models", "m2g_net_v3", "--seeds", "42", "--split-seed", "42", "--split", "random",
           "--gate-priors", "0.1", "--max-epochs", "150", "--patience", "20", "--bootstrap", "0", "--output", str(out)]
    return item, subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT).returncode


def validation_scores(root):
    z = np.load(root / "validation_predictions.npz")
    macro = compute_all_metrics(z["targets"], z["probabilities"], z["masks"])["macro"]
    return macro["roc_auc"], macro["brier"]


def main():
    items = [(m, s) for m in MECHS for s in SEEDS]
    with ThreadPoolExecutor(4) as pool:
        for item, code in pool.map(fit, items):
            print(item, "ok" if not code else f"FAIL {code}", flush=True)
    rows = []
    for mech, seed in items:
        for model in ("m2g_net_v3", *COMPARE):
            root = (OUT / f"{mech}-g{seed}-t42-m2g_net_v3" if model == "m2g_net_v3" else frozen_unit(mech, seed, model)) / "random_seed_42" / model
            auc, brier = validation_scores(root)
            rows.append({"mechanism": mech, "generator_seed": seed, "model": model, "val_auc": auc, "val_brier": brier})
    frame = pd.DataFrame(rows)
    Path("docs/paper/extension-v3").mkdir(parents=True, exist_ok=True)
    frame.to_csv("docs/paper/extension-v3/dev-validation.csv", index=False)
    print(frame.groupby(["mechanism", "model"])[["val_auc", "val_brier"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
