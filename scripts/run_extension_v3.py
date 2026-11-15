"""LOCKED TEST for m2g_net_v3: generator seeds 1711-1730, run ONCE after FREEZE.json is written and the user confirms.

Cloned from scripts/run_extension.py. Differences: seeds, model list (8 opponents + m2g_net_v3), sources verified against
docs/paper/extension-v3/FREEZE.json (not the v2 run-plan), pre-declared criteria computed in the score stage.
Writes only outputs/paper-v3-locked and docs/paper/extension-v3. Re-running with variants after seeing results is post hoc.
ponytail: subprocess thread pool; upgrade path is a job queue if >4 workers are needed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate_paper_simulation import sha256
from scripts.run_paper_unit import read
from scripts.summarize_paper_results import METRICS, score_predictions

PLAN = Path("docs/paper/run-plan.json")
FREEZE = Path("docs/paper/extension-v3/FREEZE.json")
SEEDS = range(1711, 1731)
TRAINING = (42, 101)
OUT = Path("outputs/paper-v3-locked")
DOC = Path("docs/paper/extension-v3")
NEW = "m2g_net_v3"
OPPONENTS = ("m2g_net", "shared_gate", "no_interaction", "early_fusion", "single_task_mlp",
             "early_fusion_mlp", "logistic_regression", "catboost")
MECHANISMS = ("additive", "interaction", "partial")
MIN_MEAN_DELTA, MIN_POSITIVE_SETS = 0.005, 18


def verify_freeze():
    for filename, expected in read(FREEZE)["sha256"].items():
        if sha256(filename) != expected:
            raise ValueError("Frozen v3 source changed: " + filename)


def units():
    template = read(PLAN)["units"][0]["command"]
    result = []
    for mechanism in MECHANISMS:
        for seed in SEEDS:
            data = OUT / "data" / f"{mechanism}-{seed}"
            for training in TRAINING:
                for model in (*OPPONENTS, NEW):
                    uid = f"{mechanism}-g{seed}-t{training}-{model}"
                    out = OUT / "units" / uid
                    command = list(template)
                    command[command.index("--data") + 1] = str(data / "data.csv")
                    command[command.index("--models") + 1] = model
                    command[command.index("--seeds") + 1] = str(training)
                    command[command.index("--output") + 1] = str(out)
                    result.append({"id": uid, "mechanism": mechanism, "generator_seed": seed, "training_seed": training,
                                   "model": model, "data": data, "output": out, "command": command})
    return result


def generate(items):
    for item in items:
        data = item["data"]
        if data.exists():
            continue
        subprocess.run([".venv/bin/python", "scripts/generate_paper_simulation.py", "--seed", str(item["generator_seed"]),
                        "--mechanism", item["mechanism"], "--rows", "3000", "--output", str(data)], check=True)
        subprocess.run([".venv/bin/python", "scripts/check_data.py", "--data", str(data / "data.csv"),
                        "--require-masks"], check=True, stdout=subprocess.DEVNULL)


def fit(item):
    if (item["output"] / f"random_seed_{item['training_seed']}" / item["model"] / "complete.json").exists():
        return item["id"], 0.0, 0
    log = DOC / "logs" / (item["id"] + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log.open("w") as stream:
        code = subprocess.run(item["command"], stdout=stream, stderr=subprocess.STDOUT).returncode
    return item["id"], time.perf_counter() - started, code


def score(items):
    rows = []
    for item in items:
        root = item["output"] / f"random_seed_{item['training_seed']}" / item["model"]
        predictions = pd.read_csv(root / "predictions.csv")
        oracle = pd.read_csv(item["data"] / "oracle.csv")
        rows.extend(score_predictions(predictions, oracle, item, read(root / "evaluation.json")["split_rows"]["test"]))
    return pd.DataFrame(rows)


def per_set(frame):
    """Macro metric per (mechanism, model, generator_seed), averaged over the two training seeds."""
    macro = frame.loc[frame.task.eq("macro")]
    if macro.groupby(["mechanism", "model", "generator_seed"]).size().ne(len(TRAINING)).any():
        raise ValueError("Each (mechanism, model, set) needs exactly both training seeds")
    return macro.groupby(["mechanism", "model", "generator_seed"], as_index=False)[list(METRICS)].mean()


def describe(values):
    values = np.asarray(values, dtype=float)
    return {"n_sets": len(values), "mean": values.mean(), "sd": values.std(ddof=1), "mcse": values.std(ddof=1) / np.sqrt(len(values)),
            "min": values.min(), "max": values.max(), "n_positive": int((values > 0).sum())}


def criteria(sets):
    wide = {metric: sets.pivot_table(index=["mechanism", "generator_seed"], columns="model", values=metric) for metric in METRICS}
    rows = []
    for mechanism in MECHANISMS:
        auc = wide["roc_auc"].loc[mechanism]
        brier = wide["brier"].loc[mechanism]
        for opponent in OPPONENTS:
            delta = describe(auc[NEW] - auc[opponent])
            if mechanism == "additive":
                ok = None if opponent not in ("catboost", "m2g_net") else delta["mean"] >= 0
                rule = "report only; must not lose on average to catboost and m2g_net"
            else:
                ok = bool(delta["n_positive"] >= MIN_POSITIVE_SETS and delta["mean"] > MIN_MEAN_DELTA)
                rule = f"n_positive>={MIN_POSITIVE_SETS}/20 and mean>{MIN_MEAN_DELTA}"
            rows.append({"criterion": "macro_auc", "mechanism": mechanism, "opponent": opponent, "rule": rule,
                         **{"delta_" + key: value for key, value in delta.items()}, "pass": ok})
        for opponent in ("early_fusion_mlp", "catboost"):
            delta = describe(brier[NEW] - brier[opponent])  # negative = v3 better
            rows.append({"criterion": "brier", "mechanism": mechanism, "opponent": opponent,
                         "rule": "mean(v3 - opponent) <= 0 per mechanism",
                         **{"delta_" + key: value for key, value in delta.items()}, "pass": bool(delta["mean"] <= 0)})
    brier = wide["brier"]
    for opponent in ("early_fusion_mlp", "catboost"):  # protocol wording: average across the 3 mechanisms
        diff = (brier[NEW] - brier[opponent]).groupby(level="generator_seed").mean()
        delta = describe(diff)
        rows.append({"criterion": "brier", "mechanism": "aggregate", "opponent": opponent,
                     "rule": "mean over mechanisms of (v3 - opponent) <= 0 (PROTOCOL CRITERION)",
                     **{"delta_" + key: value for key, value in delta.items()}, "pass": bool(delta["mean"] <= 0)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "score"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--confirmed-locked-test", action="store_true",
                        help="Required for the run stage; user must have confirmed after reading FREEZE.json.")
    args = parser.parse_args()
    verify_freeze()
    items = units()
    if args.stage == "run":
        if not args.confirmed_locked_test:
            parser.error("run stage generates LOCKED TEST data; pass --confirmed-locked-test only after user confirmation")
        generate(items)
        failed = []
        with ThreadPoolExecutor(args.workers) as pool:
            for uid, seconds, code in pool.map(fit, items):
                print(uid, f"{seconds:.1f}s", "FAIL" if code else "ok", flush=True)
                if code:
                    failed.append(uid)
        verify_freeze()
        print("FAILED:", failed if failed else "none")
        return
    DOC.mkdir(parents=True, exist_ok=True)
    results = score(items)
    sets = per_set(results)
    table = criteria(sets)
    summary = sets.groupby(["mechanism", "model"])[list(METRICS)].agg(["mean", "std", "min", "max"])
    outputs = {"results-locked.csv": results, "per-set-locked.csv": sets, "criteria.csv": table}
    for name, frame in outputs.items():
        frame.to_csv(DOC / name, index=False, float_format="%.17g")
    summary.to_csv(DOC / "summary-locked.csv", float_format="%.17g")
    verify_freeze()
    json.dump({"freeze_sha256": sha256(FREEZE), "scorer_sha256": sha256("scripts/summarize_paper_results.py"),
               "runner_sha256": sha256(__file__), "units": len(items),
               "outputs": {name: sha256(DOC / name) for name in (*outputs, "summary-locked.csv")}},
              open(DOC / "locked-manifest.json", "w"), indent=2)
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
