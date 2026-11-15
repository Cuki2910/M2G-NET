"""Post-hoc extension: generator seeds 1704-1710 with the frozen procedures and scorer.

Read-only on the frozen 150-unit artifacts; writes only outputs/paper-extension and docs/paper/extension.
ponytail: runs fits as subprocesses with a thread pool; upgrade path is a job queue if >4 workers needed.
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
from scripts.run_paper_unit import read, verify_sources
from scripts.summarize_paper_results import METRICS, score_predictions

PLAN = Path("docs/paper/run-plan.json")
SEEDS = range(1704, 1711)
TRAINING = (42, 101)
OUT = Path("outputs/paper-extension")
DOC = Path("docs/paper/extension")


def units(plan):
    template = plan["units"][0]["command"]
    models = {}
    for unit in plan["units"]:
        models.setdefault(unit["mechanism"], [])
        if unit["model"] not in models[unit["mechanism"]]:
            models[unit["mechanism"]].append(unit["model"])
    result = []
    for mechanism, names in models.items():
        for seed in SEEDS:
            data = OUT / "data" / f"{mechanism}-{seed}"
            for training in TRAINING:
                for model in names:
                    uid = f"{mechanism}-g{seed}-t{training}-{model}"
                    out = OUT / "units" / uid
                    command = list(template)
                    command[command.index("--data") + 1] = str(data / "data.csv")
                    command[command.index("--models") + 1] = model
                    command[command.index("--seeds") + 1] = str(training)
                    command[command.index("--output") + 1] = str(out)
                    result.append({"id": uid, "mechanism": mechanism, "generator_seed": seed,
                                   "training_seed": training, "model": model, "data": data,
                                   "output": out, "command": command})
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
    done = item["output"] / f"random_seed_{item['training_seed']}" / item["model"] / "complete.json"
    if done.exists():
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
        evaluation = read(root / "evaluation.json")
        predictions = pd.read_csv(root / "predictions.csv")
        oracle = pd.read_csv(item["data"] / "oracle.csv")
        rows.extend(score_predictions(predictions, oracle, item, evaluation["split_rows"]["test"]))
    return pd.DataFrame(rows)


def summarize(frame, group_keys):
    rows = []
    for keys, group in frame.groupby(group_keys, sort=False):
        row = dict(zip(group_keys, keys))
        per = group.groupby("generator_seed")[list(METRICS)].mean(skipna=False)
        n = len(per)
        row["n_generator_replicates"] = n
        for metric in METRICS:
            v = per[metric]
            row.update({metric + "_mean": v.mean(), metric + "_sd": v.std(ddof=1), metric + "_min": v.min(),
                        metric + "_max": v.max(), metric + "_mcse": v.std(ddof=1) / np.sqrt(n),
                        metric + "_n_positive": int((v > 0).sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "score"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    plan = read(PLAN)
    verify_sources(plan)
    items = units(plan)
    if args.limit:
        items = items[:args.limit]
    if args.stage == "run":
        generate(items)
        failed = []
        with ThreadPoolExecutor(args.workers) as pool:
            for uid, seconds, code in pool.map(fit, items):
                print(uid, f"{seconds:.1f}s", "FAIL" if code else "ok", flush=True)
                if code:
                    failed.append(uid)
        verify_sources(plan)
        print("FAILED:", failed if failed else "none")
        return
    DOC.mkdir(parents=True, exist_ok=True)
    new = score(items)
    old = pd.read_csv("docs/paper/results.csv")
    cols = [c for c in old.columns if c in new.columns]
    allr = pd.concat([old[cols], new[cols]], ignore_index=True)
    full = allr.loc[allr.model.eq("m2g_net")]
    contrasts = []
    for model in allr.model.unique():
        if model == "m2g_net":
            continue
        pair = full.merge(allr.loc[allr.model.eq(model)], on=["mechanism", "generator_seed", "training_seed", "task"],
                          suffixes=("_f", "_c"), validate="one_to_one")
        part = pair[["mechanism", "generator_seed", "training_seed", "task"]].copy()
        part["model"] = model
        for metric in METRICS:
            part[metric] = pair[metric + "_f"] - pair[metric + "_c"]
        contrasts.append(part)
    delta = pd.concat(contrasts, ignore_index=True)
    outputs = {"results-new.csv": new, "results-all.csv": allr,
               "results-summary-10.csv": summarize(allr, ["mechanism", "model", "task"]),
               "paired-summary-10.csv": summarize(delta, ["mechanism", "model", "task"]),
               "paired-deltas-10.csv": delta}
    for name, frame in outputs.items():
        frame.to_csv(DOC / name, index=False, float_format="%.17g")
    verify_sources(plan)
    manifest = {"plan_sha256": sha256(PLAN), "scorer_sha256": sha256("scripts/summarize_paper_results.py"),
                "runner_sha256": sha256(__file__), "new_units": len(items),
                "outputs": {name: sha256(DOC / name) for name in outputs}}
    (DOC / "extension-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("scored", len(new), "rows; replicates per cell:", allr.groupby("mechanism").generator_seed.nunique().to_dict())


if __name__ == "__main__":
    main()
