"""Read-only DEVELOPMENT-validation analysis of cached M5 members (no fits, no locked/test rows).

A: ensemble-size curve of wd_base (mean AUC over all member subsets of size M).
B: heterogeneous pools built from cached gate/decay members, scored with the same
   dev screen as run_v3_gate_dev. Caches are read with the manifest sha256 and exact
   row/label/mask alignment but WITHOUT the source-identity check (gate caches predate
   the weight-decay source edit), so this is exploratory dev evidence only.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as cfg
from scripts import run_v3_gate_dev as r

POOL = ("wd_base", "nonlinear", "task_blend", "nonlinear_task_blend", "wd_strong")
OUT = ROOT / "docs/paper/extension-v3/ENSEMBLE-ANALYSIS"


def members(mechanism, seed, training, variant, expected):
    out = []
    for i in range(cfg.V3_ENSEMBLE_SIZE):
        d = r.unit_path((mechanism, seed, training, variant)) / f"member-{i}"
        manifest = json.loads((d / "complete.json").read_text())
        art = d / "validation_predictions.npz"
        assert manifest["prediction_sha256"] == r.sha256(art), d
        assert manifest["identity"]["job"] == [mechanism, seed, training, variant]
        with np.load(art, allow_pickle=False) as z:
            a = {k: z[k] for k in ("row_ids", "targets", "probabilities", "masks")}
        r.align_validation(a, expected)
        out.append(a["probabilities"])
    return out


def auc_brier(expected, probs):
    return r.macro_scores(dict(expected, probabilities=np.mean(probs, axis=0)))


def main():
    rows, curve = [], []
    seeds, trains = cfg.V3_DEV_GENERATOR_SEEDS, cfg.V3_DEV_TRAINING_SEEDS
    for mech, seed, tr in itertools.product(cfg.V3_DEV_MECHANISMS, seeds, trains):
        _, val = r.validation_frames(mech, seed)
        exp = r.reference_arrays(val)
        dh = r.sha256(r.dataset_path(mech, seed))
        for opp in cfg.V3_DEV_OPPONENTS:
            s, _ = r.frozen_validation(mech, seed, tr, opp, exp, dh)
            rows.append(dict(mechanism=mech, generator_seed=seed, training_seed=tr, model=opp, **s))
        mem = {v: members(mech, seed, tr, v, exp) for v in POOL}
        base = mem["wd_base"]
        for m in range(1, 6):
            vals = [auc_brier(exp, [base[i] for i in c])["auc"] for c in itertools.combinations(range(5), m)]
            curve.append(dict(mechanism=mech, generator_seed=seed, training_seed=tr, size=m,
                              family="homogeneous_wd_base", auc=float(np.mean(vals))))
        for k in range(1, 6):  # growing heterogeneous pool, 5k members
            pool = [p for v in POOL[:k] for p in mem[v]]
            curve.append(dict(mechanism=mech, generator_seed=seed, training_seed=tr, size=5 * k,
                              family="pool_prefix", auc=auc_brier(exp, pool)["auc"]))
        cand = {
            "H5_wd_base": base,
            "P25_pool": [p for v in POOL for p in mem[v]],
            # size-matched diversity test: one member per variant, mean over 5 fixed rotations
        }
        for name, probs in cand.items():
            rows.append(dict(mechanism=mech, generator_seed=seed, training_seed=tr,
                             model=f"gate:{name}", **auc_brier(exp, probs)))
        rot = [auc_brier(exp, [mem[v][(j + s) % 5] for j, v in enumerate(POOL)]) for s in range(5)]
        rows.append(dict(mechanism=mech, generator_seed=seed, training_seed=tr, model="gate:X5_one_per_variant",
                         auc=float(np.mean([x["auc"] for x in rot])), brier=float(np.mean([x["brier"] for x in rot]))))
    frame = pd.DataFrame(rows)
    means = frame.groupby(["mechanism", "generator_seed", "model"])[["auc", "brier"]].mean()
    names = sorted({m for m in frame.model if m.startswith("gate:")})
    summ, paired = [], []
    for name, mech, opp in itertools.product(names, cfg.V3_DEV_MECHANISMS, cfg.V3_DEV_OPPONENTS):
        d = {"auc_gain": [], "brier_gain": []}
        for s in seeds:
            c, b = means.loc[mech, s, name], means.loc[mech, s, opp]
            v = {"auc_gain": float(c.auc - b.auc), "brier_gain": float(b.brier - c.brier)}
            paired.append(dict(variant=name, mechanism=mech, generator_seed=s, opponent=opp, **v))
            for k, x in v.items():
                d[k].append(x)
        for k, x in d.items():
            summ.append(dict(variant=name, mechanism=mech, opponent=opp, metric=k, **r.delta_summary(x)))
    for name, opp in itertools.product(names, cfg.V3_DEV_OPPONENTS):
        vals = [np.mean([p["brier_gain"] for p in paired if p["variant"] == name and p["opponent"] == opp
                         and p["generator_seed"] == s]) for s in seeds]
        summ.append(dict(variant=name, mechanism="aggregate", opponent=opp, metric="brier_gain", **r.delta_summary(vals)))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summ).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(curve).to_csv(OUT / "curve-units.csv", index=False)
    cur = (pd.DataFrame(curve).groupby(["family", "mechanism", "size"]).auc.mean().unstack("size").round(5))
    cur.to_csv(OUT / "curve.csv")
    print(cur.to_string())
    for name in names:
        checks = r.screening([x for x in summ if x["variant"] == name])
        failed = [f"{c['mechanism']}/{c['opponent']}" for c in checks if not c["passed"]]
        print(f"\n{name}: pass={not failed} failed={failed}")
        for x in summ:
            if x["variant"] == name and x["metric"] == "auc_gain" and x["mechanism"] != "additive" \
                    and x["opponent"] in ("m2g_net", "shared_gate", "no_interaction", "early_fusion"):
                print(f"  {x['mechanism']:11s}{x['opponent']:15s} mean={x['mean']:+.4f} mcse={x['mcse']:.4f} pos={x['positive']}/10")


if __name__ == "__main__":
    main()
