"""Generate prespecified synthetic encounters; keep latent outcomes separate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg


FEATURE_LAW = {
    "rider_type": (["Delivery", "RideHailing", "Private"], [.30, .30, .40], [1, 0, -1]),
    "gender": (["Female", "Male"], [.40, .60], [-1, 1]),
    "age_group": (["Young", "Middle", "Older"], [.35, .45, .20], [1, 0, -1]),
    "police_presence": (["No", "Yes"], [.70, .30], [-1, 1]),
    "traffic_condition": (["Light", "Moderate", "Heavy"], [.25, .45, .30], [-1, 0, 1]),
    "weather": (["Dry", "Rain"], [.80, .20], [-1, 1]),
    "time_slot": (["OffPeak", "Peak"], [.50, .50], [-1, 1]),
    "weekend": (["No", "Yes"], [.70, .30], [-1, 1]),
}
COEFFICIENTS = np.array([
    [-.50, .35, .25, .10, .35, -.35, .10, .20, .10, .10, .10],
    [-.40, .25, .20, .10, .30, -.20, .15, .15, .05, .10, -.10],
    [-1.00, -.20, .25, .10, .10, -.40, .10, .05, .15, -.10, .10],
    [-.90, .30, .15, .10, .20, -.20, -.10, .20, .05, .10, -.10],
])
INTERACTIONS = np.array([[1.20, -.70, .50], [-1.00, .60, .80],
                         [.60, -1.00, -.50], [1.00, .50, -.80]])
OBSERVATION_OFFSETS = np.array([-.40, -.10, .20, .40])
MECHANISMS = ("additive", "interaction", "partial")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(seed, mechanism, rows=3000):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0 or rows % 12:
        raise ValueError("rows must be a positive multiple of 12")
    if mechanism not in MECHANISMS:
        raise ValueError("unknown mechanism")
    feature_rng, event_rng, outcome_rng, observation_rng = [
        np.random.Generator(np.random.PCG64(child))
        for child in np.random.SeedSequence(seed).spawn(4)
    ]
    sites = feature_rng.permutation(np.repeat(np.arange(12), rows // 12))
    data = pd.DataFrame({"simulation_row_id": np.arange(rows), cfg.SITE_ID_COL: sites})
    contrasts = {}
    for feature, (categories, probabilities, values) in FEATURE_LAW.items():
        indexes = feature_rng.choice(len(categories), rows, p=probabilities)
        data[feature] = np.array(categories)[indexes]
        contrasts[feature] = np.array(values)[indexes]
    data["has_signal"] = np.where(sites < 8, "Yes", "No")
    data["num_lanes"] = 2 + sites % 3
    data["intersection_type"] = np.where(sites % 2 == 0, "Cross", "T")
    data["red_phase_encounter"] = ((event_rng.random(rows) < .75) & (sites < 8)).astype(int)
    data["made_turn"] = (event_rng.random(rows) < .70).astype(int)
    eligible = np.column_stack([data.red_phase_encounter, data.made_turn,
                                np.ones(rows, dtype=int), np.ones(rows, dtype=int)])
    design = np.column_stack([
        np.ones(rows), contrasts["rider_type"], contrasts["age_group"],
        contrasts["gender"], contrasts["traffic_condition"], contrasts["police_presence"],
        contrasts["weather"], contrasts["time_slot"], contrasts["weekend"],
        data.num_lanes - 3, np.where(sites % 2 == 0, 1, -1),
    ])
    offsets = .25 * np.sin(2 * np.pi * (sites[:, None] + 1) * (np.arange(4) + 1) / 13)
    logits = design @ COEFFICIENTS.T + offsets
    if mechanism != "additive":
        products = np.column_stack([
            contrasts["rider_type"] * contrasts["traffic_condition"],
            contrasts["age_group"] * contrasts["police_presence"],
            contrasts["rider_type"] * contrasts["weather"],
        ])
        logits += products @ INTERACTIONS.T
    probabilities = 1 / (1 + np.exp(-logits))
    latent = (outcome_rng.random((rows, 4)) < probabilities).astype(int)
    propensities = np.ones((rows, 4))
    if mechanism == "partial":
        observation_logit = (OBSERVATION_OFFSETS +
                             (.50 * contrasts["rider_type"] - .40 * contrasts["traffic_condition"] +
                              .30 * contrasts["time_slot"] + .20 * contrasts["gender"])[:, None])
        propensities = .75 + .20 / (1 + np.exp(-observation_logit))
    observed = ((observation_rng.random((rows, 4)) < propensities) * eligible).astype(int)
    oracle = data[["simulation_row_id", cfg.SITE_ID_COL]].copy()
    for index, task in enumerate(cfg.TASK_NAMES):
        data[task + "_eligible"] = eligible[:, index]
        data[cfg.TASK_MASK_COLS[task]] = observed[:, index]
        data[task] = np.where(observed[:, index], latent[:, index], np.nan)
        oracle[task + "_eligible"] = eligible[:, index]
        oracle[cfg.TASK_MASK_COLS[task]] = observed[:, index]
        oracle[task + "_latent"] = np.where(eligible[:, index], latent[:, index], np.nan)
        oracle[task + "_probability"] = np.where(eligible[:, index], probabilities[:, index], np.nan)
        oracle[task + "_propensity"] = np.where(eligible[:, index], propensities[:, index], np.nan)
    validate(data, oracle, mechanism)
    return data, oracle


def validate(data, oracle, mechanism):
    if len(data) == 0 or len(data) % 12 or not data.simulation_row_id.equals(oracle.simulation_row_id):
        raise ValueError("invalid row identity")
    if not np.array_equal(data.simulation_row_id, np.arange(len(data))):
        raise ValueError("row IDs must be positional and unique")
    if not data[cfg.SITE_ID_COL].value_counts().reindex(range(12)).eq(len(data) // 12).all():
        raise ValueError("unbalanced sites")
    sites = data[cfg.SITE_ID_COL].to_numpy()
    if not (data.has_signal.eq(np.where(sites < 8, "Yes", "No")).all() and
            data.num_lanes.eq(2 + sites % 3).all() and
            data.intersection_type.eq(np.where(sites % 2 == 0, "Cross", "T")).all()):
        raise ValueError("site attributes changed")
    for feature, (categories, _, _) in FEATURE_LAW.items():
        if not data[feature].isin(categories).all():
            raise ValueError("invalid categorical feature: " + feature)
    for event in ("red_phase_encounter", "made_turn"):
        if not data[event].isin([0, 1]).all():
            raise ValueError("invalid event")
    if data.loc[data.has_signal.eq("No"), "red_phase_encounter"].any():
        raise ValueError("red encounter without signal")
    for index, task in enumerate(cfg.TASK_NAMES):
        eligibility = data[task + "_eligible"]
        mask = data[cfg.TASK_MASK_COLS[task]]
        expected = data.red_phase_encounter if index == 0 else data.made_turn if index == 1 else 1
        if not eligibility.eq(expected).all() or not mask.isin([0, 1]).all() or (mask > eligibility).any():
            raise ValueError("invalid eligibility/observation")
        if not oracle[task + "_eligible"].equals(eligibility) or not oracle[cfg.TASK_MASK_COLS[task]].equals(mask):
            raise ValueError("oracle mask mismatch")
        if not data.loc[mask.eq(1), task].isin([0, 1]).all() or data.loc[mask.eq(0), task].notna().any():
            raise ValueError("invalid or revealed target")
        latent = oracle[task + "_latent"]
        if not latent[eligibility.eq(1)].isin([0, 1]).all() or latent[eligibility.eq(0)].notna().any():
            raise ValueError("invalid latent outcome")
        if not data.loc[mask.eq(1), task].equals(latent[mask.eq(1)]):
            raise ValueError("observed target differs from oracle")
        for suffix in ("probability", "propensity"):
            values = oracle[task + "_" + suffix]
            selected = values[eligibility.eq(1)]
            if values[eligibility.eq(0)].notna().any() or not np.isfinite(selected).all():
                raise ValueError("invalid oracle probability")
            valid = selected.between(0, 1, inclusive="neither") if suffix == "probability" else (
                selected.between(.75, .95, inclusive="neither") if mechanism == "partial" else selected.eq(1))
            if not valid.all():
                raise ValueError("probability out of bounds")
        if mechanism != "partial" and not mask.equals(eligibility):
            raise ValueError("unexpected missing eligible outcome")
    if any(column.endswith(("_latent", "_probability", "_propensity")) for column in data):
        raise ValueError("oracle leakage")


def write_dataset(seed, mechanism, rows, output):
    output = Path(output)
    data, oracle = generate(seed, mechanism, rows)
    output.mkdir(parents=True, exist_ok=False)
    data.to_csv(output / "data.csv", index=False, float_format="%.17g")
    oracle.to_csv(output / "oracle.csv", index=False, float_format="%.17g")
    counts = {}
    for task in cfg.TASK_NAMES:
        labels = data.loc[data[cfg.TASK_MASK_COLS[task]].eq(1), task]
        counts[task] = {"eligible": int(data[task + "_eligible"].sum()), "observed": len(labels),
                        "positive": int(labels.sum()), "negative": int(len(labels) - labels.sum())}
    manifest = {
        "synthetic_only": True, "seed": seed, "mechanism": mechanism, "rows": rows, "sites": 12,
        "created_at": datetime.now(timezone.utc).isoformat(), "numpy_version": np.__version__,
        "pandas_version": pd.__version__, "rng": "PCG64; SeedSequence.spawn(4): feature,event,outcome,observation",
        "parameters": {"feature_law": FEATURE_LAW, "coefficients": COEFFICIENTS.tolist(),
                       "coefficient_order": ["intercept", "r", "a", "g", "t", "p", "w", "q", "d", "l", "c"],
                       "interactions": INTERACTIONS.tolist(), "observation_offsets": OBSERVATION_OFFSETS.tolist(),
                       "observation": {"base": .75, "range": .20, "r": .50, "t": -.40, "q": .30, "g": .20},
                       "site_offset": ".25*sin(2*pi*(s+1)*(k+1)/13)",
                       "red_encounter_probability": .75, "turn_probability": .70,
                       "site_design": {"has_signal": "s<8", "num_lanes": "2+s%3", "intersection_type": "Cross if s%2==0 else T"}},
        "task_counts": counts,
        "complete_case_count": int(data[list(cfg.TASK_MASK_COLS.values())].eq(1).all(axis=1).sum()),
        "event_rates": {event: float(data[event].mean()) for event in ("red_phase_encounter", "made_turn")},
        "source_hashes": {"generator": sha256(__file__), "spec": sha256(Path(__file__).resolve().parents[1] / "docs/paper/generator-spec.md")},
        "file_hashes": {name: sha256(output / name) for name in ("data.csv", "oracle.csv")},
        "validation": "passed: generator invariants; independent check_data validation remains separate",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mechanism", choices=MECHANISMS, required=True)
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = write_dataset(args.seed, args.mechanism, args.rows, args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Generation failed; output may be incomplete: {error}\n")
    print(json.dumps({"output": str(args.output), "counts": manifest["task_counts"]}))


if __name__ == "__main__":
    main()
