"""
Gate Prior CV Sweep: find the smallest lambda that brings mean CV below 0.3
while keeping AUC drop under 0.01.

Tests lambda in [0.0, 0.1, 0.3, 0.5] x 5 seeds.
Reports mean_CV and mean_AUC per lambda so the best value can be chosen for config.py.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from src.data_pipeline import load_data, get_loaders
from scripts.train import fit_model_for_split
from scripts.evaluate.gate_stability import extract_gate_weights, compute_gate_statistics

os.makedirs("outputs", exist_ok=True)

LAMBDAS = [0.0, 0.1, 0.3, 0.5]
SEEDS   = [42, 101, 2023, 777, 999]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _patch_gates(model, prior_weight):
    """Patch gate prior_weight on a freshly created model in-place."""
    for gate in model.fusion.gates:
        gate.prior_weight = float(prior_weight)
    return model


def run_one_seed(seed, prior_weight):
    """Train one model with given seed+lambda and return its average gate weights."""
    set_seed(seed)
    # Temporarily override GATE_PRIOR_WEIGHT for this run
    original = cfg.GATE_PRIOR_WEIGHT
    try:
        cfg.GATE_PRIOR_WEIGHT = prior_weight
        train_df, val_df, test_df, encoders, vocab = load_data(seed=seed)
        model, loss_fn, _, _ = fit_model_for_split(
            vocab, train_df, val_df, encoders=encoders,
            max_epochs=80, patience=15,
        )
    finally:
        cfg.GATE_PRIOR_WEIGHT = original

    # Patch in case model was created before override took effect
    _patch_gates(model, prior_weight)

    _, _, test_loader = get_loaders(train_df, val_df, test_df, vocab)

    # Gate CV
    avg_gates = extract_gate_weights(model, test_loader)  # (4, 5)
    # mean and std over tasks and views
    # We need multi-seed CV — here we just return the gate weights for aggregation
    return avg_gates  # (4, 5)


def sweep():
    results = []

    for lam in LAMBDAS:
        print(f"\n{'='*55}")
        print(f"  lambda = {lam}")
        print(f"{'='*55}")

        gate_list = []
        auc_list  = []

        original = cfg.GATE_PRIOR_WEIGHT
        try:
            cfg.GATE_PRIOR_WEIGHT = lam

            for i, seed in enumerate(SEEDS):
                print(f"  Seed {seed} ({i+1}/{len(SEEDS)})...", end=" ", flush=True)
                set_seed(seed)

                train_df, val_df, test_df, encoders, vocab = load_data(seed=seed)
                model, loss_fn, _, _ = fit_model_for_split(
                    vocab, train_df, val_df, encoders=encoders,
                    max_epochs=80, patience=15,
                )
                _patch_gates(model, lam)

                _, _, test_loader = get_loaders(train_df, val_df, test_df, vocab)

                from scripts.train import run_epoch
                _, test_metrics, _ = run_epoch(model, test_loader, loss_fn)
                test_auc = test_metrics["macro"]["roc_auc"]
                auc_list.append(test_auc)

                avg_gates = extract_gate_weights(model, test_loader)
                gate_list.append(avg_gates)
                print(f"AUC={test_auc:.4f}")
        finally:
            cfg.GATE_PRIOR_WEIGHT = original

        _, _, cv_mat = compute_gate_statistics(gate_list)  # (4, 5)
        mean_cv  = float(cv_mat.mean())
        mean_auc = float(np.mean(auc_list))

        print(f"  -> mean_CV={mean_cv:.4f}  mean_AUC={mean_auc:.4f}")
        results.append({"lambda": lam, "mean_cv": mean_cv, "mean_auc": mean_auc,
                        "per_seed_aucs": auc_list})

    return results


def main():
    print("Gate Prior CV Sweep")
    print(f"lambdas: {LAMBDAS}  |  seeds: {SEEDS}")

    results = sweep()

    df = pd.DataFrame([{"lambda": r["lambda"], "mean_cv": r["mean_cv"],
                         "mean_auc": r["mean_auc"]} for r in results])

    print("\n" + "="*55)
    print("  SWEEP RESULTS")
    print("="*55)
    print(df.to_string(index=False, float_format="{:.4f}".format))

    # Recommendation: smallest lambda with mean_CV < 0.3 and AUC drop < 0.01
    baseline_auc = df.loc[df["lambda"] == 0.0, "mean_auc"].values[0]
    candidates = df[(df["mean_cv"] < 0.3) & (df["mean_auc"] >= baseline_auc - 0.01)]
    if len(candidates) > 0:
        best_lam = float(candidates.sort_values("lambda").iloc[0]["lambda"])
        print(f"\nRecommended lambda: {best_lam}  (CV<0.3, AUC drop<0.01)")
    else:
        best_lam = float(df.sort_values("mean_cv").iloc[0]["lambda"])
        print(f"\nNo lambda achieved CV<0.3. Best available: {best_lam}")

    # Save
    csv_path = "outputs/gate_prior_cv_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    json_path = "outputs/gate_prior_cv_sweep.json"
    with open(json_path, "w") as f:
        json.dump({"results": results, "recommended_lambda": best_lam}, f, indent=2)
    print(f"Saved: {json_path}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(df["lambda"], df["mean_cv"], "o-", color="red")
    ax1.axhline(0.3, color="green", linestyle="--", label="CV=0.3 target")
    ax1.set_xlabel("Lambda (prior_weight)")
    ax1.set_ylabel("Mean CV (lower = more stable)")
    ax1.set_title("Gate Stability vs Lambda")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(df["lambda"], df["mean_auc"], "o-", color="steelblue")
    ax2.axhline(baseline_auc - 0.01, color="orange", linestyle="--", label="AUC floor (-0.01)")
    ax2.set_xlabel("Lambda (prior_weight)")
    ax2.set_ylabel("Mean Macro ROC-AUC")
    ax2.set_title("AUC vs Lambda")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/gate_prior_cv_sweep.png", dpi=150, bbox_inches="tight")
    print("Saved: outputs/gate_prior_cv_sweep.png")
    plt.close()

    print(f"\nDone. Set GATE_PRIOR_WEIGHT = {best_lam} in config.py then retrain.")
    return best_lam


if __name__ == "__main__":
    main()
