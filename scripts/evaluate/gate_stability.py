"""
Gate Stability Analysis
Trains the model with multiple random seeds and analyzes the variance of gate weights.
This addresses the concern that gate weights might converge to random attention configurations.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import config as cfg
from src.data_pipeline import load_data, get_loaders
from scripts.train import train

os.makedirs("outputs", exist_ok=True)

VIEW_LABELS = ["Rider Role", "Rider Traits", "Road Context", "Environment", "Site"]


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_gate_weights(model, test_loader):
    """Extract average gate weights over test set."""
    all_gates = [[] for _ in range(cfg.NUM_TASKS)]
    with torch.no_grad():
        for views, _, _ in test_loader:
            _, gate_weights, _ = model(views)
            for k, gw in enumerate(gate_weights):
                # Only take first 5 views (exclude interaction term for stability analysis)
                all_gates[k].append(gw[:, :5].numpy())

    avg_gates = np.array([np.vstack(all_gates[k]).mean(axis=0)
                          for k in range(cfg.NUM_TASKS)])  # (4, 5)
    return avg_gates


def compute_gate_statistics(gate_weights_list):
    """
    Compute mean, std, and coefficient of variation for gate weights.
    gate_weights_list: list of (4, 5) arrays, one per seed
    Returns: mean, std, cv (all shape (4, 5))
    """
    gates_array = np.stack(gate_weights_list, axis=0)  # (n_seeds, 4, 5)
    mean = gates_array.mean(axis=0)  # (4, 5)
    std = gates_array.std(axis=0)    # (4, 5)
    cv = std / (mean + 1e-9)          # coefficient of variation
    return mean, std, cv


def plot_gate_stability(mean, std, cv):
    """Plot gate weights with error bars showing stability across seeds."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    x = np.arange(5)
    width = 0.6

    for k, task in enumerate(cfg.TASK_NAMES):
        ax = axes[k]
        bars = ax.bar(x, mean[k], width, yerr=std[k], capsize=5,
                      color="steelblue", alpha=0.8, ecolor="darkred", linewidth=2)

        ax.set_xticks(x)
        ax.set_xticklabels(VIEW_LABELS, fontsize=9, rotation=15, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(task.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_ylabel("Gate Weight (mean ± std)")
        ax.grid(axis="y", alpha=0.3)

        # Add CV annotations
        for i in range(5):
            if cv[k][i] < 0.2:
                color = "green"
                marker = "✓"
            elif cv[k][i] < 0.5:
                color = "orange"
                marker = "~"
            else:
                color = "red"
                marker = "!"
            ax.text(i, mean[k][i] + std[k][i] + 0.03, marker,
                   ha="center", fontsize=12, color=color, fontweight="bold")

    plt.suptitle("Gate Weight Stability Across Random Seeds\n(✓ = CV < 0.2, ~ = CV < 0.5, ! = CV ≥ 0.5)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "outputs/gate_stability.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_gate_heatmap(mean, std, cv):
    """Heatmap showing coefficient of variation for each task-view pair."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Mean
    im1 = axes[0].imshow(mean, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    axes[0].set_title("Mean Gate Weight", fontsize=12, fontweight="bold")
    axes[0].set_xticks(range(5))
    axes[0].set_xticklabels(VIEW_LABELS, rotation=45, ha="right", fontsize=9)
    axes[0].set_yticks(range(4))
    axes[0].set_yticklabels([t.replace("_", " ").title() for t in cfg.TASK_NAMES], fontsize=9)
    plt.colorbar(im1, ax=axes[0])

    # Add values
    for i in range(4):
        for j in range(5):
            axes[0].text(j, i, f"{mean[i,j]:.2f}", ha="center", va="center",
                        color="white" if mean[i,j] > 0.5 else "black", fontsize=9)

    # Std
    im2 = axes[1].imshow(std, cmap="Oranges", aspect="auto", vmin=0, vmax=0.3)
    axes[1].set_title("Std Dev Across Seeds", fontsize=12, fontweight="bold")
    axes[1].set_xticks(range(5))
    axes[1].set_xticklabels(VIEW_LABELS, rotation=45, ha="right", fontsize=9)
    axes[1].set_yticks(range(4))
    axes[1].set_yticklabels([t.replace("_", " ").title() for t in cfg.TASK_NAMES], fontsize=9)
    plt.colorbar(im2, ax=axes[1])

    for i in range(4):
        for j in range(5):
            axes[1].text(j, i, f"{std[i,j]:.2f}", ha="center", va="center",
                        color="white" if std[i,j] > 0.15 else "black", fontsize=9)

    # CV
    im3 = axes[2].imshow(cv, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    axes[2].set_title("Coefficient of Variation (CV)", fontsize=12, fontweight="bold")
    axes[2].set_xticks(range(5))
    axes[2].set_xticklabels(VIEW_LABELS, rotation=45, ha="right", fontsize=9)
    axes[2].set_yticks(range(4))
    axes[2].set_yticklabels([t.replace("_", " ").title() for t in cfg.TASK_NAMES], fontsize=9)
    plt.colorbar(im3, ax=axes[2])

    for i in range(4):
        for j in range(5):
            axes[2].text(j, i, f"{cv[i,j]:.2f}", ha="center", va="center",
                        color="white" if cv[i,j] > 0.5 else "black", fontsize=9)

    plt.suptitle("Gate Weight Stability Analysis (Lower CV = More Stable)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = "outputs/gate_stability_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def main():
    n_runs = 10
    seeds = [42, 101, 2023, 777, 999, 1234, 5678, 8888, 3141, 2718]

    print(f"=== Gate Stability Analysis ({n_runs} runs) ===\n")

    gate_weights_list = []
    test_aucs = []

    for i, seed in enumerate(seeds):
        print(f"Run {i+1}/{n_runs} (Seed: {seed})...")
        set_random_seed(seed)

        # Load data
        train_df, val_df, test_df, encoders, vocab = load_data(seed=seed)

        # Train model
        model, loss_fn, test_metrics, history = train(
            vocab, train_df, val_df, test_df,
            max_epochs=cfg.MAX_EPOCHS,
            patience=cfg.EARLY_STOPPING_PATIENCE,
            encoders=encoders,
            split_seed=seed,
        )

        # Extract gate weights
        _, _, test_loader = get_loaders(train_df, val_df, test_df, vocab)
        gates = extract_gate_weights(model, test_loader)
        gate_weights_list.append(gates)

        test_auc = test_metrics["macro"]["roc_auc"]
        test_aucs.append(test_auc)

        print(f"  Test AUC: {test_auc:.4f}")
        print(f"  Gate weights (avg across tasks): {gates.mean(axis=0)}")
        print()

    # Compute statistics
    mean, std, cv = compute_gate_statistics(gate_weights_list)

    # Print summary
    print("\n=== GATE WEIGHT STABILITY SUMMARY ===\n")
    print("Mean Gate Weights (averaged across seeds):")
    print(f"{'View':<20}", end="")
    for task in cfg.TASK_NAMES:
        print(f"{task[:12]:>14}", end="")
    print()
    print("-" * 80)
    for v_idx, vl in enumerate(VIEW_LABELS):
        print(f"{vl:<20}", end="")
        for k in range(cfg.NUM_TASKS):
            print(f"{mean[k][v_idx]:>14.3f}", end="")
        print()

    print("\n\nStandard Deviation (across seeds):")
    print(f"{'View':<20}", end="")
    for task in cfg.TASK_NAMES:
        print(f"{task[:12]:>14}", end="")
    print()
    print("-" * 80)
    for v_idx, vl in enumerate(VIEW_LABELS):
        print(f"{vl:<20}", end="")
        for k in range(cfg.NUM_TASKS):
            print(f"{std[k][v_idx]:>14.3f}", end="")
        print()

    print("\n\nCoefficient of Variation (CV = std/mean):")
    print(f"{'View':<20}", end="")
    for task in cfg.TASK_NAMES:
        print(f"{task[:12]:>14}", end="")
    print()
    print("-" * 80)
    for v_idx, vl in enumerate(VIEW_LABELS):
        print(f"{vl:<20}", end="")
        for k in range(cfg.NUM_TASKS):
            print(f"{cv[k][v_idx]:>14.3f}", end="")
        print()

    # Stability assessment
    print("\n\n=== STABILITY ASSESSMENT ===")
    stable_count = (cv < 0.2).sum()
    moderate_count = ((cv >= 0.2) & (cv < 0.5)).sum()
    unstable_count = (cv >= 0.5).sum()
    total = cv.size

    print(f"Stable (CV < 0.2):     {stable_count}/{total} ({100*stable_count/total:.1f}%)")
    print(f"Moderate (0.2 ≤ CV < 0.5): {moderate_count}/{total} ({100*moderate_count/total:.1f}%)")
    print(f"Unstable (CV ≥ 0.5):   {unstable_count}/{total} ({100*unstable_count/total:.1f}%)")

    if stable_count / total > 0.7:
        print("\n✓ Conclusion: Gate weights are STABLE across random seeds.")
    elif moderate_count / total > 0.5:
        print("\n~ Conclusion: Gate weights show MODERATE stability across random seeds.")
    else:
        print("\n✗ Conclusion: Gate weights are UNSTABLE across random seeds.")

    # Test AUC stability
    print(f"\n\nTest AUC Stability:")
    print(f"Mean: {np.mean(test_aucs):.4f}")
    print(f"Std:  {np.std(test_aucs):.4f}")
    print(f"CV:   {np.std(test_aucs)/np.mean(test_aucs):.4f}")

    # Plot
    plot_gate_stability(mean, std, cv)
    plot_gate_heatmap(mean, std, cv)

    print("\n✓ Gate stability analysis complete.")


if __name__ == "__main__":
    main()
