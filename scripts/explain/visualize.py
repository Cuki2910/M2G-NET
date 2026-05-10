import sys
sys.path.insert(0, "..")

"""
visualize.py — Enhanced Visualizations
1. Gate weights heatmap (avg over test set)
2. Per-sample gate attention (first 200 samples)
3. Confusion matrix per task
"""

import sys, os
sys.path.insert(0, ".")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import config as cfg
from src.data_pipeline import load_data, get_loaders
from src.model import TGMVMTGFNetV2
from src.loss  import UncertaintyWeightedLoss

os.makedirs("outputs", exist_ok=True)

VIEW_LABELS = ["Rider\nRole", "Rider\nTraits", "Road\nCtx", "Environ\n-ment", "Site", "Inter-\naction"]
TASK_LABELS = [t.replace("_", "\n") for t in cfg.TASK_NAMES]


def load_best_model(vocab):
    model   = TGMVMTGFNetV2(vocab)
    loss_fn = UncertaintyWeightedLoss()
    ckpt    = torch.load(cfg.CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    loss_fn.load_state_dict(ckpt["loss_state"])
    model.eval()
    return model, loss_fn


def collect_predictions(model, loader):
    """Collect all predictions, targets, and gate weights."""
    all_probs   = []
    all_targets = []
    all_gates   = [[] for _ in range(cfg.NUM_TASKS)]   # per-task gate weights

    with torch.no_grad():
        for views, targets, masks in loader:
            preds, gate_weights, _ = model(views)
            probs = torch.stack(preds, dim=1).numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())
            for k, gw in enumerate(gate_weights):
                # gw: (batch, 6) — take first 5 (view gates)
                all_gates[k].append(gw.numpy())

    return (np.vstack(all_probs),
            np.vstack(all_targets),
            [np.vstack(all_gates[k]) for k in range(cfg.NUM_TASKS)])


# ── 1. Gate weights heatmap ────────────────────────────────────────────────────

def plot_gate_heatmap(all_gates):
    """Average gate weight matrix: tasks × views → heatmap."""
    avg = np.array([g.mean(axis=0) for g in all_gates])  # (4, 6)
    # Normalize rows to sum to 1
    avg = avg / avg.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(avg, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(VIEW_LABELS))); ax.set_xticklabels(VIEW_LABELS, fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels(TASK_LABELS, fontsize=9)
    ax.set_title("Gate Weight Heatmap\n(avg over test set, normalized per task)", fontsize=11)
    plt.colorbar(im, ax=ax, label="Attention weight")

    # Annotate cells
    for i in range(4):
        for j in range(len(VIEW_LABELS)):
            ax.text(j, i, f"{avg[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if avg[i,j] < 0.6 else "white")

    plt.tight_layout()
    out = "outputs/gate_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ── 2. Per-sample gate attention (first 200 samples) ─────────────────────────

def plot_per_sample_attention(all_gates, n_samples=200):
    """Show individual variation in gate attention for each task."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    cmap = sns.color_palette("husl", len(VIEW_LABELS))

    for k, task in enumerate(cfg.TASK_NAMES):
        gate_k = all_gates[k][:n_samples]            # (n_samples, 6)
        # Normalize
        gate_k = gate_k / gate_k.sum(axis=1, keepdims=True)
        ax = axes[k]
        bottom = np.zeros(n_samples)
        for v_idx, (vl, c) in enumerate(zip(VIEW_LABELS, cmap)):
            ax.bar(range(n_samples), gate_k[:, v_idx], bottom=bottom,
                   color=c, label=vl.replace("\n", " "), width=1.0, linewidth=0)
            bottom += gate_k[:, v_idx]
        ax.set_title(task.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("Sample index", fontsize=8)
        ax.set_ylabel("Gate weight", fontsize=8)
        ax.set_xlim(0, n_samples)
        ax.set_ylim(0, 1)
        if k == 0:
            ax.legend(loc="upper right", fontsize=7, ncol=2)

    plt.suptitle(f"Per-Sample Gate Attention (first {n_samples} test samples)", fontsize=12)
    plt.tight_layout()
    out = "outputs/per_sample_attention.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ── 3. Confusion matrix per task ──────────────────────────────────────────────

def plot_confusion_matrices(all_probs, all_targets, threshold=0.5):
    """2×2 confusion matrix for each of the 4 tasks."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()

    for k, task in enumerate(cfg.TASK_NAMES):
        y_true = all_targets[:, k].astype(int)
        y_pred = (all_probs[:, k] >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[k],
                    xticklabels=["Pred 0", "Pred 1"],
                    yticklabels=["True 0", "True 1"],
                    cbar=False, linewidths=0.5)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, 0)
        total = tn + fp + fn + tp
        acc   = (tn + tp) / total if total > 0 else 0
        axes[k].set_title(
            f"{task.replace('_',' ').title()}\nAcc={acc:.2%}  TP={tp}  FP={fp}  FN={fn}",
            fontsize=9)

    plt.suptitle("Confusion Matrices per Task (threshold=0.5)", fontsize=12)
    plt.tight_layout()
    out = "outputs/confusion_matrices.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_df, val_df, test_df, encoders, vocab = load_data()
    _, _, test_loader = get_loaders(train_df, val_df, test_df, vocab)
    model, loss_fn = load_best_model(vocab)

    all_probs, all_targets, all_gates = collect_predictions(model, test_loader)

    plot_gate_heatmap(all_gates)
    plot_per_sample_attention(all_gates)
    plot_confusion_matrices(all_probs, all_targets)

    print("All visualizations saved to outputs/")
