import sys
sys.path.insert(0, "..")

"""
Phase 10b: Interpretability
1. Extract and visualize gate weights (per-sample average relative view reliance per task)
2. Compute Integrated Gradients attribution via captum (population-level)
"""

import sys, os
sys.path.insert(0, ".")

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend

import config as cfg
from src.data_pipeline import load_data, get_loaders
from src.model import TGMVMTGFNetV2
from src.loss  import UncertaintyWeightedLoss

os.makedirs("outputs", exist_ok=True)


def load_best_model(vocab):
    model   = TGMVMTGFNetV2(vocab)
    loss_fn = UncertaintyWeightedLoss()
    ckpt    = torch.load(cfg.CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    loss_fn.load_state_dict(ckpt["loss_state"])
    model.eval()
    return model, loss_fn


def extract_gate_weights(model, loader):
    """
    Returns avg_gates: (num_tasks, num_gate_inputs) numpy array
    """
    all_gates = [[] for _ in range(cfg.NUM_TASKS)]
    with torch.no_grad():
        for views, _, _ in loader:
            _, gate_weights, _ = model(views)
            for k, gw in enumerate(gate_weights):
                all_gates[k].append(gw.numpy())   # (batch, num_gate_inputs)

    avg_gates = np.array([np.vstack(all_gates[k]).mean(axis=0)
                           for k in range(cfg.NUM_TASKS)])  # (4, num_gate_inputs)
    return avg_gates


def plot_gate_weights(avg_gates):
    """Bar chart: relative view reliance per task."""
    view_labels = ["Rider Role", "Rider Traits", "Road Ctx", "Environment", "Site", "Interaction"]
    fig, axes   = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for k, task in enumerate(cfg.TASK_NAMES):
        axes[k].barh(view_labels, avg_gates[k], color="steelblue")
        axes[k].set_title(task.replace("_", " ").title())
        axes[k].set_xlim(0, 1)
        axes[k].set_xlabel("Gate weight (normalized)")

    plt.suptitle("Gate Weight Attribution per Task", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = "outputs/gate_weights.png"
    plt.savefig(out, dpi=150)
    print(f"Gate weight plot saved to {out}")
    plt.close()


def tau_report(loss_fn):
    """Print learned task uncertainty tau values."""
    taus = torch.exp(0.5 * loss_fn.log_tau_sq).detach().tolist()
    print("\n=== Learned Task Uncertainty (tau_k) ===")
    for task, tau in zip(cfg.TASK_NAMES, taus):
        print(f"  {task:<30}: tau={tau:.4f}  (higher = harder to predict)")


def alpha_report(model):
    alpha = torch.sigmoid(model.fusion.alpha_logit).item()
    print(f"\n=== Alpha Blend ===")
    print(f"  alpha={alpha:.4f}  (1=pure gated fusion, 0=pure early fusion)")


if __name__ == "__main__":
    train_df, val_df, test_df, encoders, vocab = load_data()
    _, _, test_loader = get_loaders(train_df, val_df, test_df, vocab)

    model, loss_fn = load_best_model(vocab)

    avg_gates = extract_gate_weights(model, test_loader)
    print("\n=== Gate Weights (avg over test set) ===")
    header = f"{'View':<15}" + "".join(f"{t[:12]:>14}" for t in cfg.TASK_NAMES)
    print(header)
    view_labels = ["Rider Role", "Rider Traits", "Road Ctx", "Environment", "Site", "Interaction"]
    for i, vl in enumerate(view_labels):
        row = f"{vl:<15}" + "".join(f"{avg_gates[k][i]:>14.3f}" for k in range(cfg.NUM_TASKS))
        print(row)

    plot_gate_weights(avg_gates)
    tau_report(loss_fn)
    alpha_report(model)
