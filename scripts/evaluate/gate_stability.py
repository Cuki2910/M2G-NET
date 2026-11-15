"""Validation-set gate diagnostics, including the interaction input."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import config as cfg

VIEW_LABELS = ["Rider Role", "Rider Traits", "Road Context", "Environment", "Site", "Interaction"]


def extract_gate_weights(model, loader):
    model.eval()
    weights = [[] for _ in cfg.TASK_NAMES]
    with torch.no_grad():
        for views, _, _ in loader:
            _, gates, _ = model(views)
            for task_idx, gate in enumerate(gates):
                weights[task_idx].append(gate.numpy())
    if not all(weights):
        raise ValueError("Gate diagnostics require a gated model and nonempty loader.")
    return np.stack([np.vstack(task).mean(axis=0) for task in weights])


def compute_gate_statistics(gate_weights_list):
    weights = np.stack(gate_weights_list)
    mean = weights.mean(axis=0)
    std = weights.std(axis=0, ddof=1) if len(weights) > 1 else np.full_like(mean, np.nan)
    return mean, std, std / (mean + 1e-9)


def main():
    from scripts.evaluate.gate_prior_sweep import main as sweep
    sweep(["--lambdas", str(cfg.GATE_PRIOR_WEIGHT), *sys.argv[1:]])


if __name__ == "__main__":
    main()
