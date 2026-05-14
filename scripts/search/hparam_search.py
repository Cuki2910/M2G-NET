import sys
sys.path.insert(0, "..")

"""
hparam_search.py — Comprehensive grid search
Searches over: learning_rate × batch_size × hidden_dim × view_dim × dropout
Evaluates val ROC-AUC after fixed 30 epochs (no early stopping for speed).
Prints ranking table and saves best config.
"""

import sys, os
sys.path.insert(0, ".")

import itertools, json
import numpy as np
import torch
import torch.optim as optim

import config as cfg
from src.data_pipeline import load_data, get_loaders
from src.model import TGMVMTGFNetV2
from src.loss  import UncertaintyWeightedLoss
from src.metrics import compute_all_metrics


# ── Search Space ──────────────────────────────────────────────────────────────

SEARCH_SPACE = {
    "lr":         [1e-3, 5e-4, 2e-3],
    "batch_size": [32, 64, 128],
    "hidden_dim": [16, 32, 64],
    "view_dim":   [8, 16, 32],
    "dropout":    [0.2, 0.3, 0.4],
}

# Optional: smaller search for quick testing
QUICK_SEARCH = {
    "lr":         [1e-3, 5e-4],
    "batch_size": [64],
    "hidden_dim": [32],
    "view_dim":   [16],
    "dropout":    [0.3],
}

# Stage-2: gate regularization + loss type + weight decay.
# Run after stage-1 to find best arch config first, then fine-tune these.
# Reads outputs/best_config.json (written by stage-1) to fix arch params.
STAGE2_SEARCH = {
    "gate_prior_weight": [0.0, 0.05, 0.1, 0.2],
    "use_focal":         [True, False],
    "weight_decay":      [1e-5, 1e-4, 1e-3],
}

EVAL_EPOCHS = 30    # fast evaluation
USE_QUICK = False   # Set to True for quick testing


def build_model_with(hidden_dim, view_dim, dropout, vocab,
                     gate_prior_weight=None, use_focal=None):
    """Temporarily override config params, build model, then restore."""
    orig_hidden = cfg.HIDDEN_DIM
    orig_view   = cfg.VIEW_DIM
    orig_drop   = cfg.DROPOUT_RATE
    orig_gate   = cfg.GATE_PRIOR_WEIGHT
    orig_focal  = cfg.USE_FOCAL_LOSS
    orig_inter  = cfg.INTERACTION_DIM

    cfg.HIDDEN_DIM       = hidden_dim
    cfg.VIEW_DIM         = view_dim
    cfg.DROPOUT_RATE     = dropout
    cfg.INTERACTION_DIM  = view_dim   # keep interaction dim in sync with view dim
    if gate_prior_weight is not None:
        cfg.GATE_PRIOR_WEIGHT = gate_prior_weight
    if use_focal is not None:
        cfg.USE_FOCAL_LOSS = use_focal

    model   = TGMVMTGFNetV2(vocab)
    loss_fn = UncertaintyWeightedLoss()

    cfg.HIDDEN_DIM        = orig_hidden
    cfg.VIEW_DIM          = orig_view
    cfg.DROPOUT_RATE      = orig_drop
    cfg.GATE_PRIOR_WEIGHT = orig_gate
    cfg.USE_FOCAL_LOSS    = orig_focal
    cfg.INTERACTION_DIM   = orig_inter

    return model, loss_fn


def quick_train(vocab, train_df, val_df, test_df, lr, batch_size, hidden_dim, view_dim, dropout,
                gate_prior_weight=None, use_focal=None, weight_decay=None):
    """Train for EVAL_EPOCHS and return val AUC."""
    model, loss_fn = build_model_with(hidden_dim, view_dim, dropout, vocab,
                                      gate_prior_weight=gate_prior_weight,
                                      use_focal=use_focal)
    wd = weight_decay if weight_decay is not None else cfg.WEIGHT_DECAY
    optimizer = optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=lr, weight_decay=wd,
    )

    train_loader, val_loader, _ = get_loaders(
        train_df, val_df, test_df, vocab, batch_size=batch_size)

    for epoch in range(1, EVAL_EPOCHS + 1):
        model.train()
        for views, targets, masks in train_loader:
            preds, _, _ = model(views)
            loss, _ = loss_fn(preds, targets, masks)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Evaluate on val
    model.eval()
    all_probs, all_targets, all_masks = [], [], []
    with torch.no_grad():
        for views, targets, masks in val_loader:
            preds, _, _ = model(views)
            probs = torch.stack(preds, dim=1).numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())
            all_masks.append(masks.numpy())

    all_probs   = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    all_masks   = np.vstack(all_masks)
    metrics = compute_all_metrics(all_targets, all_probs, all_masks)
    return metrics["macro"]["roc_auc"]


def run_grid_search():
    train_df, val_df, test_df, encoders, vocab = load_data()

    space = QUICK_SEARCH if USE_QUICK else SEARCH_SPACE
    keys   = list(space.keys())
    values = list(space.values())
    combos = list(itertools.product(*values))

    print(f"Grid search: {len(combos)} combinations × {EVAL_EPOCHS} epochs each")
    print(f"Search space: {space}\n")
    print(f"{'LR':>8} {'Batch':>7} {'HidDim':>8} {'ViewDim':>9} {'Dropout':>9} | {'Val AUC':>9}")
    print("-" * 70)

    results = []
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        try:
            val_auc = quick_train(vocab, train_df, val_df, test_df, **params)
        except Exception as e:
            print(f"Error with config {params}: {e}")
            val_auc = float("nan")

        results.append({**params, "val_auc": val_auc})
        print(f"{params['lr']:>8.4f} {params['batch_size']:>7} {params['hidden_dim']:>8} "
              f"{params['view_dim']:>9} {params['dropout']:>9.2f} | {val_auc:>9.4f}  "
              f"[{i}/{len(combos)}]")

    # Sort and report best
    results.sort(key=lambda x: x["val_auc"] if not np.isnan(x["val_auc"]) else -1, reverse=True)
    best = results[0]

    print(f"\n{'='*70}")
    print(f"Best config:")
    print(f"  LR={best['lr']}, batch={best['batch_size']}, hidden={best['hidden_dim']}, "
          f"view={best['view_dim']}, dropout={best['dropout']}")
    print(f"  Val AUC={best['val_auc']:.4f}")
    print(f"{'='*70}")

    # Show top 5
    print("\nTop 5 configurations:")
    for i, res in enumerate(results[:5], 1):
        print(f"{i}. AUC={res['val_auc']:.4f} | LR={res['lr']:.4f}, batch={res['batch_size']}, "
              f"hidden={res['hidden_dim']}, view={res['view_dim']}, dropout={res['dropout']:.2f}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/hparam_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to outputs/hparam_results.json")

    # Save best config separately
    with open("outputs/best_config.json", "w") as f:
        json.dump(best, f, indent=2)
    print("Best config saved to outputs/best_config.json")

    return best


def run_stage2_search():
    """Stage-2: gate prior + loss type + weight decay, arch params fixed from stage-1 best."""
    train_df, val_df, test_df, encoders, vocab = load_data()

    # Load best arch params from stage-1 if available, else fall back to config defaults
    arch = {
        "hidden_dim": cfg.HIDDEN_DIM,
        "view_dim":   cfg.VIEW_DIM,
        "dropout":    cfg.DROPOUT_RATE,
    }
    lr_best    = cfg.LEARNING_RATE
    batch_best = cfg.BATCH_SIZE
    if os.path.exists("outputs/best_config.json"):
        with open("outputs/best_config.json") as f:
            best_s1 = json.load(f)
        arch["hidden_dim"] = best_s1.get("hidden_dim", arch["hidden_dim"])
        arch["view_dim"]   = best_s1.get("view_dim",   arch["view_dim"])
        arch["dropout"]    = best_s1.get("dropout",    arch["dropout"])
        lr_best    = best_s1.get("lr",         lr_best)
        batch_best = best_s1.get("batch_size", batch_best)
        print(f"Loaded stage-1 best config: {arch}, lr={lr_best}, batch={batch_best}")
    else:
        print("No outputs/best_config.json found; using config.py defaults.")

    keys   = list(STAGE2_SEARCH.keys())
    combos = list(itertools.product(*STAGE2_SEARCH.values()))
    print(f"\nStage-2 search: {len(combos)} combinations x {EVAL_EPOCHS} epochs each")
    print(f"Space: {STAGE2_SEARCH}\n")
    print(f"{'gate_prior':>12} {'focal':>7} {'weight_decay':>14} | {'Val AUC':>9}")
    print("-" * 50)

    results = []
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        try:
            val_auc = quick_train(
                vocab, train_df, val_df, test_df,
                lr=lr_best, batch_size=batch_best,
                **arch, **params,
            )
        except Exception as e:
            print(f"Error: {e}")
            val_auc = float("nan")

        results.append({**params, "val_auc": val_auc})
        print(f"{params['gate_prior_weight']:>12.2f} {str(params['use_focal']):>7} "
              f"{params['weight_decay']:>14.0e} | {val_auc:>9.4f}  [{i}/{len(combos)}]")

    results.sort(key=lambda x: x["val_auc"] if not np.isnan(x["val_auc"]) else -1, reverse=True)
    best = results[0]

    print(f"\n{'='*50}")
    print("Best stage-2 config:")
    print(f"  gate_prior={best['gate_prior_weight']}, focal={best['use_focal']}, "
          f"weight_decay={best['weight_decay']}")
    print(f"  Val AUC={best['val_auc']:.4f}")
    print(f"{'='*50}")

    print("\nTop 5 stage-2 configurations:")
    for i, res in enumerate(results[:5], 1):
        print(f"{i}. AUC={res['val_auc']:.4f} | gate={res['gate_prior_weight']:.2f}, "
              f"focal={res['use_focal']}, wd={res['weight_decay']:.0e}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/hparam_stage2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to outputs/hparam_stage2_results.json")
    return best


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",  action="store_true", help="Use quick search space (faster)")
    parser.add_argument("--stage2", action="store_true",
                        help="Run stage-2 search: gate prior + loss type + weight decay.")
    args = parser.parse_args()

    if args.stage2:
        best = run_stage2_search()
    else:
        USE_QUICK = args.quick
        if USE_QUICK:
            print("Using QUICK search space for testing\n")
        best = run_grid_search()
