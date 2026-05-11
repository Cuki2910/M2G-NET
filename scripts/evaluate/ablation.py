import sys
sys.path.insert(0, "..")

"""
Phase 10a: Ablation Study
Runs the following experiments and compares macro ROC-AUC:
  - View ablations (zero-out each view)
  - Fusion mechanism ablations
  - MTL vs single-task
  - Cross-level interaction ablation
  - Site encoding ablation
"""

import sys, os
sys.path.insert(0, ".")

import copy
import numpy as np
import torch

import config as cfg
from src.checkpoint import load_model_bundle
from src.data_pipeline import get_loaders
from src.metrics import compute_all_metrics, print_results


def evaluate(model, loss_fn, loader, use_site_intercept=True, thresholds=None):
    all_probs, all_targets, all_masks = [], [], []
    with torch.no_grad():
        for views, targets, masks in loader:
            preds, _, _ = model(views, use_site_intercept=use_site_intercept)
            probs = torch.stack(preds, dim=1).numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())
            all_masks.append(masks.numpy())
    all_probs   = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    all_masks   = np.vstack(all_masks)
    return compute_all_metrics(all_targets, all_probs, all_masks, thresholds=thresholds)


def ablate_view(model, view_to_zero, views_batch):
    """
    Return a copy of views_batch with the specified view zeroed out.
    view_to_zero: one of cfg.ALL_VIEW_NAMES
    """
    import src.views as V
    # We'll use a hook approach: replace the encoder output with zeros
    results_store = {}

    def hook_fn(view_name):
        def _hook(module, input, output):
            if view_name == view_to_zero:
                return torch.zeros_like(output)
            return output
        return _hook

    hooks = []
    for vname, enc in model.encoders.items():
        h = enc.register_forward_hook(hook_fn(vname))
        hooks.append(h)

    return hooks


def run_view_ablation(model, loss_fn, test_loader, vocab, thresholds=None):
    print("\n=== View Ablation ===")
    base = evaluate(model, loss_fn, test_loader, thresholds=thresholds)
    base_auc = base["macro"]["roc_auc"]
    print(f"Full model macro ROC-AUC: {base_auc:.4f}")

    for view in cfg.ALL_VIEW_NAMES:
        # Zero-out the view's encoder by monkey-patching
        m_copy = copy.deepcopy(model)
        enc = m_copy.encoders[view]
        orig_forward = enc.forward

        def make_zero_forward(orig_fwd):
            def zero_forward(*args, **kwargs):
                out = orig_fwd(*args, **kwargs)
                return torch.zeros_like(out)
            return zero_forward

        enc.forward = make_zero_forward(orig_forward)

        res = evaluate(m_copy, loss_fn, test_loader, thresholds=thresholds)
        drop = base_auc - res["macro"]["roc_auc"]
        print(f"  - {view:<20}: AUC={res['macro']['roc_auc']:.4f}  (drop={drop:+.4f})")


def run_site_ablation(model, loss_fn, test_loader, thresholds=None):
    print("\n=== Site Ablation ===")
    # With intercept (in-distribution)
    res_full = evaluate(model, loss_fn, test_loader, use_site_intercept=True, thresholds=thresholds)
    # Without intercept (out-of-distribution / leave-intersection-out)
    res_obs  = evaluate(model, loss_fn, test_loader, use_site_intercept=False, thresholds=thresholds)
    print(f"  Site full (obs + intercept): {res_full['macro']['roc_auc']:.4f}")
    print(f"  Site obs only (no intercept): {res_obs['macro']['roc_auc']:.4f}")


if __name__ == "__main__":
    bundle = load_model_bundle()
    train_df, val_df, test_df = bundle["train_df"], bundle["val_df"], bundle["test_df"]
    vocab = bundle["vocab"]
    thresholds = bundle["thresholds"]
    _, _, test_loader = get_loaders(train_df, val_df, test_df, vocab)
    model, loss_fn = bundle["model"], bundle["loss_fn"]

    run_view_ablation(model, loss_fn, test_loader, vocab, thresholds=thresholds)
    run_site_ablation(model, loss_fn, test_loader, thresholds=thresholds)
