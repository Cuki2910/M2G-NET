"""
Phase 8: Training Loop
- Adam optimizer with L2 regularization
- Temperature annealing for gate regularization
- Early stopping on macro ROC-AUC
- Saves best checkpoint
- Supports Leave-Intersection-Out evaluation
"""

import copy
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import config as cfg
from src.checkpoint import make_checkpoint_payload
from src.data_pipeline import (
    compute_pos_weights,
    load_data,
    get_loaders,
    encode_splits,
    RiderDataset,
    safe_train_val_split,
)
from src.model         import TGMVMTGFNetV2, count_parameters
from src.loss          import UncertaintyWeightedLoss
from src.metrics       import compute_all_metrics, print_results, tune_task_thresholds


os.makedirs("checkpoints", exist_ok=True)


def anneal_temperature(model, epoch, max_epochs=cfg.MAX_EPOCHS,
                        t_init=cfg.TEMPERATURE_INIT, t_final=cfg.TEMPERATURE_FINAL,
                        schedule="cosine"):
    """
    Gate temperature annealing: cosine (default) or linear.
    Cosine schedule gives a smooth S-curve decay, avoiding abrupt transitions.
    """
    progress = min(epoch / max_epochs, 1.0)
    if schedule == "cosine":
        # Cosine annealing: starts slow, speeds up, then slows again
        new_temp = t_final + 0.5 * (t_init - t_final) * (1 + math.cos(math.pi * progress))
    else:
        new_temp = t_init + (t_final - t_init) * progress
    for gate in model.fusion.gates:
        gate.temperature.data.fill_(new_temp)


def collect_predictions(model, loader, use_site_intercept=True):
    all_probs, all_targets, all_masks = [], [], []
    with torch.no_grad():
        for views, targets, masks in loader:
            preds, _, _ = model(views, use_site_intercept=use_site_intercept)
            all_probs.append(torch.stack(preds, dim=1).detach().numpy())
            all_targets.append(targets.numpy())
            all_masks.append(masks.numpy())
    return np.vstack(all_targets), np.vstack(all_probs), np.vstack(all_masks)


def run_epoch(model, loader, loss_fn, optimizer=None, use_site_intercept=True,
              thresholds=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    all_probs, all_targets, all_masks = [], [], []
    total_loss = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for views, targets, masks in loader:
            preds, _, _ = model(views, use_site_intercept=use_site_intercept)
            loss, task_losses = loss_fn(preds, targets, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

            probs = torch.stack(preds, dim=1).detach().numpy()   # (batch, 4)
            all_probs.append(probs)
            all_targets.append(targets.numpy())
            all_masks.append(masks.numpy())

    all_probs   = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    all_masks   = np.vstack(all_masks)
    metrics     = compute_all_metrics(all_targets, all_probs, all_masks, thresholds=thresholds)
    avg_loss    = total_loss / max(n_batches, 1)
    return avg_loss, metrics, task_losses


def fit_model_for_split(vocab, train_df, val_df, encoders=None,
                        max_epochs=cfg.MAX_EPOCHS, patience=cfg.EARLY_STOPPING_PATIENCE,
                        checkpoint_path=None):
    """Train a fresh model for one already-encoded split and return the best weights."""
    model = TGMVMTGFNetV2(vocab)
    pos_weight = compute_pos_weights(train_df) if cfg.USE_POS_WEIGHT else None
    loss_fn = UncertaintyWeightedLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    train_loader, val_loader, _ = get_loaders(train_df, val_df, val_df, vocab)

    best_val_auc = -1.0
    best_model_state = None
    best_loss_state = None
    best_epoch = 0
    patience_ctr = 0

    for epoch in range(1, max_epochs + 1):
        anneal_temperature(model, epoch, max_epochs, schedule="cosine")
        run_epoch(model, train_loader, loss_fn, optimizer)
        scheduler.step()
        _, val_metrics, _ = run_epoch(model, val_loader, loss_fn)
        val_auc = val_metrics["macro"]["roc_auc"]
        if not np.isfinite(val_auc):
            continue

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_ctr = 0
            best_model_state = copy.deepcopy(model.state_dict())
            best_loss_state = copy.deepcopy(loss_fn.state_dict())
            if checkpoint_path:
                torch.save(
                    make_checkpoint_payload(
                        epoch, model, loss_fn, val_auc, vocab, encoders=encoders),
                    checkpoint_path,
                )
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_model_state is None:
        raise RuntimeError("No finite validation ROC-AUC was produced; cannot select a checkpoint.")

    model.load_state_dict(best_model_state)
    loss_fn.load_state_dict(best_loss_state)
    return model, loss_fn, best_val_auc, best_epoch


def evaluate_leave_intersection_out(raw_df, max_epochs=50, patience=10, min_test_size=5):
    """
    True leave-intersection-out evaluation: retrain a fresh model per held-out site.
    This is intentionally separate from normal training because it is expensive.
    """
    lio_aucs = []
    for site_id in sorted(raw_df[cfg.SITE_ID_COL].unique()):
        lio_test_raw = raw_df[raw_df[cfg.SITE_ID_COL] == site_id].reset_index(drop=True)
        if len(lio_test_raw) < min_test_size:
            continue

        lio_trainval_raw = raw_df[raw_df[cfg.SITE_ID_COL] != site_id].reset_index(drop=True)
        lio_train_raw, lio_val_raw = safe_train_val_split(lio_trainval_raw)
        lio_train, lio_val, lio_test, lio_encoders, lio_vocab = encode_splits(
            lio_train_raw.reset_index(drop=True),
            lio_val_raw.reset_index(drop=True),
            lio_test_raw,
        )
        model, loss_fn, _, _ = fit_model_for_split(
            lio_vocab, lio_train, lio_val, encoders=lio_encoders,
            max_epochs=max_epochs, patience=patience,
        )
        lio_loader = torch.utils.data.DataLoader(
            RiderDataset(lio_test, lio_vocab),
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
        )
        _, lio_m, _ = run_epoch(model, lio_loader, loss_fn, use_site_intercept=False)
        lio_auc = lio_m["macro"]["roc_auc"]
        if np.isfinite(lio_auc):
            lio_aucs.append(lio_auc)

    if not lio_aucs:
        raise RuntimeError("No finite leave-intersection-out ROC-AUC values were produced.")
    return {
        "macro_roc_auc_mean": float(np.mean(lio_aucs)),
        "macro_roc_auc_std": float(np.std(lio_aucs)),
        "n_folds": len(lio_aucs),
        "fold_aucs": lio_aucs,
    }


def train(vocab, train_df, val_df, test_df,
          max_epochs=cfg.MAX_EPOCHS, patience=cfg.EARLY_STOPPING_PATIENCE,
          use_site_intercept_train=True, run_lio=False, encoders=None,
          split_seed=cfg.RANDOM_SEED, data_path=cfg.DATA_PATH):

    model   = TGMVMTGFNetV2(vocab)
    pos_weight = compute_pos_weights(train_df) if cfg.USE_POS_WEIGHT else None
    loss_fn = UncertaintyWeightedLoss(pos_weight=pos_weight)
    print(f"Model parameters: {count_parameters(model):,}")
    if pos_weight is not None:
        print("Positive class weights:", {
            task: round(float(weight), 3)
            for task, weight in zip(cfg.TASK_NAMES, pos_weight)
        })

    optimizer = optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY,
    )
    # Cosine LR scheduler — restarts every MAX_EPOCHS (single cycle)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

    train_loader, val_loader, test_loader = get_loaders(train_df, val_df, test_df, vocab)

    best_val_auc = -1.0
    patience_ctr = 0
    history = {"train_loss": [], "val_auc": [], "sigma": [], "alpha": []}

    print("\nStarting training...")
    for epoch in range(1, max_epochs + 1):
        anneal_temperature(model, epoch, max_epochs, schedule="cosine")

        tr_loss, tr_m, tr_tl = run_epoch(model, train_loader, loss_fn, optimizer,
                                          use_site_intercept=use_site_intercept_train)
        scheduler.step()
        vl_loss, vl_m, vl_tl = run_epoch(model, val_loader,   loss_fn,
                                          use_site_intercept=use_site_intercept_train)

        val_auc = vl_m["macro"]["roc_auc"]
        tau     = tr_tl.get("tau", ["-"] * cfg.NUM_TASKS)
        alpha   = torch.sigmoid(model.fusion.alpha_logit).item()

        history["train_loss"].append(tr_loss)
        history["val_auc"].append(val_auc)
        history["sigma"].append(tau)
        history["alpha"].append(alpha)

        if epoch % 10 == 0 or epoch == 1:
            sigma_str = ", ".join(f"{s:.3f}" for s in tau) if isinstance(tau, list) else str(tau)
            print(f"[Epoch {epoch:03d}] "
                  f"tr_loss={tr_loss:.4f} | val_AUC={val_auc:.4f} | "
                  f"tau=[{sigma_str}] | alpha={alpha:.3f}")

        # Early stopping
        if not np.isfinite(val_auc):
            patience_ctr += 1
            if patience_ctr >= patience:
                raise RuntimeError("Validation ROC-AUC stayed non-finite; cannot select a reliable checkpoint.")
            continue

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_ctr = 0
            torch.save(
                make_checkpoint_payload(
                    epoch, model, loss_fn, val_auc, vocab,
                    encoders=encoders, split_seed=split_seed, data_path=data_path),
                cfg.CHECKPOINT_PATH,
            )
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\nEarly stopping at epoch {epoch} (best val AUC: {best_val_auc:.4f})")
                break

    # Load best model and evaluate on test set
    if best_val_auc < 0:
        raise RuntimeError("No finite validation ROC-AUC was produced; checkpoint not saved.")
    ckpt = torch.load(cfg.CHECKPOINT_PATH, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    loss_fn.load_state_dict(ckpt["loss_state"])

    val_targets, val_probs, val_masks = collect_predictions(
        model, val_loader, use_site_intercept=use_site_intercept_train)
    thresholds = tune_task_thresholds(val_targets, val_probs, val_masks, metric="f1")
    print("Validation-tuned thresholds:", {
        task: round(float(threshold), 3)
        for task, threshold in thresholds.items()
    })

    torch.save(
        make_checkpoint_payload(
            ckpt["epoch"], model, loss_fn, ckpt["val_auc"], vocab,
            encoders=encoders, thresholds=thresholds,
            split_seed=split_seed, data_path=data_path),
        cfg.CHECKPOINT_PATH,
    )

    _, test_metrics_default, _ = run_epoch(
        model, test_loader, loss_fn,
        use_site_intercept=use_site_intercept_train)
    print_results(test_metrics_default, title="Test Results (Random Split, threshold=0.5)")

    _, test_metrics, _ = run_epoch(
        model, test_loader, loss_fn,
        use_site_intercept=use_site_intercept_train,
        thresholds=thresholds,
    )
    print_results(test_metrics, title="Test Results (Random Split, validation-tuned thresholds)")

    if run_lio:
        print("\n--- Leave-Intersection-Out Evaluation (fresh model per held-out site) ---")
        lio = evaluate_leave_intersection_out(pd.read_csv(cfg.DATA_PATH))
        print(
            "Leave-Intersection-Out macro ROC-AUC: "
            f"{lio['macro_roc_auc_mean']:.4f} ± {lio['macro_roc_auc_std']:.4f} "
            f"({lio['n_folds']} folds)"
        )

    return model, loss_fn, test_metrics, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-lio", action="store_true",
                        help="Run true leave-intersection-out evaluation after random-split training.")
    args = parser.parse_args()

    train_df, val_df, test_df, encoders, vocab = load_data()
    model, loss_fn, test_metrics, history = train(
        vocab, train_df, val_df, test_df,
        encoders=encoders,
        run_lio=args.run_lio,
    )
