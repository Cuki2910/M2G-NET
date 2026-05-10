"""
Phase 8: Training Loop
- Adam optimizer with L2 regularization
- Temperature annealing for gate regularization
- Early stopping on macro ROC-AUC
- Saves best checkpoint
- Supports Leave-Intersection-Out evaluation
"""

import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import config as cfg
from src.data_pipeline import load_data, get_loaders, leave_intersection_out_split, RiderDataset
from src.model         import TGMVMTGFNetV2, count_parameters
from src.loss          import UncertaintyWeightedLoss
from src.metrics       import compute_all_metrics, print_results


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


def run_epoch(model, loader, loss_fn, optimizer=None, use_site_intercept=True):
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
    metrics     = compute_all_metrics(all_targets, all_probs, all_masks)
    avg_loss    = total_loss / max(n_batches, 1)
    return avg_loss, metrics, task_losses


def train(vocab, train_df, val_df, test_df,
          max_epochs=cfg.MAX_EPOCHS, patience=cfg.EARLY_STOPPING_PATIENCE,
          use_site_intercept_train=True):

    model   = TGMVMTGFNetV2(vocab)
    loss_fn = UncertaintyWeightedLoss()
    print(f"Model parameters: {count_parameters(model):,}")

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
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_ctr = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "loss_state":  loss_fn.state_dict(),
                "val_auc":     val_auc,
            }, cfg.CHECKPOINT_PATH)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\nEarly stopping at epoch {epoch} (best val AUC: {best_val_auc:.4f})")
                break

    # Load best model and evaluate on test set
    ckpt = torch.load(cfg.CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    loss_fn.load_state_dict(ckpt["loss_state"])

    _, test_metrics, _ = run_epoch(model, test_loader, loss_fn,
                                    use_site_intercept=use_site_intercept_train)
    print_results(test_metrics, title="Test Results (Random Split)")

    # Leave-intersection-out evaluation
    print("\n--- Leave-Intersection-Out Evaluation ---")
    lio_aucs = []
    import pandas as pd
    full_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    for site_id in sorted(full_df[cfg.SITE_ID_COL].unique()):
        lio_train, lio_test = leave_intersection_out_split(full_df, site_id)
        if len(lio_test) < 5:
            continue
        lio_ds  = RiderDataset(lio_test, vocab)
        from torch.utils.data import DataLoader
        lio_loader = DataLoader(lio_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
        # Evaluate with intercept OFF (unseen site)
        _, lio_m, _ = run_epoch(model, lio_loader, loss_fn, use_site_intercept=False)
        lio_aucs.append(lio_m["macro"]["roc_auc"])
    print(f"Leave-Intersection-Out macro ROC-AUC: {np.mean(lio_aucs):.4f} ± {np.std(lio_aucs):.4f}")

    return model, loss_fn, test_metrics, history


if __name__ == "__main__":
    train_df, val_df, test_df, encoders, vocab = load_data()
    model, loss_fn, test_metrics, history = train(vocab, train_df, val_df, test_df)
