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
from src.model         import TGMVMTGFNetV2
from src.loss          import UncertaintyWeightedLoss
from src.metrics       import compute_all_metrics, print_results
from src.calibration import TemperatureScaler, fit_validation_rule, prediction_metrics



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
    model.eval()
    all_probs, all_targets, all_masks = [], [], []
    with torch.no_grad():
        for views, targets, masks in loader:
            preds, _, _ = model(views, use_site_intercept=use_site_intercept)
            all_probs.append(torch.stack(preds, dim=1).detach().numpy())
            all_targets.append(targets.numpy())
            all_masks.append(masks.numpy())
    return np.vstack(all_targets), np.vstack(all_probs), np.vstack(all_masks)


def run_epoch(model, loader, loss_fn, optimizer=None, use_site_intercept=True,
              thresholds=None, temperature=1.0):
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
    if temperature != 1.0:
        scaler = TemperatureScaler()
        scaler.temperature = temperature
        all_probs = scaler.transform(all_probs)
    metrics     = compute_all_metrics(all_targets, all_probs, all_masks, thresholds=thresholds)
    avg_loss    = total_loss / max(n_batches, 1)
    return avg_loss, metrics, task_losses


def fit_model_for_split(vocab, train_df, val_df, encoders=None,
                        max_epochs=cfg.MAX_EPOCHS, patience=cfg.EARLY_STOPPING_PATIENCE,
                        checkpoint_path=None, ablation="full", use_uncertainty=cfg.USE_UNCERTAINTY_WEIGHTING,
                        gate_prior_weight=None, use_site_intercept=True,
                        split_seed=cfg.RANDOM_SEED, data_path=cfg.DATA_PATH,
                        label_policy=cfg.LABEL_POLICY, validation_use_site_intercept=None,
                        use_focal=cfg.USE_FOCAL_LOSS, interaction_skip=False, gate_variant="legacy",
                        diagnostics=None, weight_decay=cfg.WEIGHT_DECAY):
    """Train a fresh model for one already-encoded split and return the best weights."""
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive.")
    if (isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float, np.number))
            or not np.isrealobj(weight_decay) or not np.isfinite(weight_decay) or weight_decay < 0):
        raise ValueError("weight_decay must be finite and nonnegative.")
    if validation_use_site_intercept is None:
        validation_use_site_intercept = use_site_intercept
    if diagnostics is not None:
        if not isinstance(diagnostics, dict) or diagnostics:
            raise ValueError("diagnostics must be an empty dict or None.")
        diagnostics["history"] = []
    np.random.seed(split_seed)
    torch.manual_seed(split_seed)
    model = TGMVMTGFNetV2(vocab, ablation=ablation, gate_prior_weight=gate_prior_weight,
                          interaction_skip=interaction_skip, gate_variant=gate_variant)
    pos_weight = compute_pos_weights(train_df) if cfg.USE_POS_WEIGHT else None
    loss_fn = UncertaintyWeightedLoss(pos_weight=pos_weight, use_uncertainty=use_uncertainty, use_focal=use_focal)
    optimizer = optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=cfg.LEARNING_RATE, weight_decay=weight_decay,
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
        _, train_metrics, _ = run_epoch(model, train_loader, loss_fn, optimizer, use_site_intercept=use_site_intercept)
        scheduler.step()
        _, val_metrics, _ = run_epoch(model, val_loader, loss_fn, use_site_intercept=validation_use_site_intercept)
        if diagnostics is not None:
            diagnostics["history"].append({"epoch": epoch,
                                            "training_metrics_dropout": copy.deepcopy(train_metrics),
                                            "validation_metrics": copy.deepcopy(val_metrics)})
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
                os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
                torch.save(
                    make_checkpoint_payload(
                        epoch, model, loss_fn, val_auc, vocab, encoders=encoders,
                        split_seed=split_seed, data_path=data_path, label_policy=label_policy),
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
    if diagnostics is not None:
        # Sequential DataLoader iteration also draws a base seed; telemetry must not change RNG.
        numpy_state = np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                train_eval_loader = torch.utils.data.DataLoader(
                    train_loader.dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
                y_train, p_train, m_train = collect_predictions(model, train_eval_loader, use_site_intercept)
                y_val, p_val, m_val = collect_predictions(model, val_loader, validation_use_site_intercept)
                diagnostics["best"] = {
                    "epoch": best_epoch,
                    "train_metrics": compute_all_metrics(y_train, p_train, m_train),
                    "validation_metrics": compute_all_metrics(y_val, p_val, m_val),
                }
        finally:
            np.random.set_state(numpy_state)
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
    from src.metrics import lso_summary
    summary = lso_summary(lio_aucs)
    return {
        "macro_roc_auc_mean": summary["G_bar_LSO"],
        "macro_roc_auc_std":  summary["s_G"],
        "n_folds":            summary["n_folds"],
        "fold_aucs":          lio_aucs,
    }


def train(vocab, train_df, val_df, test_df,
          max_epochs=cfg.MAX_EPOCHS, patience=cfg.EARLY_STOPPING_PATIENCE,
          use_site_intercept_train=True, run_lio=False, encoders=None,
          split_seed=cfg.RANDOM_SEED, data_path=cfg.DATA_PATH,
          checkpoint_path=None, label_policy=cfg.LABEL_POLICY):
    model, loss_fn, val_auc, epoch = fit_model_for_split(
        vocab, train_df, val_df, encoders=encoders,
        max_epochs=max_epochs, patience=patience, split_seed=split_seed,
        use_site_intercept=use_site_intercept_train,
    )
    _, val_loader, test_loader = get_loaders(train_df, val_df, test_df, vocab)
    targets, probs, masks = collect_predictions(model, val_loader, use_site_intercept_train)
    scaler, thresholds = fit_validation_rule(targets, probs, masks)
    split_rows = None
    if all("_row_id" in frame for frame in (train_df, val_df, test_df)):
        split_rows = {key: frame["_row_id"].tolist() for key, frame in
                      zip(("train", "validation", "test"), (train_df, val_df, test_df))}
    checkpoint_path = checkpoint_path or cfg.CHECKPOINT_PATH
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    torch.save(make_checkpoint_payload(
        epoch, model, loss_fn, val_auc, vocab, encoders=encoders,
        thresholds=thresholds, temperature=scaler.temperature, split_rows=split_rows,
        split_seed=split_seed, data_path=data_path, label_policy=label_policy,
    ), checkpoint_path)
    targets, probs, masks = collect_predictions(model, test_loader, use_site_intercept_train)
    reports = prediction_metrics(targets, probs, masks, scaler, thresholds)
    for rule, metrics in reports.items():
        print_results(metrics, title=f"Development random-split test ({rule})")
    if run_lio:
        raw = pd.read_csv(data_path)
        from src.data_pipeline import prepare_labels
        print(evaluate_leave_intersection_out(prepare_labels(raw, label_policy)))
    return model, loss_fn, reports["calibrated_tuned"], {"best_epoch": epoch, "val_auc": val_auc}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-lio", action="store_true",
                        help="Run true leave-intersection-out evaluation after random-split training.")
    parser.add_argument("--label-policy", choices=["strict", "mask"], default=cfg.LABEL_POLICY)
    parser.add_argument("--data", default=cfg.DATA_PATH)
    parser.add_argument("--max-epochs", type=int, default=cfg.MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=cfg.EARLY_STOPPING_PATIENCE)
    args = parser.parse_args()

    train_df, val_df, test_df, encoders, vocab = load_data(data_path=args.data, label_policy=args.label_policy)
    model, loss_fn, test_metrics, history = train(
        vocab, train_df, val_df, test_df,
        encoders=encoders,
        run_lio=args.run_lio,
        max_epochs=args.max_epochs, patience=args.patience,
        data_path=args.data, label_policy=args.label_policy,
    )
