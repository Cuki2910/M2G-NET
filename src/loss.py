"""
Phase 5: Masked uncertainty-weighted multi-task loss.

Supports partially observed task labels via a task-observation mask m_ik.
The uncertainty term follows Kendall et al. (2018), parameterized as
eta_k = log(tau_k^2) for numerical stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg


def focal_loss(pred, target, alpha=cfg.FOCAL_ALPHA, gamma=cfg.FOCAL_GAMMA):
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    pt  = torch.where(target == 1, pred, 1 - pred)
    fl  = alpha * (1 - pt) ** gamma * bce
    return fl


def masked_mean(values, mask):
    """Average values over observed labels only."""
    mask = mask.to(values.dtype)
    denom = mask.sum()
    if denom.item() == 0:
        return None
    return (values * mask).sum() / denom


class UncertaintyWeightedLoss(nn.Module):
    """
    Learnable per-task uncertainty balances task losses.
    After training, exp(0.5 * log_tau_sq) gives tau_k (task uncertainty).
    """

    def __init__(self, num_tasks=cfg.NUM_TASKS, use_focal=cfg.USE_FOCAL_LOSS):
        super().__init__()
        # Parameterize log(tau^2) for numerical stability.
        self.log_tau_sq = nn.Parameter(torch.zeros(num_tasks))
        self.use_focal    = use_focal
        self.num_tasks    = num_tasks

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Accept checkpoints saved before sigma notation was renamed to tau."""
        old_key = prefix + "log_sigma_sq"
        new_key = prefix + "log_tau_sq"
        if old_key in state_dict and new_key not in state_dict:
            state_dict[new_key] = state_dict.pop(old_key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(self, predictions, targets, masks=None):
        """
        predictions: list of num_tasks tensors, each (batch,)
        targets:     (batch, num_tasks) float tensor
        masks:       optional (batch, num_tasks) float tensor, where 1 means
                     the label is observed/applicable and 0 means missing.
        returns: total_loss (scalar), per_task_losses (dict)
        """
        if masks is None:
            masks = torch.ones_like(targets)

        total_loss = predictions[0].sum() * 0.0
        task_losses = {}

        for k, task_name in enumerate(cfg.TASK_NAMES):
            pred_k   = predictions[k]                                    # (batch,)
            target_k = targets[:, k]                                     # (batch,)
            mask_k   = masks[:, k]                                       # (batch,)

            if self.use_focal:
                per_sample_loss = focal_loss(pred_k, target_k)
            else:
                per_sample_loss = F.binary_cross_entropy(pred_k, target_k, reduction="none")

            loss_k = masked_mean(per_sample_loss, mask_k)
            if loss_k is None:
                task_losses[task_name] = float("nan")
                continue

            precision   = torch.exp(-self.log_tau_sq[k])                # 1/tau^2
            total_loss += 0.5 * precision * loss_k + 0.5 * self.log_tau_sq[k]
            task_losses[task_name] = loss_k.item()

        task_losses["tau"] = torch.exp(0.5 * self.log_tau_sq).detach().tolist()
        return total_loss, task_losses
