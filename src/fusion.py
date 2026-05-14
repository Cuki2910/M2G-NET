"""
Phase 4: Regularized Task-Specific Gated Fusion + Residual Early Fusion
- RegularizedTaskGate: sparsemax with temperature scaling and uniform prior mixing
- ResidualGatedFusion: z_k = sigmoid(a) * z_gated^(k) + (1-sigmoid(a)) * z_early  (learnable a)
"""

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg


def sparsemax(z, dim=-1):
    """
    Sparsemax implementation.
    Projects onto the probability simplex, yielding sparse probability distributions.
    """
    original_size = z.size()
    z_2d = z.view(-1, original_size[dim])
    
    z_sorted, _ = torch.sort(z_2d, dim=1, descending=True)
    cssv = torch.cumsum(z_sorted, dim=1) - 1
    k = torch.arange(1, z_sorted.size(1) + 1, device=z.device, dtype=z.dtype)
    cond = z_sorted - cssv / k > 0
    k_val = torch.sum(cond, dim=1, keepdim=True)
    
    tau = torch.gather(cssv, 1, k_val.long() - 1) / k_val
    p = torch.clamp(z_2d - tau, min=0.0)
    
    return p.view(original_size)

class RegularizedTaskGate(nn.Module):
    """
    One gate per task.
    Divides logits by a learnable temperature to control attention sharpness.
    Applies uniform prior smoothing: alpha = (1-lam)*sparsemax(logits/T) + lam/K
    This reduces cross-seed variance (gate instability). When lam > 0, exact
    sparse zeros become a positive lower bound of lam/K.
    """

    def __init__(self, num_inputs, input_dim,
                 temperature_init=cfg.TEMPERATURE_INIT,
                 prior_weight=cfg.GATE_PRIOR_WEIGHT):
        super().__init__()
        self.gate         = nn.Linear(num_inputs * input_dim, num_inputs)
        # T is declared as nn.Parameter for checkpoint compatibility but is NOT
        # gradient-learned. scripts/train.py::anneal_temperature() overwrites
        # .data each epoch via cosine schedule (TEMPERATURE_INIT -> TEMPERATURE_FINAL).
        self.temperature  = nn.Parameter(torch.tensor(float(temperature_init)))
        self.prior_weight = float(prior_weight)
        self.num_inputs   = num_inputs

    @property
    def prior_weight(self):
        return self._prior_weight

    @prior_weight.setter
    def prior_weight(self, value):
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("prior_weight must be in the range [0, 1].")
        self._prior_weight = value

    def forward(self, view_reps):
        """
        view_reps: list of (batch, dim) tensors — same dim for all inputs
        returns alpha: (batch, num_inputs)
        """
        h = torch.cat(view_reps, dim=-1)                            # (batch, num_inputs*dim)
        logits       = self.gate(h)
        temp         = self.temperature.clamp(min=0.1)
        alpha_sparse = sparsemax(logits / temp, dim=-1)             # (batch, num_inputs)
        # Uniform prior smoothing: pulls extreme sparse weights toward 1/K,
        # reducing cross-seed variance while keeping relative importance ordering.
        K     = logits.size(-1)
        alpha = (1.0 - self.prior_weight) * alpha_sparse + self.prior_weight / K
        return alpha


class ResidualGatedFusion(nn.Module):
    """
    Full fusion module:
      1. Task-specific gated fusion over {h_views + h_inter}
      2. Simple early fusion MLP over {h_views only}
      3. Blend: z_t = sigmoid(alpha) * z_gated + (1-sigmoid(alpha)) * z_early
    """

    def __init__(self, num_tasks, num_gate_inputs, gate_input_dim,
                 total_view_dim, view_dim=cfg.VIEW_DIM,
                 dropout=cfg.DROPOUT_RATE):
        super().__init__()
        self.num_gate_inputs = num_gate_inputs

        # Task-specific gates (operate on 5 views + interaction = num_gate_inputs)
        self.gates = nn.ModuleList([
            RegularizedTaskGate(num_gate_inputs, gate_input_dim,
                                prior_weight=cfg.GATE_PRIOR_WEIGHT)
            for _ in range(num_tasks)
        ])

        # Early fusion MLP (concat of 5 original views only)
        self.early_mlp = nn.Sequential(
            nn.Linear(total_view_dim, view_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(view_dim * 2, view_dim),
        )

        # Projection for h_inter to match gate_input_dim
        self.inter_proj = nn.Linear(cfg.INTERACTION_DIM, gate_input_dim)

        # Learnable blend parameter: alpha=1 → pure gated, alpha=0 → pure early
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))   # sigmoid(0) = 0.5

    def forward(self, view_reps_5, h_inter):
        """
        view_reps_5: list of 5 (batch, d) tensors for the 5 views
        h_inter:     (batch, d_inter) cross-level interaction term
        returns:
          fused     : list of num_tasks tensors, each (batch, d)
          gate_weights: list of num_tasks tensors, each (batch, num_gate_inputs)
          alpha_blend : scalar tensor (blend ratio)
        """
        # Early fusion path
        z_early = self.early_mlp(torch.cat(view_reps_5, dim=-1))   # (batch, d)

        # Gated fusion path: gate and weighted sum both use views + interaction.
        # Project h_inter to match view dimension for uniform gate input
        h_inter_proj = self.inter_proj(h_inter)                    # (batch, gate_input_dim)
        gate_inputs_proj = view_reps_5 + [h_inter_proj]
        H = torch.stack(gate_inputs_proj, dim=1)   # (batch, 6, d)

        alpha_blend = torch.sigmoid(self.alpha_logit)

        fused, gate_weights = [], []
        for gate in self.gates:
            a = gate(gate_inputs_proj)              # (batch, num_gate_inputs)
            z_gated = (H * a.unsqueeze(-1)).sum(dim=1)   # (batch, d)
            z = alpha_blend * z_gated + (1 - alpha_blend) * z_early
            fused.append(z)
            gate_weights.append(a)

        return fused, gate_weights, alpha_blend
