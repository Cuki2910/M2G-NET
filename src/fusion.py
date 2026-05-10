"""
Phase 4: Regularized Task-Specific Gated Fusion + Residual Early Fusion
- RegularizedTaskGate: softmax with temperature annealing + uniform prior (avoids mode collapse)
- ResidualGatedFusion: z_t = alpha * z_gated + (1-alpha) * z_early  (learnable alpha)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg


class RegularizedTaskGate(nn.Module):
    """
    One gate per task.
    Adds a uniform prior to logits and divides by a learnable temperature
    to prevent mode collapse.
    """

    def __init__(self, num_inputs, input_dim,
                 temperature_init=cfg.TEMPERATURE_INIT):
        super().__init__()
        self.gate        = nn.Linear(num_inputs * input_dim, num_inputs)
        self.temperature = nn.Parameter(torch.tensor(float(temperature_init)))
        # Fixed uniform prior (not learnable) — encourages balanced initial attention
        self.register_buffer("prior", torch.ones(num_inputs) / num_inputs)

    def forward(self, view_reps):
        """
        view_reps: list of (batch, dim) tensors — same dim for all inputs
        returns alpha: (batch, num_inputs)
        """
        h = torch.cat(view_reps, dim=-1)                            # (batch, num_inputs*dim)
        logits = self.gate(h) + cfg.GATE_PRIOR_WEIGHT * self.prior  # add uniform prior
        temp   = self.temperature.clamp(min=0.1)                    # avoid division by zero
        return F.softmax(logits / temp, dim=-1)                     # (batch, num_inputs)


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
            RegularizedTaskGate(num_gate_inputs, gate_input_dim)
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
