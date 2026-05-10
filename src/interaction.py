"""
Phase 3: Cross-Level Interaction
Learns interactions between Individual-level and Contextual-level representations.
h_inter = ReLU(W1*h_ind + W2*h_ctx + W3*(h_ind * h_ctx) + b)
"""

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg


class CrossLevelInteraction(nn.Module):
    """
    Computes h_inter ∈ R^d_inter capturing individual × contextual interactions.
    """

    def __init__(self, ind_dim, ctx_dim, output_dim=cfg.INTERACTION_DIM,
                 dropout=cfg.DROPOUT_RATE):
        super().__init__()
        proj_dim = output_dim
        self.proj_ind = nn.Linear(ind_dim, proj_dim)
        self.proj_ctx = nn.Linear(ctx_dim, proj_dim)
        self.W_inter  = nn.Linear(proj_dim, proj_dim)
        self.out      = nn.Sequential(
            nn.Linear(proj_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, h_ind, h_ctx):
        """
        h_ind: (batch, ind_dim)
        h_ctx: (batch, ctx_dim)
        returns h_inter: (batch, output_dim)
        """
        p_ind = self.proj_ind(h_ind)        # (batch, proj_dim)
        p_ctx = self.proj_ctx(h_ctx)        # (batch, proj_dim)
        element_wise = p_ind * p_ctx         # interaction term
        combined = p_ind + p_ctx + self.W_inter(element_wise)
        return self.out(combined)
