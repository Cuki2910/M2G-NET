"""
Phase 3: Cross-Level Interaction
Learns interactions between Individual-level and Contextual-level representations.
h_int = psi(p_ind + p_ctx + W_inter(p_ind * p_ctx))   [MATHEMATICAL_FORMULAS.md §3]
"""

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg


class CrossLevelInteraction(nn.Module):
    """
    Computes h_int in R^d capturing individual x contextual interactions.

    Formula (MATHEMATICAL_FORMULAS.md §3):
        p_ind   = W_ind * h_ind          W_ind in R^(d x 2d)
        p_ctx   = W_ctx * h_ctx          W_ctx in R^(d x 3d)
        h_int   = psi(p_ind + p_ctx + W_inter(p_ind o p_ctx))
    where o is element-wise multiplication and psi: R^d -> R^d is GELU.
    """

    def __init__(self, ind_dim, ctx_dim, output_dim=cfg.INTERACTION_DIM,
                 dropout=cfg.DROPOUT_RATE):
        super().__init__()
        proj_dim = output_dim
        self.proj_ind = nn.Linear(ind_dim, proj_dim)   # W_ind: R^(d x 2d)
        self.proj_ctx = nn.Linear(ctx_dim, proj_dim)   # W_ctx: R^(d x 3d)
        self.W_inter  = nn.Linear(proj_dim, proj_dim)  # W_int: R^(d x d)
        self.out      = nn.Sequential(
            nn.Linear(proj_dim, output_dim),
            nn.GELU(),           # outer nonlinearity psi: R^d -> R^d
            nn.Dropout(dropout),
        )

    def forward(self, h_ind, h_ctx):
        """
        h_ind: (batch, ind_dim)
        h_ctx: (batch, ctx_dim)
        returns h_int: (batch, output_dim)
        """
        p_ind = self.proj_ind(h_ind)                          # (batch, d)
        p_ctx = self.proj_ctx(h_ctx)                          # (batch, d)
        combined = p_ind + p_ctx + self.W_inter(p_ind * p_ctx)  # three-term sum §3
        return self.out(combined)
