"""
Phase 2: View-Specific Encoders
Each view gets its own small MLP.  SiteAwareEncoder combines
observed infrastructure features with a regularized random intercept.
"""

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg


class ViewEncoder(nn.Module):
    """
    Base encoder: multiple categorical features → embeddings → concat → small MLP → h ∈ R^d
    """

    def __init__(self, vocab_sizes_list, emb_dim=cfg.EMBEDDING_DIM,
                 hidden_dim=cfg.HIDDEN_DIM, output_dim=cfg.VIEW_DIM,
                 dropout=cfg.DROPOUT_RATE):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(n, emb_dim) for n in vocab_sizes_list
        ])
        in_dim = emb_dim * len(vocab_sizes_list)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # x: (batch, num_features)  — LongTensor
        embs = [self.embeddings[i](x[:, i]) for i in range(x.shape[1])]
        h = torch.cat(embs, dim=-1)
        return self.mlp(h)


class SiteAwareEncoder(nn.Module):
    """
    V2 Site encoder: observed infra features + regularized random intercept.
    At inference on unseen sites, call forward(obs, site_id, use_intercept=False).
    """

    def __init__(self, obs_vocab_sizes, num_sites,
                 emb_dim=cfg.EMBEDDING_DIM, hidden_dim=cfg.HIDDEN_DIM,
                 output_dim=cfg.VIEW_DIM, dropout=cfg.DROPOUT_RATE):
        super().__init__()
        # (a) Observed site feature encoder
        self.obs_encoder = ViewEncoder(obs_vocab_sizes, emb_dim, hidden_dim, output_dim, dropout)
        # (b) Site random intercept (regularized via weight_decay in optimizer)
        self.site_intercept = nn.Embedding(num_sites, output_dim)
        nn.init.normal_(self.site_intercept.weight, std=0.01)
        # Projection to merge (a) and (b)
        self.proj = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
        )

    def forward(self, site_obs, site_id, use_intercept=True):
        h_obs = self.obs_encoder(site_obs)                          # (batch, d)
        if use_intercept:
            h_rand = self.site_intercept(site_id)                   # (batch, d)
        else:
            h_rand = torch.zeros_like(h_obs)                        # zero-out for unseen sites
        return self.proj(torch.cat([h_obs, h_rand], dim=-1))        # (batch, d)


def build_encoders(vocab):
    """
    Instantiate all view encoders from vocab dict.
    Returns nn.ModuleDict with keys matching view names.
    """
    rider_role_enc   = ViewEncoder([vocab["rider_type"]])
    rider_traits_enc = ViewEncoder([vocab["gender"], vocab["age_group"]])
    road_ctx_enc     = ViewEncoder([
        vocab["police_presence"], vocab["traffic_condition"],
        vocab["num_lanes"], vocab["has_signal"]
    ])
    env_enc = ViewEncoder([vocab["weather"], vocab["time_slot"], vocab["weekend"]])
    site_enc = SiteAwareEncoder(
        obs_vocab_sizes=[vocab["intersection_type"]],
        num_sites=vocab["num_sites"],
    )

    return nn.ModuleDict({
        "rider_role":   rider_role_enc,
        "rider_traits": rider_traits_enc,
        "road_context": road_ctx_enc,
        "environment":  env_enc,
        "site":         site_enc,
    })
