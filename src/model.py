"""
Phase 6: Full M2G-Net v2 Model Assembly
Wires: ViewEncoders → CrossLevelInteraction → ResidualGatedFusion → OutputHeads
"""

import torch
import torch.nn as nn
import warnings

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg
from src.views       import build_encoders, SiteAwareEncoder
from src.interaction import CrossLevelInteraction
from src.fusion      import ResidualGatedFusion


class OutputHead(nn.Module):
    def __init__(self, input_dim=cfg.VIEW_DIM):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid())

    def forward(self, z):
        return self.head(z).squeeze(-1)   # (batch,)


class TGMVMTGFNetV2(nn.Module):
    """
    Full v2 model:
      Individual Encoders (role, traits)
      Contextual Encoders (road, env, site)
      CrossLevelInteraction(h_ind, h_ctx)
      ResidualGatedFusion(views + h_inter)
      OutputHeads × 4
    """

    def __init__(self, vocab):
        super().__init__()
        d = cfg.VIEW_DIM

        # ── Encoders ──────────────────────────────────────────────────────────
        self.encoders = build_encoders(vocab)

        # ── Cross-Level Interaction ────────────────────────────────────────────
        ind_dim = d * len(cfg.INDIVIDUAL_VIEWS)   # role + traits
        ctx_dim = d * len(cfg.CONTEXTUAL_VIEWS)   # road + env + site
        self.interaction = CrossLevelInteraction(ind_dim, ctx_dim,
                                                  output_dim=cfg.INTERACTION_DIM)

        # ── Gated Fusion ───────────────────────────────────────────────────────
        # gate_input_dim: all view_reps + h_inter projected to same d
        # We project h_inter to VIEW_DIM inside fusion for simplicity
        self.fusion = ResidualGatedFusion(
            num_tasks      = cfg.NUM_TASKS,
            num_gate_inputs= cfg.NUM_GATE_INPUTS,   # 5 views + 1 interaction
            gate_input_dim = d,
            total_view_dim = d * cfg.NUM_VIEWS,     # for early MLP
        )

        # ── Output heads ──────────────────────────────────────────────────────
        self.heads = nn.ModuleList([OutputHead(d) for _ in range(cfg.NUM_TASKS)])

    def forward(self, views, use_site_intercept=True):
        """
        views: dict with keys: rider_role, rider_traits, road_context,
                               environment, site_obs, site_id
        returns:
          predictions  : list of (batch,) tensors
          gate_weights : list of (batch, num_gate_inputs) tensors per task
          alpha_blend  : scalar — ratio of gated vs early fusion
        """
        # ── Encode each view ──────────────────────────────────────────────────
        h_role  = self.encoders["rider_role"](views["rider_role"])
        h_trait = self.encoders["rider_traits"](views["rider_traits"])
        h_road  = self.encoders["road_context"](views["road_context"])
        h_env   = self.encoders["environment"](views["environment"])
        h_site  = self.encoders["site"](views["site_obs"], views["site_id"],
                                        use_intercept=use_site_intercept)

        # ── Cross-level interaction ───────────────────────────────────────────
        h_ind  = torch.cat([h_role, h_trait],       dim=-1)   # Individual
        h_ctx  = torch.cat([h_road, h_env, h_site], dim=-1)   # Contextual
        h_inter = self.interaction(h_ind, h_ctx)

        # ── Gated fusion ──────────────────────────────────────────────────────
        view_reps_5 = [h_role, h_trait, h_road, h_env, h_site]
        fused, gate_weights, alpha_blend = self.fusion(view_reps_5, h_inter)

        # ── Output heads ──────────────────────────────────────────────────────
        predictions = [self.heads[k](fused[k]) for k in range(cfg.NUM_TASKS)]

        return predictions, gate_weights, alpha_blend

    def load_state_dict(self, state_dict, strict=True):
        """Pad older checkpoints that did not reserve an unknown-site embedding row."""
        key = "encoders.site.site_intercept.weight"
        if key in state_dict:
            current = self.state_dict()[key]
            saved = state_dict[key]
            if saved.shape != current.shape and saved.ndim == current.ndim and saved.shape[1:] == current.shape[1:]:
                warnings.warn(
                    "Padding site_intercept weights from an older checkpoint. "
                    "Verify checkpoint preprocessing metadata before trusting site-specific metrics.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                adapted = current.clone()
                rows = min(saved.shape[0], current.shape[0])
                adapted[:rows] = saved[:rows]
                state_dict = dict(state_dict)
                state_dict[key] = adapted
        return super().load_state_dict(state_dict, strict=strict)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick smoke test
    import sys
    sys.path.insert(0, ".")
    from src.data_pipeline import load_data, get_loaders
    train_df, val_df, test_df, encoders, vocab = load_data()
    model = TGMVMTGFNetV2(vocab)
    print(f"Total parameters: {count_parameters(model):,}")
    loader, _, _ = get_loaders(train_df, val_df, test_df, vocab, batch_size=8)
    views, targets, masks = next(iter(loader))
    preds, gates, alpha = model(views)
    print("Predictions:", [p.shape for p in preds])
    print("Gate weights:", [g.shape for g in gates])
    print("Alpha blend:", alpha.item())
