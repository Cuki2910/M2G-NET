# Dev screening of m2g_net_v3 variants (validation split only, generator seeds 1701-1710, training seed 42)

Declared before running: M3, M5 (= frozen v3), M10 (ensemble size), MIX5 (5 members, gate prior cycled 0.0/0.1/0.2/0.0/0.1).
All use BCE (no focal). Script: `scripts/run_v3_variants_dev.py`; raw rows: `dev-variants.csv`. No test split, no LOCKED seeds read.
These are validation-split numbers, synthetic data; not evidence for the LOCKED criteria.

Mean macro-AUC paired difference vs m2g_net (positive seeds of 10):

| Variant | additive | interaction | partial |
|---|---|---|---|
| M3 | +0.0053 (8) | +0.0011 (6) | +0.0024 (8) |
| M5 (frozen) | +0.0055 (8) | +0.0033 (6) | +0.0045 (9) |
| M10 | +0.0059 (8) | +0.0032 (6) | +0.0059 (10) |
| MIX5 | +0.0054 (8) | +0.0035 (6) | +0.0057 (10) |

Validation Brier: every variant ~0.189-0.219, always below early_fusion_mlp and catboost.

Reading: gain saturates at ~5 members; no variant gets interaction above +0.0035 vs m2g_net (criterion needs mean > +0.005 and >=18/20 sets).
Variant spread (<=0.0014 between M5/M10/MIX5) is within paired noise (~0.001-0.003), so no variant is selected over frozen M5.
Decision: keep M5, FREEZE.json unchanged.
