# Architecture change tried: direct interaction skip (dev validation only, seeds 1701-1710, training seed 42)

Change (declared before running): logit_k += s_k^T h_int, s_k zero-initialised, bypassing gate and blend.
Code: `src/model.py` (`interaction_skip`, default False), `scripts/train.py`, `scripts/evaluate/paper_pipeline.py`, `config.V3_INTERACTION_SKIP` (False).
Math: `docs/MATHEMATICAL_FORMULAS.md` section 6. Default model is bitwise unchanged (checked on 3 ablations).
Checks: `tests/test_gate_prior_smoothing.py` + 15 related tests pass (16 total). Skip formula checked with numbers (logit 0.1 + skip 0.8 -> 0.7109).
Data: synthetic; validation split only; no test split, no LOCKED seeds. Raw rows: `dev-variants-skip.csv`.

Result (5-member ensemble, BCE; mean macro-AUC paired difference, positive seeds of 10):

| Contrast | additive | interaction | partial |
|---|---|---|---|
| SKIP5 - M5 | -0.0008 (3) | -0.0005 (4) | -0.0005 (4) |
| SKIP5 - m2g_net | +0.0047 (8) | +0.0028 (6) | +0.0040 (9) |
| M5 - m2g_net | +0.0055 (8) | +0.0033 (6) | +0.0045 (9) |

Validation Brier: SKIP5 equals M5 within 0.0002.
Decision: skip does not help (slightly negative, within noise). Not adopted; `V3_INTERACTION_SKIP` stays False; FREEZE.json unchanged.
Consistent with ARCH-HEADROOM.md: no architecture headroom at interaction/partial with ~2400 training rows.
