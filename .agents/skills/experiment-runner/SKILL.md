---
name: experiment-runner
description: "Standardized M2G-NET experiment pipeline: data check → train → evaluate → save results to docs/. Use when running training, ablation, baseline comparison, gate stability, or any multi-phase experiment on M2G-NET."
tags: [m2g-net, experiment, training, evaluation, pipeline]
version: 1.0.0
user-invocable: true
allowed-tools: [Bash, Read, Write, Edit, Glob]
---

# Experiment Runner — M2G-NET

Standardized workflow for running and recording M2G-NET experiments. Ensures reproducibility and consistent result logging.

## When to Use

- Running or resuming training (`scripts/train.py`)
- Running ablation study (`scripts/evaluate/ablation.py`)
- Comparing against baselines (`baselines/run_all_baselines.py`)
- Gate stability / prior sweep analysis
- Any multi-step experiment that produces results to report

---

## Phase 0 — Pre-flight Checks

Before running any experiment, verify:

```bash
# 1. Data integrity
python scripts/check_data.py

# 2. Tests pass
python -m pytest tests/ -q

# 3. Review config
# Open config.py — confirm paths, VIEW_COLS, hyperparameters match intent
```

**Stop if any check fails.** Do not proceed with stale data or broken tests.

---

## Phase 1 — Training

```bash
python scripts/train.py
```

**What to watch:**
- Early stopping trigger (macro ROC-AUC plateau)
- Gate temperature annealing logs
- Checkpoint saved to `checkpoints/`

**After training:**
- Note checkpoint filename and final metric values
- Update `.claude/pipeline-state.json` if multi-phase run

---

## Phase 2 — Core Evaluation

Run in this order (each depends on a trained checkpoint):

| Script | Purpose | Output |
|--------|---------|--------|
| `scripts/evaluate/temperature_scaling.py` | Calibrate model | calibration curve |
| `scripts/evaluate/ablation.py` | View/task ablation | ablation table |
| `baselines/run_all_baselines.py` | XGBoost, LightGBM | baseline table |
| `scripts/evaluate/gate_stability.py` | Gate prior sensitivity | stability report |
| `scripts/evaluate/repeated_runs_significance.py` | Statistical significance | p-values, CI |

---

## Phase 3 — Extended Evaluation (optional)

| Script | When to run |
|--------|------------|
| `scripts/evaluate/leave_site_out.py` | Cross-site generalization |
| `scripts/evaluate/gate_prior_cv_sweep.py` | Hyperparameter sensitivity |
| `scripts/evaluate/fairness_calibration.py` | Fairness analysis |
| `scripts/evaluate/threshold_tuning.py` | Decision threshold optimization |
| `scripts/evaluate/test_on_independent_set.py` | Final held-out evaluation |

---

## Phase 4 — Save Results

All experiment results go to `docs/`. Use this naming convention:

```
docs/<EXPERIMENT_NAME>_RESULTS.md
```

**Minimum content per results file:**
1. Experiment goal (1 sentence)
2. Config snapshot (key hyperparams from `config.py`)
3. Metric table (M2G-NET vs. baselines if applicable)
4. Key findings (2–3 bullets)
5. Checkpoint filename used

---

## Invariants (never skip)

1. **Always run Phase 0** — data check and tests before any experiment.
2. **Always save to `docs/`** — results not saved are results lost.
3. **Never modify `config.py` during a run** — set config before Phase 1, not between phases.
4. **For runs >2 phases**, activate `checkpoint-resume` skill and update `.claude/pipeline-state.json` after each phase.
5. **Baselines live in `baselines/`** — never move baseline logic into `src/`.

---

## Quick Reference — Key Paths

```
config.py                          ← all hyperparams and paths
data/raw/synthetic_rider_data.csv  ← training data
data/test/independent_test_set.csv ← held-out test set
checkpoints/                       ← saved .pt files
outputs/                           ← CSV outputs
docs/                              ← human-readable result reports
.claude/pipeline-state.json        ← resume state for long runs
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| ROC-AUC not improving | LR too high or gate temperature stuck | Check cosine annealing schedule in `train.py` |
| Gate weights collapse to uniform | Prior smoothing too strong | Reduce `gate_prior_strength` in `config.py` |
| Calibration curve far from diagonal | Temperature not fitted | Run `temperature_scaling.py` after training |
| Baseline outperforms M2G-NET | Feature leakage or data split issue | Re-run `check_data.py`, verify stratification |
