# M2G-Net v2

Incorporating behavioral theory into deep learning: a Multi-View Multi-Task Gated Fusion Network (M2G-Net) for motorcycle risky-behavior prediction.

The architecture uses theory-informed grouping of observed rider and contextual variables. Demographic features do not directly measure Theory of Planned Behavior constructs; attribution is not a test of those theories. The repository is a **synthetic proof-of-concept**, not deployed policy support or real-world safety validation.

This repository contains a research implementation for training, evaluating, comparing baselines, and generating explanation analyses for multi-task behavioral prediction.

## Quick Start

```bash
pip install -r requirements.txt
python scripts/check_data.py                    # Fails on contradictory source labels
python scripts/check_data.py --label-policy mask # Explicitly exclude inapplicable labels
python -m unittest discover -s tests
python scripts/evaluate/paper_pipeline.py --label-policy mask --smoke --seeds 42 --gate-priors 0.1 --bootstrap 30
python scripts/evaluate/paper_pipeline.py --label-policy mask
python scripts/evaluate/paper_pipeline.py --label-policy mask --split site
python scripts/train.py --label-policy mask     # Standalone checkpoint
python scripts/evaluate/test_on_independent_set.py # Development out-of-site data, not final test
python scripts/evaluate/interpret.py              # Gate weights analysis
python scripts/explain/ig_explain.py --checkpoint checkpoints/best_model.pt --output outputs/ig_analysis
python scripts/explain/visualize.py
```

The source CSVs are never rewritten. Default `strict` policy rejects inconsistent
red-light labels; `mask` explicitly excludes labels at nonsignalized sites.
Use `--require-masks` for observational data with verified task eligibility.
The historical independent-site CSV was previously used for tuning and is not
an untouched final test. See `docs/PIPELINE_PROTOCOL.md`.

Benchmarks, ablations, and repeated-run entry points share the paper pipeline.
It selects hyperparameters on validation only, retrains module ablations,
fits temperature before selecting thresholds, and writes versioned artifacts
under `outputs/paper_pipeline/`. Baselines use train-only preprocessing and do
not train on validation rows. Neural seeds share fixed split membership.
Short `--smoke` runs are explicitly marked unsuitable for publication.
The IG CLI integrates jointly from zero encoder outputs, preserves signed
attributions and rejects numerical completeness failures. It uses the saved
test rows, observed task masks and checkpoint site-intercept mode. Output must
be a new directory. These are view-level encoder-space explanations, not raw
categorical-feature attributions or causal effects. See `docs/IG_VALIDATION_RESULTS.md`.
Publication readiness remains open: data provenance/eligibility, full runs,
untouched validation and study-level explanation claims are not established.
See `docs/PUBLISH_READINESS.md`; pipeline repair completion is not submission readiness.

On macOS, XGBoost/LightGBM need an OpenMP runtime (`brew install libomp`).
When unavailable, choose `--models m2g_net catboost logistic_regression`.

## Architecture

M2G-Net uses:
1. **Five view-specific encoders:** Rider role, rider traits, road context, environment, and site-aware context.
2. **Cross-level interaction:** Blends individual and contextual representations.
3. **Gated fusion:** Task-specific gated fusion with temperature annealing.
4. **Masked multi-task loss:** Handles partially observed risky-behavior labels.

## Repository Structure

```text
src/
  data_pipeline.py     data loading, encoding, task masks
  views.py             view-specific and site-aware encoders
  interaction.py       cross-level interaction module
  fusion.py            task-specific gated fusion
  loss.py              masked uncertainty-weighted loss
  metrics.py           masked metrics
  model.py             M2G-Net v2 model

scripts/
  train.py             Main training pipeline
  evaluate/            Test sets, ablations, fairness, significance tests
  explain/             Integrated Gradients & visualizations
  search/              Hyperparameter search

baselines/             Tabular and MLP baselines
docs/                  Mathematical formulations and architecture visual
```

## Key Documentation

- **Architecture:** `docs/ARCHITECTURE_VISUALIZATION.md`
- **Evaluation protocol:** `docs/PIPELINE_PROTOCOL.md`
- **Publication readiness and blockers:** `docs/PUBLISH_READINESS.md`
- **Math Formulas:** `docs/MATHEMATICAL_FORMULAS.md`
- **Methods & Terms Glossary:** `docs/METHODS_TERMS_GLOSSARY.md`
- **Competing Methods Research:** `docs/METHOD_COMPETITORS_RESEARCH.md`
- **Strong Baseline Repeated Runs:** `docs/STRONG_BASELINE_REPEATED_RUNS_RESULTS.md`
- **Current Metrics:** `docs/results/current_metrics.md`
- **Results:** `docs/INDEPENDENT_TEST_RESULTS.md`
