# M2G-Net v2 Architecture

This document describes the implemented architecture and uses the same
hyperparameters as `config.py`. It is a technical reference, not a submission
claim about real-world external validity.

## Paper Figure

The updated paper-facing SVG figure is saved at:

```text
outputs/m2g_net_architecture_v2.svg
```

It includes the cross-level interaction block, six gated inputs, task-specific
observation masks, site-aware context terminology, and view-reliance wording.

## Data Status

The default repository configuration uses:

```text
data/raw/synthetic_rider_data.csv
data/test/independent_test_set.csv
```

These files should be treated as synthetic proof-of-concept data unless they
are replaced by real observational data. Results on sites 32-50 therefore
indicate synthetic out-of-site performance under the current data-generating
process, not external real-world validation.

## Architecture Flow

```text
Input rider/intersection record
        |
        v
Five view encoders
  1. rider_role      -> h_role
  2. rider_traits    -> h_traits
  3. road_context    -> h_road
  4. environment     -> h_env
  5. site-aware view -> h_site
        |
        v
Individual/context grouping
  h_ind = [h_role; h_traits]
  h_ctx = [h_road; h_env; h_site]
        |
        v
Cross-level interaction
  h_int = psi(p_ind + p_ctx + W_int(p_ind * p_ctx))
        |
        v
Task-specific gated fusion
  alpha_raw_k = softmax((A_k r + c_k) / T)
  alpha_k = (alpha_raw_k + lambda * uniform_prior) / (1 + lambda)
  z_gated_k = sum_j alpha_kj * h_tilde_j
        |
        v
Residual early-fusion blend
  z_k = sigmoid(a) z_gated_k + (1 - sigmoid(a)) z_early
        |
        v
Four task heads
  red_light_running
  no_turn_signal
  helmet_nonuse
  mobile_phone_use
```

## Site-Aware View

The site view combines observed site covariates with a regularized
site-specific embedding:

```text
h_site = phi([h_obs; h_site_id])
```

The site-id component is a learnable embedding trained with weight decay. It is
best described as a regularized site-specific intercept-like representation,
not as a full probabilistic mixed-effects model. For unseen sites, evaluation
sets this component to zero and uses observed site covariates only.

## Partial-Observation Labels

The implementation supports a task mask:

```text
m_ik = 1 if task k is observable/applicable for sample i
m_ik = 0 otherwise
```

The masked multi-task objective is used when mask columns are present in the
CSV. If mask columns are absent, all labels default to observed for backward
compatibility with the current synthetic files.

Expected optional mask columns:

| Task | Mask column |
|---|---|
| `red_light_running` | `red_light_running_observed` |
| `no_turn_signal` | `no_turn_signal_observed` |
| `helmet_nonuse` | `helmet_nonuse_observed` |
| `mobile_phone_use` | `mobile_phone_use_observed` |

## Implemented Hyperparameters

These values are read from `config.py`:

| Parameter | Value |
|---|---:|
| `EMBEDDING_DIM` | 8 |
| `HIDDEN_DIM` | 32 |
| `VIEW_DIM` | 16 |
| `INTERACTION_DIM` | 16 |
| `DROPOUT_RATE` | 0.30 |
| `NUM_GATE_INPUTS` | 6 |
| `TEMPERATURE_INIT` | 2.0 |
| `TEMPERATURE_FINAL` | 1.0 |
| `GATE_PRIOR_WEIGHT` | 0.1 |
| `LEARNING_RATE` | 1e-3 |
| `WEIGHT_DECAY` | 1e-4 |
| `BATCH_SIZE` | 64 |
| `MAX_EPOCHS` | 150 |
| `EARLY_STOPPING_PATIENCE` | 20 |
| `USE_FOCAL_LOSS` | True |

## Interpretation Language

Gate weights are reported as the model's relative reliance on learned view
representations for each task. They should not be described as causal feature
importance unless validated against ablation, Integrated Gradients, and
stability analyses.

## Code Mapping

| Component | File |
|---|---|
| Data pipeline and masks | `src/data_pipeline.py` |
| View encoders and site-aware encoder | `src/views.py` |
| Cross-level interaction | `src/interaction.py` |
| Gated and residual fusion | `src/fusion.py` |
| Output heads | `src/model.py` |
| Masked uncertainty-weighted loss | `src/loss.py` |
| Metrics with task masks | `src/metrics.py` |
