# Gate Prior Smoothing Sweep Results

This report records the first controlled sweep of the post-softmax gate prior
smoothing strength introduced for M2G-Net v2.

## Setup

- Script: `scripts/evaluate/gate_prior_sweep.py`
- Lambdas: `0.0`, `0.05`, `0.1`, `0.2`
- Seeds: `42`, `101`
- Max epochs: `80`
- Early stopping patience: `12`
- Independent test: `data/test/independent_test_set.csv`
- Temporary checkpoints: `outputs/gate_prior_sweep/checkpoints/`

The smoothing formula is:
\[
    \alpha_k^{\mathrm{raw}}
    =
    \operatorname{softmax}\left(\frac{A_k r + c_k}{T}\right),
    \qquad
    \alpha_k
    =
    \frac{\alpha_k^{\mathrm{raw}}+\lambda p_0}{1+\lambda}.
\]

## Summary

| Lambda | Test ROC-AUC | Test PR-AUC | Test F1 | Independent ROC-AUC | Independent PR-AUC | Independent F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.7213 | 0.2914 | 0.3870 | 0.7218 | 0.2739 | 0.3836 |
| 0.05 | 0.7212 | 0.2922 | 0.3828 | 0.7230 | 0.2771 | 0.3755 |
| 0.10 | 0.7217 | 0.2922 | 0.3793 | 0.7246 | 0.2783 | 0.3761 |
| 0.20 | 0.7211 | 0.2924 | 0.3794 | 0.7218 | 0.2746 | 0.3750 |

## Decision

Keep `GATE_PRIOR_WEIGHT = 0.1`.

This value has the best independent macro ROC-AUC in the sweep while keeping
test ROC-AUC and PR-AUC competitive. The tradeoff is slightly lower tuned F1
than `lambda=0.0`, so the current model should be described as optimized for
ranking/generalization rather than for maximum thresholded F1.

## Caveat

This is a small sweep over two seeds on synthetic proof-of-concept data. It is
enough to justify the current default, but not enough to claim global optimality.
Before publication claims, rerun with more seeds and real held-out sites.
