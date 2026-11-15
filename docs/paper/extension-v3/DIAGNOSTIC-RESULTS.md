# Fixed M5 diagnosis — 300 fits completed

Synthetic DEVELOPMENT only. Full tables/provenance in `diagnostics/1794732210625341000/`. No test/locked/oracle scores used. All 300 validation prediction arrays match the previous M5 control bitwise; see DIAGNOSTIC-VERIFICATION.md. Independent result-analyst checked coverage, CSV hashes and metric-level training-seed averaging.

## Macro eval-mode train versus validation

Five-member probability ensemble; metrics averaged across training seeds 42/101 within each generator, then equally across ten generators.

| Mechanism | Train AUC | Validation AUC | AUC gap train−validation |
|---|---:|---:|---:|
| additive | .6661 | .6262 | .0399 |
| interaction | .7879 | .7547 | .0332 |
| partial | .7935 | .7406 | .0529 |

## Per-task eval-mode gaps

Each cell: AUC train−validation / Brier validation−train. Positive gaps mean training predictions score better, not a proof of causal error decomposition.

| Task | Additive | Interaction | Partial |
|---|---:|---:|---:|
| Red-light running | .0194 / .0060 | .0599 / .0232 | .0829 / .0295 |
| No turn signal | .0557 / .0083 | .0429 / .0131 | .0581 / .0196 |
| Helmet nonuse | .0466 / .0079 | .0297 / .0107 | .0330 / .0117 |
| Mobile-phone use | .0377 / .0056 | .0002 / .0018 | .0378 / .0103 |

Task gaps differ materially. In particular, mobile-phone interaction has nearly zero mean AUC gap while red-light partial has .0829. Global added capacity is not supported by these measurements alone.

## Epochs and learning trajectory

| Mechanism | Mean selected epoch | Median selected epoch | Member range | Mean selected→final validation AUC decline |
|---|---:|---:|---:|---:|
| additive | 15.65 | 14 | 3–48 | .0154 |
| interaction | 22.40 | 20 | 10–59 | .0112 |
| partial | 23.64 | 21.5 | 9–50 | .0113 |

Every fit ended exactly 20 epochs after its selected checkpoint; none hit the 150-epoch limit. Macro validation Brier worsened .0040/.0042/.0048 by termination. These declines are conditioned on selecting the best validation epoch; they are not independent proof of overfitting. Online training/dropout metrics are retained separately and were not compared to eval-mode validation as if equivalent.

## Next bounded hypothesis

Recommendation: test stronger weight decay, not a larger network or longer training first. Compare **1e−3** against existing **1e−4**, holding all other hyperparameters, architecture, BCE, five members, seeds, early stopping and rows fixed. Values must enter config.py before that new experiment, with new cache identity and preflight. This is a hypothesis; no gain is measured yet.

Use all three mechanisms/ten dev generators/two training seeds, per-task metrics plus exact eight-opponent screen. Inspect whether red-light partial improves without sacrificing other tasks. No generator changes, no extra data pooling, no changed success thresholds. Development remains adaptive; only the one authorized locked evaluation can confirm final success.

## Current boundary

Diagnosis complete, not success at the original target. Stronger-decay experiment has NOT started. Existing FREEZE.json remains stale; locked data remain untouched. No commit, push or submission. Full verification before diagnosis: 75 tests and 50 subtests pass; lint unavailable.
