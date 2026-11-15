# Gate search — completed DEVELOPMENT round

All 1,200 member fits and scoring completed with exit code 0. Synthetic DEVELOPMENT validation only, generator seeds 1701–1710, training seeds 42/101 averaged at metric level per generated set, five probability-averaged initialization members. No locked-test result is available.

**Decision: none of the four variants passes the complete development screen. Do not claim superiority.** Gate modifications were not adopted; default selected gate remains legacy. Existing FREEZE.json is stale after source changes; no new freeze or locked run performed.

## Macro AUC gains against old m2g_net

| Variant | Mechanism | Mean ± MCSE | Min | Max | Positive / 10 |
|---|---|---:|---:|---:|---:|
| legacy | additive | +0.005149 ± 0.001546 | -0.000195 | +0.017804 | 9/10 |
| legacy | interaction | +0.003909 ± 0.001617 | -0.003385 | +0.011976 | 8/10 |
| legacy | partial | +0.003219 ± 0.000601 | -0.000534 | +0.005272 | 9/10 |
| nonlinear | additive | +0.005965 ± 0.001992 | +0.000572 | +0.021828 | 10/10 |
| nonlinear | interaction | +0.002317 ± 0.001754 | -0.005209 | +0.011973 | 7/10 |
| nonlinear | partial | +0.002183 ± 0.001159 | -0.003745 | +0.007319 | 7/10 |
| task_blend | additive | +0.005112 ± 0.001456 | -0.001953 | +0.015917 | 9/10 |
| task_blend | interaction | +0.003776 ± 0.001625 | -0.003478 | +0.011834 | 8/10 |
| task_blend | partial | +0.002642 ± 0.000621 | -0.000772 | +0.005006 | 9/10 |
| nonlinear_task_blend | additive | +0.005565 ± 0.001810 | +0.000796 | +0.020501 | 10/10 |
| nonlinear_task_blend | interaction | +0.002685 ± 0.001683 | -0.003479 | +0.012442 | 6/10 |
| nonlinear_task_blend | partial | +0.002026 ± 0.000782 | -0.002320 | +0.005652 | 7/10 |

Dev screening mirrors the predeclared mean > +0.005 and ≥90% positives, but with 10 dev generators (≥9/10), not the final 18/20 locked criterion. All variants fail AUC conditions; all satisfy additive mean noninferiority and aggregate Brier conditions on dev. Adaptive dev selection is not final evidence. No p-values.

## Aggregate Brier improvements (opponent minus candidate)

| Variant | Opponent | Mean ± MCSE | Min | Max | Positive / 10 |
|---|---|---:|---:|---:|---:|
| legacy | early_fusion_mlp | +0.003446 ± 0.000402 | +0.000857 | +0.005062 | 10/10 |
| legacy | catboost | +0.005238 ± 0.000354 | +0.003377 | +0.006737 | 10/10 |
| nonlinear | early_fusion_mlp | +0.003281 ± 0.000480 | +0.000885 | +0.005577 | 10/10 |
| nonlinear | catboost | +0.005073 ± 0.000391 | +0.002984 | +0.006380 | 10/10 |
| task_blend | early_fusion_mlp | +0.003395 ± 0.000390 | +0.000819 | +0.004945 | 10/10 |
| task_blend | catboost | +0.005187 ± 0.000364 | +0.003296 | +0.006879 | 10/10 |
| nonlinear_task_blend | early_fusion_mlp | +0.003239 ± 0.000458 | +0.001139 | +0.005425 | 10/10 |
| nonlinear_task_blend | catboost | +0.005030 ± 0.000410 | +0.003044 | +0.006450 | 10/10 |

## Failed development checks

### legacy
- interaction vs m2g_net: auc_superiority
- interaction vs shared_gate: auc_superiority
- partial vs m2g_net: auc_superiority
- partial vs shared_gate: auc_superiority
- partial vs no_interaction: auc_superiority
- partial vs early_fusion: auc_superiority
### nonlinear
- interaction vs m2g_net: auc_superiority
- interaction vs shared_gate: auc_superiority
- interaction vs no_interaction: auc_superiority
- interaction vs early_fusion: auc_superiority
- partial vs m2g_net: auc_superiority
- partial vs shared_gate: auc_superiority
- partial vs no_interaction: auc_superiority
- partial vs early_fusion: auc_superiority
### task_blend
- interaction vs m2g_net: auc_superiority
- interaction vs shared_gate: auc_superiority
- partial vs m2g_net: auc_superiority
- partial vs shared_gate: auc_superiority
- partial vs no_interaction: auc_superiority
- partial vs early_fusion: auc_superiority
### nonlinear_task_blend
- interaction vs m2g_net: auc_superiority
- interaction vs shared_gate: auc_superiority
- interaction vs no_interaction: auc_superiority
- interaction vs early_fusion: auc_superiority
- partial vs m2g_net: auc_superiority
- partial vs shared_gate: auc_superiority
- partial vs no_interaction: auc_superiority
- partial vs early_fusion: auc_superiority

## Complete audit tables

- `docs/paper/extension-v3/gate-search/score-1794730441543881000/validation-units.csv`
- `docs/paper/extension-v3/gate-search/score-1794730441543881000/paired-deltas.csv`
- `docs/paper/extension-v3/gate-search/score-1794730441543881000/summary.csv`
- `docs/paper/extension-v3/gate-search/score-1794730441543881000/screening.json`

## Verification and limitations

- Preflight: 68 tests and 50 subtests passed; 480 opponent validation contracts checked. Independent math and runner review completed; see GATE-VERIFICATION.md.
- flake8 absent; lint not claimed as passed. No source/config changes during fitting.
- Prior document failures repaired by user-confirmed single-author byte-exact DOCX restoration, not weakening tests.
- No new training round has started at this report. User authorized continued development; any next direction must be declared before new fits and retain all negative results.
