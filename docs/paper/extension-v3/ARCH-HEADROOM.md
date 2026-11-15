# Headroom check before changing the architecture (dev validation split, seeds 1701-1710, training seed 42)

> Correction (gate development round): the LR fits below are diagnostics, NOT mathematical upper bounds on a neural model. The old interpretations in lines below are retained as historical statements, not validated conclusions. Equality to a structure-aware LR does not prove zero headroom. The stated ~2400 training rows was also inaccurate: with configured 20% test and 10% validation, training uses 70% of 3000 rows (2100). Neither the oracle gap nor the asserted 0.001–0.003 noise level isolates estimation noise from approximation/optimization error. Do not use these diagnostics to declare gate search futile. New selection uses observed DEVELOPMENT validation only, no oracle probabilities.

Question: can a new interaction branch / gate change plausibly add > +0.005 macro AUC over m2g_net at interaction and partial?
Method: fit reference models that KNOW the generator's structure (diagnostic ceilings only, not comparators, not for the paper).
Script: `scripts/diag_v3_headroom.py` (logistic regression, true feature contrasts + the 3 true product terms + site one-hot).
No test split, no LOCKED seeds. Validation-split means; synthetic data.

| Model (macro AUC) | additive | interaction | partial |
|---|---:|---:|---:|
| m2g_net (old) | 0.6213 | 0.7516 | 0.7356 |
| **v3 = M5 (frozen)** | 0.6268 | 0.7549 | 0.7401 |
| LR with the 3 true product terms | 0.6233 | 0.7596 | 0.7400 |
| LR, all 15 individual x context pairs, C=0.3 | 0.6080 | 0.7539 | 0.7339 |
| LR, all pairs, L1, C=0.3 | 0.6136 | 0.7571 | 0.7362 |
| True probabilities (oracle, not learnable from 2400 rows) | 0.6322 | 0.7678 | 0.7516 |

Reading:
- partial: v3 (0.7401) already equals the structure-aware LR (0.7400). No headroom from architecture at this sample size.
- interaction: structure-aware LR is only +0.0047 above v3 and +0.0080 above old m2g_net. An architecture change that matched this
  best-case knowledge would still land near the +0.005 threshold vs m2g_net, with no margin for the 18/20 requirement.
- Generic interaction learners (all-pairs LR) do no better than v3, so a more flexible interaction branch is unlikely to help.
- The gap to the oracle (~0.013-0.015) is estimation noise from n~2400, not missing model capacity.
