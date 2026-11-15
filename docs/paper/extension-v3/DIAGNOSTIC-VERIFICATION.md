# Fixed M5 diagnostic verification

- Preflight: 75 tests passed, 50 subtests (`outputs/v3-diagnostic-preflight.log`); all 30 DEVELOPMENT train/validation label contracts validated.
- Dedicated telemetry/runner synthetic tests: 7 passed; independent code review found no blocking defects. CPU only; no claim about CUDA/MPS RNG preservation.
- Optional `diagnostics` in `fit_model_for_split` records already-computed epoch metrics and only adds sequential eval-mode forwards after the best model/loss states are restored. Torch fork_rng plus NumPy state restoration protects the original RNG stream. Original dropout/sampling/checkpoint paths remain unchanged in regression.
- 300 fixed M5 member fits and scoring completed (background command exit 0). No model math/loss/gate changes or hyperparameter sweep in this round.
- All **300/300 member validation predictions** compared to the prior gate-search legacy control: identical row IDs, labels, masks, and raw probabilities; **maximum difference = 0.0**. Record: `outputs/v3-diagnostic-reproducibility.json`. Old control cache was read only for this validation reproduction audit, not reused as a substitute for new diagnostics.
- New diagnostic cache separately hashes source/config/runtime/data and member predictions+telemetry. Per-task metrics recomputed from arrays agree with selected-state telemetry; reports atomic and immutable timestamp directories.
- Data scope is synthetic proof-of-concept. DEVELOPMENT train/validation only; no oracle or test predictions used. LOCKED TEST not generated/evaluated.
- Result tables: `diagnostics/1794732210625341000/` (unit/generator/group metrics, epoch curves and member/group epochs, provenance).
- Lint unavailable (flake8 not installed); not counted as passing. No commit, push, upload, submission, or new freeze.
- Existing FREEZE.json remains stale during continued development. Any adopted model needs a new source/config freeze and explicit user confirmation before the one final locked evaluation.
