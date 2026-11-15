# Gate round verification (before development fits)

## Scope and declared configuration

See GATE-SEARCH-PLAN.md. New gate modes are optional and default legacy. Config centrally defines search dimensions, hyperparameters, paths, and thresholds. No new library dependency; flake8 is not installed (lint skipped, not reported pass). No p-values. No LOCKED TEST data generation or evaluation.

## Code review / reliability

- `src/fusion.py`: optional Linear–GELU–Linear gate logits; same sparsemax, scheduled positive temperature, and uniform-prior smoothing. Optional per-task blend indexes task before multiplying [batch, view_dim], avoiding task/feature broadcasting.
- `src/model.py` and `scripts/train.py`: explicit optional gate argument; invalid variants or no-gate combinations rejected before random initialization. Default architecture unchanged. Existing ablations still accepted.
- `src/checkpoint.py`: gate mode, skip flag and nonlinear hidden width persisted; bundle reconstruction uses checkpoint mode (missing metadata defaults legacy). Conditional hidden-width mismatch and gate-mode mismatch reject.
- `scripts/evaluate/paper_pipeline.py`: v3 consumes selected config and ensemble exposes correct member metadata. Config snapshot JSON-normalized so tuple-valued search settings do not falsely make resume fail after JSON list serialization. No frozen run-plan or results rewritten.
- `scripts/evaluate/interpret.py`: handles scalar/shared and vector/task blends without `.item()` failure.
- New dev runner uses OS subprocesses for independent torch RNG. Only train/validation encoded and predicted. Held-out return value discarded; no oracle file used. Each cached member has data/source/runtime identity, prediction hash, exact row/target/mask checks, per-unit process lock, and atomic publication. Score input hashes are checked before atomic report publication. Old unsafe cache not reused.
- Scores average training-seed metrics within each generated dataset before paired deltas. Positive delta favors candidate (AUC candidate minus opponent; Brier opponent minus candidate). Aggregate Brier is averaged across three mechanisms per set, not silently replaced by three separate conditions. Missing coverage/errors fail rather than dropping datasets.

## Math assumptions and numerical verification

For finite encoded inputs, B=batch size, J=active gate inputs, d=view width, H=config gate hidden width, K=task count. Nonlinear logits [B,Jd]→[B,H]→[B,J]; sparsemax of q/T then (1−λ) times sparsemax + λ/J yields simplex weights for T>=0.1 and λ∈[0,1]. Gates are representations, not causal importance.

Concrete check retained in tests: logits [2,1,0], T=1, λ=.3 yield [.8,.1,.1] for both gate types; λ=1 yields uniform weights. Convex blend checks use d=K explicitly to expose accidental feature-axis broadcasting.

Independent read-only math-checker findings (no dataset access):
- Five legacy fingerprints identical before/after: parameter state/key ordering, post-init RNG, eval outputs, dropout outputs, gradients. This is preservation of legacy, not a claim new task-blend training is equivalent.
- 29 valid gate/ablation/skip combinations: shapes, simplex, finite gradients and complete in-memory checkpoint reload outputs pass, including nondefault prior .37, temperature 1.27, and nonzero blends.
- Invalid combinations reject before consuming RNG; nonlinear finite-difference input gradcheck passes.
- λ=1 correctly zeroes gate-network gradients. Task-blend coordinates have distinct gradients/Adam moments; they are not equivalent to a shared scalar after training starts.
- Existing verify_math.py exits successfully, but checks obsolete standalone SwiGLU logic and cannot substantiate the new production architecture alone. Dedicated production unit tests and independent checks above supply that coverage.

## Preflight result

Full suite: **68 passed, 50 subtests passed** (`outputs/v3-gate-tests.log`). Includes mandatory tests/test_gate_prior_smoothing.py and new tests/test_v3_gate_variants.py / tests/test_v3_gate_dev.py. All 30 development dataset label contracts and 480 opponent validation artifact identity/hash/alignment contracts checked before fit.

## Manuscript packaging repair (separate user request)

User explicitly confirmed restoring single-author identity from the author-input record. Only `docs/paper/submission/OICTitleFile.docx` was restored. Original two-author file backed up under outputs/title-repair-backup-2026-11-15/.

Restored DOCX hash: a7cb6cadd8cb08f1a95f6e2894fee66527b4740fa3a82510e8d2658284ce8877 — byte-identical to recorded original render input. Existing PDF hash: 9eb97e7b4343e5b59fac6f8fb9eb2c81b6b2abea7075946b5117df3c2eeef478 — still matches provenance. No rerender or expected-hash rewrite was needed or claimed. Six document tests pass unchanged; independent agent visually inspected one-page preview and checked protected manuscript/render sources unchanged. This repairs those two failures, not a blanket certification that submission is ready. No upload/submission or commit.

## Status

Gate search fits running in `outputs/paper-v3-gate-dev/`; four workers. Source/config fixed during fitting. Existing FREEZE.json is stale until development selection and new verified freeze. No success claim before full dev comparison; no locked pass/fail before one authorized locked evaluation.
