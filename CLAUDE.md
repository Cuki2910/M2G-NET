# CLAUDE.md - M2G-NET Project

## Repo Map

```text
M2G-NET/
├── CLAUDE.md                        <- project constitution
├── .agents/
│   ├── README.md                    <- agent architecture index
│   ├── skills/                      <- project-specific and shared skills
│   └── subagents/                   <- isolated verifier/analyst definitions
├── .claude/
│   ├── settings.local.json          <- permissions allowlist + deterministic hooks
│   └── pipeline-state.json          <- checkpoint-resume state
├── config.py                        <- central config: paths, hyperparams, view columns
├── src/
│   ├── model.py                     <- TGMVMTGFNetV2 architecture
│   ├── fusion.py                    <- ResidualGatedFusion + RegularizedTaskGate
│   ├── views.py                     <- per-view encoders
│   ├── interaction.py               <- cross-level interaction layer
│   ├── calibration.py               <- temperature scaling
│   ├── loss.py                      <- multi-task loss
│   ├── metrics.py                   <- evaluation metrics
│   ├── data_pipeline.py             <- synthetic data loader, stratified splits
│   └── checkpoint.py                <- model save/load
├── scripts/
│   ├── train.py                     <- main training loop
│   ├── check_data.py
│   ├── verify_math.py
│   ├── evaluate/                    <- ablation, calibration, significance, site tests
│   ├── explain/                     <- Integrated Gradients and visualizations
│   └── search/                      <- hyperparameter search
├── baselines/                       <- sklearn/XGBoost/LightGBM/MLP baselines
├── data/                            <- synthetic proof-of-concept data
├── checkpoints/                     <- saved model weights
├── outputs/                         <- ignored experiment logs/plots/raw outputs
├── docs/                            <- methodology, formulas, tracked result reports
└── tests/                           <- regression tests
```

## Agent System Layers

This repo uses a 5-layer local agent structure:

1. `CLAUDE.md` - project constitution and invariants.
2. `.agents/skills/` - task lenses and workflows.
3. `.claude/settings.local.json` - deterministic hooks and command permissions.
4. `.agents/subagents/` - isolated verifier/analyst definitions.
5. `.agents/README.md` - architecture index and common workflows.

Hooks are deterministic shell commands, not AI reviewers. Treat hook output as a guardrail; still run the relevant tests and reviews before finalizing.

## Project Invariants

These constraints must not be violated without explicit user confirmation:

1. `config.py` is the single source of truth. Never hardcode paths, column names, or hyperparameters inline. If `config.py` changes, state which ablation/evaluation scripts need to be re-run.
2. `docs/MATHEMATICAL_FORMULAS.md` is the math source of truth. Any change to model math, loss, gate computation, calibration, or metrics must be cross-checked against it.
3. Data is synthetic proof-of-concept unless replaced by real observational records. Do not imply real-world validation is completed.
4. `tests/test_gate_prior_smoothing.py` must pass before any commit touching `src/fusion.py`.
5. Baselines live in `baselines/`. Do not add baseline logic into `src/` or `scripts/evaluate/`.

## Behavioral Guidelines

### 1. Think Before Coding

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them.
- If a simpler approach exists, say so.
- If something is unclear, stop and ask.

### 2. Simplicity First

- No features beyond what was asked.
- No abstractions for single-use code.
- If a change can be smaller and still correct, keep it smaller.

### 3. Surgical Changes

- Do not improve adjacent code, comments, or formatting unless it directly supports the task.
- Match existing style.
- Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

For multi-step tasks, state a brief plan first:

```text
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
```

Use `.claude/pipeline-state.json` for multi-phase tasks via `checkpoint-resume`.

## Mandatory Skill Lenses

| Task type | Skill to apply |
|-----------|----------------|
| Any code edit | `code-review-expert` |
| Training / evaluation experiment | `experiment-runner` |
| Math formula / loss / metric | `math-validator`; use `math-checker` for an independent second opinion when model math changed |
| Experiment output summary | `result-analyst` |
| External research / survey | `deep-research` |
| Multi-phase / long-running task | `checkpoint-resume` |
| Scientific writing in transportation | `critical-transport-editor` |

## Code Review Checklist

Apply before finalizing any edit to `src/`:

- SOLID / architecture impact
- Security and reliability
- Tensor shape, broadcasting, masking correctness
- Boundary conditions and edge cases
- Passing smoke tests is not sufficient for evaluation logic or checkpoint changes

## Math Lens Rules

Apply when editing `src/fusion.py`, `src/model.py`, `src/interaction.py`, `src/loss.py`, `src/calibration.py`, `src/metrics.py`, or `scripts/verify_math.py`:

- Verify tensor shapes and broadcasting match `docs/MATHEMATICAL_FORMULAS.md`.
- State assumptions: domains, dimensions, approximations.
- Include a verification step with concrete numbers for changes to loss, calibration, or gate computation.
- Run `math-validator` first. For model math changes, use the isolated `math-checker` subagent for a second opinion before commit.

## Checkpoint Lens Rules

Apply for any task spanning multiple phases:

- Build an execution plan before starting.
- Save progress to `.claude/pipeline-state.json` after each phase.
- On session start, check whether `pipeline-state.json` exists and resume if applicable.
- Active checkpoint schema uses `completed_phases`, `current_phase`, and `remaining_phases`. Hooks and manual status checks should read those fields, not an older `phases[]` schema.

## Hook Rules

Hooks live in `.claude/settings.local.json`.

- `PostToolUse` after file edits: run a fast `flake8 src/` lint guard.
- `PreToolUse` before training commands: remind the agent to run tests first.
- `Stop`: summarize active checkpoint phase and remaining phases.

Hooks must be portable and must not create stray files such as `nul`.

## Subagent Rules

Subagent definitions live in `.agents/subagents/`.

- `math-checker`: isolated read-only math/code consistency verifier.
- `result-analyst`: isolated read-only experiment output summarizer.

Use subagents only when the user explicitly asks for subagent/delegated work or a skill specifically launches them.

## Transport Editor Lens

Apply when reviewing manuscript text in `docs/`:

- Use the 6-tier assessment: Fatal Flaws -> Methodological Weaknesses -> Overclaims -> Reproducibility -> Writing Quality -> Positioning.
- Stop and demand fixes if Tier 1 fatal flaws are found.
- Maintain synthetic-data limitations unless real observational validation is actually added.

---

These guidelines are working if there are fewer unnecessary diffs, fewer rewrites due to overcomplication, and clarifying questions come before mistakes.
