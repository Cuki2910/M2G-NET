# M2G-NET Agent Architecture

5-layer Claude Code agent system for the M2G-NET research project.

## Layer Map

```
Layer 1 — CLAUDE.md          Project constitution (root CLAUDE.md)
Layer 2 — Skills             .agents/skills/
Layer 3 — Hooks              .claude/settings.local.json → "hooks"
Layer 4 — Subagents          .agents/subagents/
Layer 5 — This file          .agents/README.md (architecture index)
```

---

## Layer 1 — CLAUDE.md

**File**: `CLAUDE.md` (project root)

Always loaded. Contains:
- Full repo map
- 5 project invariants (config.py, math docs, synthetic data, tests, baselines)
- Behavioral guidelines (surgical changes, simplicity, goal-driven)
- Mandatory skill lenses table

---

## Layer 2 — Skills

Invoked on-demand via `/skill-name` or matched by description.

| Skill | Trigger | Purpose |
|-------|---------|---------|
| `experiment-runner` | Train / evaluate / ablation requests | 4-phase standardized experiment pipeline |
| `math-validator` | Edit to `src/fusion.py`, `src/model.py` | Cross-check code vs. `MATHEMATICAL_FORMULAS.md` |
| `math-checker` | Independent math verification needed | Spawns isolated math-checker subagent |
| `result-analyst` | Summarize experiment outputs | Spawns isolated result-analyst subagent |
| `code-review-expert` | Any code diff | Senior engineer lens: SOLID, security, boundary |
| `math` | Formula computation / verification | SymPy, Z3, unit conversion |
| `deep-research` | External knowledge / survey | Citation-tracked multi-source research |
| `checkpoint-resume` | Multi-phase tasks | Pipeline state → `.claude/pipeline-state.json` |

**Project-specific skills** (in this directory):
- `.agents/skills/experiment-runner/`
- `.agents/skills/math-validator/`
- `.agents/skills/math-checker/`
- `.agents/skills/result-analyst/`

---

## Layer 3 — Hooks

**File**: `.claude/settings.local.json`

| Event | Trigger | Action |
|-------|---------|--------|
| `PostToolUse` | Edit or Write any file | `flake8 src/` lint check |
| `PreToolUse` | Bash command containing `train.py` | Warn to run tests first |
| `Stop` | Session ends | Report `current_phase` and `remaining_phases` from `pipeline-state.json` |

Hooks are **deterministic shell commands** — not AI. They run unconditionally.

---

## Layer 4 — Subagents

Isolated Claude instances with their own context, tools, and permissions.

| Subagent | Definition | Spawned by |
|----------|-----------|-----------|
| `math-checker` | `.agents/subagents/math-checker/agent.yaml` | `math-checker` skill |
| `result-analyst` | `.agents/subagents/result-analyst/agent.yaml` | `result-analyst` skill |

**Key properties:**
- Run in isolated context — no access to current conversation
- Read-only tools only (`Read`, `Bash` for math-checker; `Read`, `Glob` for result-analyst)
- Return a single structured report to the main agent

---

## Common Workflows

### Start a training run
```
1. /experiment-runner        ← activates experiment-runner skill
2. Phase 0: check_data + pytest
3. Phase 1: train.py
4. Phase 4: save results to docs/
```

### Review a code change to fusion.py
```
1. /math-validator           ← quick self-check
2. /math-checker             ← spawn independent subagent for second opinion
3. /code-review-expert       ← SOLID + security review
```

### Summarize experiment results
```
1. /result-analyst           ← spawn subagent, get metric table + next steps
```

### Long multi-phase task
```
1. /checkpoint-resume        ← build plan, save to pipeline-state.json
2. After each phase: update `completed_phases`, `current_phase`, and `remaining_phases`
3. Resume: /checkpoint-resume reads state and continues
```

---

## Invariants (enforced by CLAUDE.md + Hooks)

1. `config.py` — single source of truth for all hyperparams
2. `docs/MATHEMATICAL_FORMULAS.md` — single source of truth for all math
3. `tests/test_gate_prior_smoothing.py` — must pass before any commit to `src/fusion.py`
4. `data/` — synthetic only; real-world data is future work
5. Baselines — live in `baselines/`, never in `src/`
