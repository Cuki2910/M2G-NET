---
name: math-checker
description: "Spawn an isolated math-checker subagent to independently verify that src/ code matches docs/MATHEMATICAL_FORMULAS.md. Use after any edit to fusion.py, model.py, interaction.py, or loss.py."
tags: [m2g-net, math, verification, subagent]
version: 1.0.0
user-invocable: true
allowed-tools: [Agent]
---

# Math Checker — Subagent Launcher

Spawns an isolated subagent that independently verifies M2G-NET math implementation.
The subagent has no access to the current conversation context — it gives a clean second opinion.

## When to Invoke

- After any edit to `src/fusion.py`, `src/model.py`, `src/interaction.py`, `src/loss.py`
- Before committing a change that touches loss, gate, or calibration logic
- When the `math-validator` skill flags a WARN and you want independent confirmation

## How to Invoke

Use the Agent tool with the subagent definition:

```
Agent(
  description="Independent math verification for M2G-NET",
  subagent_type="general-purpose",
  prompt=[contents of .agents/subagents/math-checker/agent.yaml default_prompt]
)
```

## What it Checks

1. `docs/MATHEMATICAL_FORMULAS.md` vs. `src/fusion.py` — sparsemax, gate shapes, temperature
2. `src/loss.py` — mask `m_k` applied correctly (zero, not average)
3. `src/model.py` — view concatenation order matches formula notation
4. `src/interaction.py` — cross-level interaction dimensions
5. `python scripts/verify_math.py` — automated shape checks

## Output

Returns a table: Formula | File:Line | ✅PASS / ⚠️WARN / ❌FAIL | Detail

A clean run should show all PASS. Any FAIL must be fixed before the next commit to `src/`.
