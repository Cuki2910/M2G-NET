---
name: result-analyst
description: "Spawn an isolated result-analyst subagent to read docs/ and outputs/ and produce a structured experiment summary with metric tables, key findings, and next-step recommendations."
tags: [m2g-net, results, analysis, subagent]
version: 1.0.0
user-invocable: true
allowed-tools: [Agent]
---

# Result Analyst — Subagent Launcher

Spawns an isolated read-only subagent that analyzes M2G-NET experiment outputs.
No files are modified. The subagent only reads and summarizes.

## When to Invoke

- After completing a training + evaluation run
- When you want a summary of what the latest experiments show
- Before writing the results section of the manuscript
- When comparing multiple experiment variants

## How to Invoke

```
Agent(
  description="Analyze M2G-NET experiment results",
  subagent_type="general-purpose",
  prompt=[contents of .agents/subagents/result-analyst/agent.yaml default_prompt]
)
```

## What it Analyzes

- All `.md` files in `docs/` matching `*RESULTS*.md`, `*REPORT*.md`, `*SUMMARY*.md`
- All `.csv` files in `outputs/`
- Identifies best model variant, regressions, and anomalies

## Output

Returns a structured report under 400 words:
- Metric table (ROC-AUC, ECE, F1 per model variant)
- Key findings with file references
- Regressions vs. prior runs
- Recommended next experiments (specific, actionable)
