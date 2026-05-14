---
name: math-validator
description: "Validate that M2G-NET code (src/fusion.py, src/model.py, src/interaction.py) matches the documented math in docs/MATHEMATICAL_FORMULAS.md. Use whenever editing model architecture, loss, gate computation, calibration, or metrics."
tags: [m2g-net, math, validation, fusion, model]
version: 1.0.0
user-invocable: true
allowed-tools: [Bash, Read, Grep]
---

# Math Validator — M2G-NET

Structural cross-check between `docs/MATHEMATICAL_FORMULAS.md` (the specification) and `src/` (the implementation). Produces a pass/fail report with specific mismatches.

## When to Use

- After editing `src/fusion.py`, `src/model.py`, `src/interaction.py`, or `src/loss.py`
- When adding or modifying a loss term, gate weight, or calibration step
- When translating a new formula from the paper into code
- Before submitting any commit that touches model math

---

## Validation Workflow

### Step 1 — Load the specification

```
Read("docs/MATHEMATICAL_FORMULAS.md")
```

Extract and list every named formula with its:
- Symbol and domain (e.g., `g_k ∈ ℝ^V`, sparsemax output)
- Tensor shape contract (e.g., `[batch, V]` → `[batch, K, V]`)
- Key operations (softmax vs. sparsemax, temperature, masking)

### Step 2 — Map formulas to code

| Formula section | File to check |
|----------------|--------------|
| §1 View Encoding `h_i = f_i(x_i)` | `src/views.py` |
| §2 Site-Aware Representation | `src/views.py` (site embedding block) |
| §3 Cross-Level Interaction | `src/interaction.py` |
| §4 Gated Fusion (sparsemax, prior smoothing) | `src/fusion.py` — `ResidualGatedFusion`, `RegularizedTaskGate` |
| §5 Multi-Task Loss + mask `m_k` | `src/loss.py` |
| §6 Temperature Scaling / Calibration | `src/calibration.py` |
| Metrics (ROC-AUC, ECE) | `src/metrics.py` |

### Step 3 — Run the automated check

```bash
python scripts/verify_math.py
```

This script checks tensor shapes and key invariants. Review its output first before doing manual inspection.

### Step 4 — Manual spot-checks (for each changed formula)

For each formula that was modified, verify:

1. **Shape contract**: Do tensor dimensions match the formula subscripts?
   - e.g., gate weights `g` should be `[batch, K, V]` if formula says `g_k ∈ ℝ^V` per task
2. **Operation identity**: Is the implementation using the right operation?
   - sparsemax ≠ softmax — check import and call site in `src/fusion.py`
   - Temperature scaling: is `τ` applied before or after the gate?
3. **Mask correctness**: Is `m_k^(n)` applied correctly in the loss?
   - Loss should be zeroed for unobserved labels, not included in mean
4. **Prior smoothing**: Gate prior `π` should blend uniformly, not collapse weights
   - Check `gate_prior_strength` parameter in `RegularizedTaskGate`
5. **Out-of-site mode**: When site ID is unknown, `h_rand = 0` (zeroed embedding)
   - Verify in `src/views.py` site-aware block

### Step 5 — Concrete numerical check (for critical changes)

When changing loss, calibration, or gate computation, run a sanity check with known values:

```bash
python -c "
import torch
from src.fusion import RegularizedTaskGate

# Smoke test: uniform input should produce near-uniform gate weights
gate = RegularizedTaskGate(num_views=5, num_tasks=4)
x = torch.zeros(2, 128)  # batch=2, hidden=128
g = gate(x)
print('Gate shape:', g.shape)       # expect [2, 4, 5]
print('Gate sum per task:', g.sum(dim=-1))  # expect ~1.0 per task (sparsemax)
"
```

---

## Output Format

After completing Steps 1–5, report as:

```markdown
## Math Validation Report

**Formula sections checked**: [list]
**Files inspected**: [list]
**verify_math.py result**: PASS / FAIL (paste relevant output)

### Findings

| Formula | Code location | Status | Note |
|---------|--------------|--------|------|
| Sparsemax gate g_k | src/fusion.py:L42 | ✅ PASS | Shape [B,K,V] correct |
| Loss mask m_k | src/loss.py:L18 | ⚠️ WARN | Mean includes masked zeros |
| Temperature τ | src/calibration.py:L31 | ✅ PASS | Applied post-logit |

### Action Required

[List any FAIL or WARN items with specific fix]
```

---

## Key Invariants to Never Violate

| Invariant | Where enforced |
|-----------|---------------|
| sparsemax (not softmax) for gate weights | `src/fusion.py` |
| Gate weights sum to ≤ 1 per task (sparse) | sparsemax property |
| Loss masked by `m_k` — never average over unobserved | `src/loss.py` |
| `h_rand = 0` for unknown sites | `src/views.py` |
| Temperature `τ > 0` enforced (softplus or clamp) | `src/fusion.py` |
| `docs/MATHEMATICAL_FORMULAS.md` is source of truth | All of `src/` |
