# Review Rubric for Transportation & Green Transition Research

This rubric provides scoring criteria for evaluating research quality across multiple dimensions.

## Overall Score Interpretation

| Score | Interpretation | Action |
|-------|----------------|--------|
| 9-10 | Exceptional, publication-ready | Accept with minor edits |
| 7-8 | Strong, needs minor revision | Minor revision |
| 5-6 | Promising, needs major revision | Major revision |
| 3-4 | Significant flaws, may be salvageable | Reject & resubmit |
| 1-2 | Fatal flaws, not salvageable | Reject |

---

## Dimension 1: Scientific Rigor (Weight: 30%)

### 10/10 - Exemplary
- Causal claims backed by appropriate design (RCT, natural experiment, valid IV)
- All confounders identified and addressed
- Uncertainty quantified with appropriate methods
- Sensitivity analyses for key assumptions
- Limitations discussed honestly

### 7-9/10 - Strong
- Mostly rigorous with minor gaps
- Causal language appropriate for design
- Most confounders addressed
- Uncertainty reported
- Limitations acknowledged

### 4-6/10 - Adequate
- Some methodological weaknesses
- Occasional causal overclaims
- Some confounders unaddressed
- Limited uncertainty quantification
- Limitations mentioned but not thoroughly discussed

### 1-3/10 - Weak
- Major methodological flaws
- Causal claims from correlational data
- Confounders ignored
- No uncertainty quantification
- Limitations not discussed

**Red flags for low scores**:
- "X causes Y" without causal design
- No baseline comparisons
- Cherry-picked metrics
- Hyperparameter tuning on test set

---

## Dimension 2: Data Quality & Reproducibility (Weight: 20%)

### 10/10 - Exemplary
- Data fully described (source, size, collection method, quality checks)
- Missing data handling transparent with sensitivity analysis
- Code and data available (or clear plan for release)
- All hyperparameters and random seeds reported
- Preprocessing pipeline fully documented

### 7-9/10 - Strong
- Data well described
- Missing data handling reported
- Key implementation details provided
- Reproducibility feasible with effort

### 4-6/10 - Adequate
- Basic data description
- Some implementation details missing
- Reproducibility challenging

### 1-3/10 - Weak
- Vague data description
- Missing data handling unclear
- Implementation details insufficient
- Not reproducible

**Red flags for low scores**:
- "We processed the data appropriately" (no details)
- No mention of missing data
- "We tuned hyperparameters" (no specifics)
- Synthetic data only, claiming real-world applicability

---

## Dimension 3: Generalization & External Validity (Weight: 15%)

### 10/10 - Exemplary
- Tested on multiple diverse contexts
- Boundary conditions explicitly stated
- Out-of-distribution validation performed
- Generalization claims appropriately scoped

### 7-9/10 - Strong
- Tested on reasonably diverse data
- Generalization scope stated
- Limitations acknowledged

### 4-6/10 - Adequate
- Single-context evaluation
- Generalization scope somewhat vague
- Some overgeneralization

### 1-3/10 - Weak
- Single-context evaluation with broad claims
- "Works for all X" without evidence
- Synthetic-only validation claiming real-world applicability

**Red flags for low scores**:
- "Our model generalizes to all cities" (tested on one)
- No discussion of context-specificity
- Ignoring deployment challenges

---

## Dimension 4: Contribution & Novelty (Weight: 15%)

### 10/10 - Exemplary
- Clear, significant advance over prior work
- Novel methodology or insight
- Addresses important gap
- Contribution well-articulated

### 7-9/10 - Strong
- Meaningful contribution
- Incremental but valuable advance
- Clear positioning vs prior work

### 4-6/10 - Adequate
- Modest contribution
- Incremental advance
- Positioning could be clearer

### 1-3/10 - Weak
- Unclear what's new
- Replication without novelty
- Contribution not articulated

**Red flags for low scores**:
- "First ever" when prior work exists
- No related work section
- Unclear what's new vs. what's replicated

---

## Dimension 5: Writing Quality & Clarity (Weight: 10%)

### 10/10 - Exemplary
- Crystal clear, well-structured
- Technical terms defined
- Logical flow
- Figures/tables publication-ready
- Abstract conveys key contribution

### 7-9/10 - Strong
- Clear and well-organized
- Minor clarity issues
- Good flow

### 4-6/10 - Adequate
- Understandable but could be clearer
- Some organizational issues
- Jargon not always defined

### 1-3/10 - Weak
- Unclear or confusing
- Poor organization
- Jargon-heavy without explanation
- Abstract doesn't convey contribution

**Red flags for low scores**:
- "We processed the data appropriately" (vague)
- Undefined acronyms
- Inconsistent terminology
- Weak abstract

---

## Dimension 6: Policy/Practical Relevance (Weight: 10%)

### 10/10 - Exemplary
- Clear practical implications
- Actionable recommendations
- Feasibility discussed
- Equity impacts considered
- Cost-benefit analysis (if applicable)

### 7-9/10 - Strong
- Practical implications clear
- Recommendations provided
- Feasibility mentioned

### 4-6/10 - Adequate
- Some practical relevance
- Recommendations vague
- Feasibility not discussed

### 1-3/10 - Weak
- Unclear practical relevance
- No actionable recommendations
- Ignores implementation challenges

**Red flags for low scores**:
- "This is important" (no explanation why)
- No discussion of implementation
- Ignoring equity/justice implications
- Unrealistic recommendations

---

## Domain-Specific Criteria

### For Green Transition Research

Additional considerations:
- [ ] Lifecycle emissions considered (not just operational)
- [ ] Rebound effects discussed
- [ ] Equity impacts analyzed
- [ ] Baseline clearly defined ("compared to what?")
- [ ] Technology limitations acknowledged

### For Traffic Safety Research

Additional considerations:
- [ ] Exposure controlled (per VMT, per trip, per hour)
- [ ] Severity stratified (fatal, injury, PDO)
- [ ] Surrogate measures validated against crashes
- [ ] Behavioral theory grounded (TPB, Risk Compensation, etc.)
- [ ] Ethical considerations addressed

### For Machine Learning Applications

Additional considerations:
- [ ] Baseline comparisons (naive, simple, strong)
- [ ] Train/validation/test split proper
- [ ] Overfitting checks (train vs validation curves)
- [ ] Interpretability validated (not just claimed)
- [ ] Fairness analysis (if applicable)
- [ ] Deployment gap discussed

---

## Scoring Worksheet

| Dimension | Weight | Raw Score (1-10) | Weighted Score |
|-----------|--------|------------------|----------------|
| Scientific Rigor | 30% | ___ | ___ |
| Data Quality & Reproducibility | 20% | ___ | ___ |
| Generalization & External Validity | 15% | ___ | ___ |
| Contribution & Novelty | 15% | ___ | ___ |
| Writing Quality & Clarity | 10% | ___ | ___ |
| Policy/Practical Relevance | 10% | ___ | ___ |
| **Total** | **100%** | | **___** |

---

## Decision Matrix

| Total Score | Decision | Typical Revision Time |
|-------------|----------|----------------------|
| 9.0-10.0 | Accept with minor edits | 1-2 weeks |
| 7.0-8.9 | Minor revision | 1-2 months |
| 5.0-6.9 | Major revision | 3-6 months |
| 3.0-4.9 | Reject & resubmit | 6-12 months |
| 1.0-2.9 | Reject | Not salvageable |

---

## Fatal Flaw Override

Regardless of overall score, **immediate rejection** if any of these are present:

- [ ] Causal claims without causal design (and no acknowledgment)
- [ ] Data fabrication or integrity red flags
- [ ] Ethical violations (missing consent, privacy breach)
- [ ] Plagiarism or undisclosed self-plagiarism
- [ ] Fundamental methodology errors (wrong test, violated assumptions)

---

## Venue-Specific Thresholds

### Nature/Science
- Minimum score: 8.5
- Novelty must be 9+
- Rigor must be 9+
- No fatal flaws, no major weaknesses

### Transportation Research Part A/B/C
- Minimum score: 7.0
- Rigor must be 7+
- Reproducibility must be 7+
- Policy relevance must be 6+

### Accident Analysis & Prevention
- Minimum score: 7.0
- Rigor must be 8+ (safety-critical)
- Data quality must be 7+
- Ethical considerations required

### IEEE Transactions on ITS
- Minimum score: 7.0
- Technical depth must be 7+
- Reproducibility must be 8+
- Baseline comparisons required

---

## Calibration Examples

### Example 1: High Score (8.5/10)

**Strengths**:
- RCT design with proper randomization
- Multiple cities tested
- Code and data available
- Clear contribution
- Honest limitations discussion

**Weaknesses**:
- Minor: some figures could be clearer
- Minor: one confounding variable not fully addressed

**Decision**: Accept with minor revision

---

### Example 2: Medium Score (6.0/10)

**Strengths**:
- Interesting research question
- Reasonable methodology
- Clear writing

**Weaknesses**:
- Single-city evaluation with broad claims
- No baseline comparisons
- Missing data handling unclear
- Contribution not well-positioned

**Decision**: Major revision required

---

### Example 3: Low Score (3.5/10)

**Strengths**:
- Timely topic

**Weaknesses**:
- Causal claims from observational data
- Confounders ignored
- Synthetic data only, claiming real-world applicability
- No uncertainty quantification
- Vague methodology

**Decision**: Reject & resubmit (needs fundamental rework)

---

## Using This Rubric

1. **Score each dimension independently** (1-10)
2. **Calculate weighted total**
3. **Check for fatal flaws** (override if present)
4. **Map to decision** (accept/minor/major/reject)
5. **Provide specific feedback** for each dimension
6. **Suggest concrete improvements** for low-scoring dimensions

Remember: The goal is not to reject, but to **elevate work to publication standard**.
