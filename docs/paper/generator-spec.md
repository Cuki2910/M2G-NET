# P09: architecture-independent generator specification

Date: 2026-10-15. Inputs: `protocol-draft.md`, configured feature/label schema,
and `src/data_pipeline.py:22`. This document proposes a precise generator;
it is not evidence of generation. On 2026-10-15 the user explicitly delegated
technical approval ("cứ tiếp tục liên tục, tự duyệt"). Self-review accepted
this design and P08's prior-0.1 budget/isolated-sharing withdrawal before P10.
This approval is not a claim of experimental or organizer verification.

## Purpose and fixed site design

Generate categorical inputs and four eligible binary endpoints directly from
prespecified equations, without fitting or calling any prediction model.
No coefficient is estimated from the motivating article or historical CSVs.
The task names are synthetic endpoints, not approved empirical recodings.

Default N=3,000, S=12, exactly N/S=250 encounters per site, site IDs s=0,...,11.
For other test sizes require positive N divisible by S, S=12; reject invalid
inputs rather than changing the design silently. Permute a balanced site list
before rowwise feature generation. Sites are fixed across dataset replicates:

| Feature | Fixed value for site s |
| --- | --- |
| has_signal | Yes for s<8, otherwise No. |
| num_lanes | 2+(s mod 3). |
| intersection_type | Cross for even s, T for odd s. |
| site outcome offset u_sk | 0.25*sin(2*pi*(s+1)*(k+1)/13), k=0,...,3. |

These offsets are audit-only parameters; predictors may use site IDs but not
the offsets. Replicates vary encounters/outcomes conditional on this fixed site
configuration; no between-site-population generalization is implied.

## Feature law and numeric contrasts

Conditional on the balanced site list, draw these features independently; all
probabilities are explicit design choices, not empirical estimates:

| Feature | Categories and probabilities | Contrast for equations |
| --- | --- | --- |
| rider_type | Delivery .30, RideHailing .30, Private .40 | r=1,0,-1 respectively. |
| gender | Female .40, Male .60 | g=-1,1. |
| age_group | Young .35, Middle .45, Older .20 | a=1,0,-1. |
| police_presence | No .70, Yes .30 | p=-1,1. |
| traffic_condition | Light .25, Moderate .45, Heavy .30 | t=-1,0,1. |
| weather | Dry .80, Rain .20 | w=-1,1. |
| time_slot | OffPeak .50, Peak .50 | q=-1,1. |
| weekend | No .70, Yes .30 | d=-1,1. |
| num_lanes | Fixed by site | l=num_lanes-3. |
| intersection_type | Fixed by site | c=1 for Cross, -1 for T. |

Do not add an unobserved outcome-driving rowwise factor. Shared covariates and
fixed sites may induce marginal task dependence; conditional outcome draws
are independent given all generated predictors. No target-correlation matching.

## Eligibility before observation

Independently draw red_phase_encounter ~ Bernoulli(.75) at signalized sites
(zero elsewhere) and made_turn ~ Bernoulli(.70). These are generated events,
not inferred from signal presence or labels. Define

```text
E_red = (has_signal == Yes) * red_phase_encounter
E_turn = made_turn
E_helmet = E_phone = 1
```

Eligibility precedes outcome and observation masking. A rider without a
red-phase encounter is not an observed red-light negative; a non-turner is not
an observed turn-signal negative. The two unrestricted endpoints are defined
synthetic helmet non-use and handheld-phone presence, not claims of verified
active phone use or a recoding of the source study's categories.

## Outcome law

Let k order tasks as in `config.TASK_NAMES`. Form the additive logit

```text
eta_ik = b_k + B_rk*r + B_ak*a + B_gk*g + B_tk*t + B_pk*p
         + B_wk*w + B_qk*q + B_dk*d + B_lk*l + B_ck*c + u_sk
```

| Task | b | B_r | B_a | B_g | B_t | B_p | B_w | B_q | B_d | B_l | B_c |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| red_light_running | -.50 | .35 | .25 | .10 | .35 | -.35 | .10 | .20 | .10 | .10 | .10 |
| no_turn_signal | -.40 | .25 | .20 | .10 | .30 | -.20 | .15 | .15 | .05 | .10 | -.10 |
| helmet_nonuse | -1.00 | -.20 | .25 | .10 | .10 | -.40 | .10 | .05 | .15 | -.10 | .10 |
| mobile_phone_use | -.90 | .30 | .15 | .10 | .20 | -.20 | -.10 | .20 | .05 | .10 | -.10 |

For mechanism `additive`, use eta above. For `interaction` and `partial`, add
G_1k*r*t + G_2k*a*p + G_3k*r*w, with rows in task order:

```text
G = [[1.20, -0.70,  0.50],
     [-1.00, 0.60,  0.80],
     [0.60, -1.00, -0.50],
     [1.00,  0.50, -0.80]]
```

Set P_ik=1/(1+exp(-eta_ik)); draw Y_ik=1[U_ik<P_ik] with independent uniform
U_ik. For E=0 store oracle Y/P as missing rather than implying applicability.
Coefficients remain fixed across replicates; no search for a favorable label
balance, task correlation, model ranking, or interaction strength.

## Observation law and scientific interpretation

For `additive` and `interaction`, R=1 on eligible labels. For `partial`, use

```text
C = [-.40, -.10, .20, .40]
rho_ik = .75 + .20*sigmoid(C_k + .50*r - .40*t + .30*q + .20*g)
R_ik = 1[V_ik < rho_ik] on eligible labels; zero when E=0
M_ik = E_ik * R_ik
```

Thus eligible observation probability lies strictly between .75 and .95,
depends on observed inputs rather than hidden outcomes, and is never used as
a predictor or loss weight. This is one specified covariate-dependent process,
not a general missing-data solution. Complete-case procedures also condition
on the joint eligibility/observation pattern. Their contrast does not isolate
masking alone. Primary scoring uses all eligible test outcomes after fitting,
not only observed labels, as required by the P08 estimand.

## Random streams, development/final separation

Use NumPy Generator/PCG64 from four children of SeedSequence(seed): feature,
eligibility, outcome-uniform, observation-uniform streams, in that fixed order.
Mechanisms using the same seed reuse feature/event/outcome uniforms. Therefore
`interaction` and `partial` have identical latent outcomes and eligibility;
only observation differs. `additive` changes only logits/outcomes relative to
the shared base design. No model state, model output, or repository CSV is read.

Development seed 1601 is reserved for P11. Small implementation checks may use
11/12 with reduced N divisible by 12. Final seeds 1701/1702/1703 remain untouched
until P12 freeze. Training seeds 42/101 and split seed 42 do not drive generator
parameter selection. Record NumPy version; byte identity is asserted within
the recorded environment, not promised across arbitrary library versions.

## Artifact contract and separation

P10 implements one generator under `scripts/`, one small test under `tests/`,
using existing NumPy/pandas plus stdlib. Proposed CLI:

```text
.venv/bin/python scripts/generate_paper_simulation.py --seed SEED --mechanism additive|interaction|partial --rows 3000 --output FRESH_DIRECTORY
```

The output directory must not exist. Never overwrite source CSVs or an earlier
run. Emit:

1. `data.csv`: configured feature/site columns, simulation_row_id=0,...,N-1,
   event/eligibility audit flags, four observed-label columns, four explicit
   `*_observed` masks. Write label NaN whenever M=0; predictor exports contain
   no latent label/probability/logit/coefficient/propensity columns.
2. `oracle.csv`: row/site IDs, eligibility, observation indicators, latent
   eligible labels, true eligible probabilities and eligible observation
   probabilities. This file is never passed to the predictor CLI.
3. `manifest.json`: synthetic-only designation, seed/mechanism/N/S, complete
   parameters/feature law, RNG/library identity, generator/spec hashes, file
   SHA256 values, per-task eligibility/observation/positive/class counts,
   complete-case count and event rates. Real generation time, not a fabricated
   completion timestamp. Record validation status separately from content hashes.

Persist stable row order: pipeline `_row_id` is positional, so later prediction
row_id must match simulation_row_id exactly. P14 validates a one-to-one join
and source hashes before isolated oracle scoring; no silent row reordering.

## Required checks before accepting implementation

- Deterministic data/oracle bytes for identical parameters/environment; changed
  seed changes content. Runtime timestamps need not be identical.
- All required categorical/site fields are valid, finite, nonmissing; IDs unique
  and balanced; site-fixed attributes never change within site.
- Explicit masks binary/nonmissing; observed targets binary; no target revealed
  at M=0; M<=E and red/turn events imply their eligibility rules.
- True probabilities finite and in (0,1) where eligible; eligible observation
  propensities bounded as specified; E=0 gives absent oracle outcome/probability.
- Same-seed interaction/partial features, events and latent outcomes match;
  additive/interaction feature/event law matches without model dependence.
- Feature extraction ignores audit flags, and no oracle fields occur in data.csv.
- Existing `scripts/check_data.py --require-masks` accepts the export without
  changing it. That check alone does not certify true event masks/provenance.
- Output-directory collision, malformed arguments, invalid sizes, and write
  failures stop without replacing an existing dataset. Report incomplete output
  honestly; do not give it a successful validation/completion marker.

## Confirmation gate

This gate was satisfied by delegated technical approval on 2026-10-15,
recorded in P09 state: this specification and P08's one-prior neural
budget/withdrawal of an isolated-sharing claim are accepted. P11 then measures
runtime on development-only data; P12 freezes final seeds/budgets/units with
any substantive changes explicitly recorded before final-test access. No
unfavorable final comparison may be dropped retroactively.
