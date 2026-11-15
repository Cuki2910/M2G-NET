# Mathematical Formulation: M2G-Net v2

This file is the final mathematical reference for M2G-Net v2. It is
intended to be the single source of truth for the current model formulation.
The repository data files are synthetic proof-of-concept datasets unless
replaced by real observational data.

## Notation

Let
\[
\mathcal{D}=\{(x^{(n)},y^{(n)})\}_{n=1}^{N}
\]
denote the training set, where
\[
y^{(n)}=(y_1^{(n)},\ldots,y_K^{(n)})\in\{0,1\}^{K},
\qquad K=4 .
\]
Each input sample is represented as a multi-view tuple
\[
x = (x_1,\ldots,x_V), \qquad V=5,
\]
corresponding to rider role, rider traits, road context, environment, and
site-observed views.

Let \(S_{\mathrm{train}}\) be the number of sites observed during training.
The encoded train-time site identifier satisfies
\[
s\in\{0,\ldots,S_{\mathrm{train}}-1\}.
\]
The implementation reserves one additional unknown-site id,
\(s_{\mathrm{unk}}=S_{\mathrm{train}}\), for sites not seen by the train-time
preprocessing map.
For partially observed outcomes, define
\[
m_k^{(n)}\in\{0,1\},
\]
where \(m_k^{(n)}=1\) only when task \(k\)'s label is observed and applicable
for sample \(n\).

## 1. View-Specific Encoding

Each view is encoded by a view-specific embedding and MLP encoder:
\[
h_i = f_i(x_i),
\qquad
h_i\in\mathbb{R}^{d},
\quad i=1,\ldots,V .
\]
The individual and contextual representations are
\[
h_{\mathrm{ind}} = [h_{\mathrm{role}};h_{\mathrm{traits}}],
\qquad
h_{\mathrm{ctx}} = [h_{\mathrm{road}};h_{\mathrm{env}};h_{\mathrm{site}}].
\]

## 2. Site-Aware Representation

Let \(s\) denote the site identifier and let \(h_{\mathrm{obs}}\) denote the
encoded observed site-level covariates from the site-observed view. The
site-aware representation uses a regularized site-specific embedding:
\[
h_{\mathrm{site}}
=
\phi\!\left([h_{\mathrm{obs}};h_{\mathrm{rand}}]\right),
\qquad
h_{\mathrm{rand}} = E_{\mathrm{site}}(s),
\]
where \(E_{\mathrm{site}}\) is a learnable site embedding trained with weight
decay. This component acts as a site-specific intercept-like representation,
but it should not be interpreted as a full probabilistic mixed-effects model.
For true out-of-site evaluation, the site-specific component is disabled:
\[
h_{\mathrm{rand}}=\mathbf{0}.
\]
The site projection function is
\[
\phi:\mathbb{R}^{d_{\mathrm{obs}}+d_{\mathrm{site}}}\rightarrow
\mathbb{R}^{d}.
\]

## 3. Cross-Level Interaction

The model captures interactions between individual-level and contextual-level
factors using projected additive and multiplicative terms:
\[
p_{\mathrm{ind}} = W_{\mathrm{ind}}h_{\mathrm{ind}},
\qquad
p_{\mathrm{ctx}} = W_{\mathrm{ctx}}h_{\mathrm{ctx}},
\]
with
\[
W_{\mathrm{ind}}\in\mathbb{R}^{d\times 2d},
\qquad
W_{\mathrm{ctx}}\in\mathbb{R}^{d\times 3d}.
\]
\[
h_{\mathrm{int}}
=
\psi\!\left(
p_{\mathrm{ind}} + p_{\mathrm{ctx}} + W_{\mathrm{int}}
\left(p_{\mathrm{ind}}\odot p_{\mathrm{ctx}}\right)
\right),
\qquad
h_{\mathrm{int}}\in\mathbb{R}^{d},
\]
where \(\odot\) denotes element-wise multiplication.
Here
\[
W_{\mathrm{int}}\in\mathbb{R}^{d\times d},
\qquad
\psi:\mathbb{R}^{d}\rightarrow\mathbb{R}^{d}.
\]

## 4. Task-Specific Gated Fusion

For task \(k\), gated attention is computed over the view representations and
the cross-level interaction representation. Let
\[
\tilde{h}_j=h_j,\quad j=1,\ldots,V,
\qquad
\tilde{h}_{V+1}=P_{\mathrm{int}}h_{\mathrm{int}},
\]
where
\[
P_{\mathrm{int}}\in\mathbb{R}^{d\times d}.
\]
All \(\tilde{h}_j\) therefore have the same dimensionality \(d\). Define
\[
r=[\tilde{h}_1;\ldots;\tilde{h}_{V+1}]\in\mathbb{R}^{(V+1)d}.
\]
The task-specific attention weights are
\[
\alpha_k^{\mathrm{raw}}
=
\operatorname{sparsemax}
\left(
\frac{A_k r + c_k}{T}
\right),
\qquad
\alpha_k
=
(1-\lambda)\alpha_k^{\mathrm{raw}}+\lambda p_0,
\]
where
\[
A_k\in\mathbb{R}^{(V+1)\times (V+1)d},
\qquad
c_k\in\mathbb{R}^{V+1}.
\]
The uniform prior used by smoothing is
\[
p_0=\frac{1}{V+1}\mathbf{1}_{V+1},
\qquad
0\le\lambda\le 1.
\]
This sparsemax uniform prior mixing keeps the attention distribution normalized
while giving every gate input a lower bound when \(\lambda>0\):
\[
\alpha_{k,j}\ge
\frac{\lambda}{V+1}.
\]
The gate temperature \(T>0\) remains active and controls the sharpness of the
raw attention distribution before smoothing.
The gated representation for task \(k\) is
\[
z_{\mathrm{gated}}^{(k)}
=
\sum_{j=1}^{V+1}\alpha_{k,j}\tilde{h}_j .
\]

### Development gate variants (v3 gate search)

The default `legacy` gate above is unchanged. `nonlinear` and
`nonlinear_task_blend` replace only its affine logits with
\[
q_k=B_{2,k}\,\operatorname{GELU}(B_{1,k}r+b_{1,k})+b_{2,k},
\quad B_{1,k}\in\mathbb R^{H\times Jd},\quad
B_{2,k}\in\mathbb R^{J\times H},
\]
where \(J\) is the number of active gate inputs and
\(H=\texttt{V3\_GATE\_HIDDEN\_DIM}\). Apply the same sparsemax to
\(q_k/T\), then the same uniform prior mixing. For finite inputs,
\(T\ge0.1\) and \(0\le\lambda\le1\), the weights have shape
\([B,J]\), sum to one, and satisfy \(\alpha_{k,j}\ge\lambda/J\).
These optional variants do not apply to `early_fusion` (no gated path).

## 5. Residual Early-Fusion Blending

The early-fusion baseline representation is
\[
z_{\mathrm{early}} = g([h_1;\ldots;h_V]).
\]
where
\[
g:\mathbb{R}^{Vd}\rightarrow\mathbb{R}^{d}.
\]
M2G-Net combines early fusion and task-specific gated fusion using a
learnable scalar blending coefficient:
\[
z_k
=
\operatorname{sigmoid}(a)z_{\mathrm{gated}}^{(k)} +
\left(1-\operatorname{sigmoid}(a)\right)z_{\mathrm{early}},
\]
where \(a\in\mathbb{R}\) is a shared scalar learned jointly with the model
parameters. The development `task_blend` and `nonlinear_task_blend` variants
instead learn \(a\in\mathbb R^K\) and use \(\operatorname{sigmoid}(a_k)\)
for task \(k\) in both terms. Both scalar and vector start at zero (blend
0.5); the vector has separate task gradients and is not training-equivalent
to the shared scalar. The returned blend is scalar for `legacy`/`nonlinear`
and shape \([K]\) for task-specific blending.

## 6. Task Prediction

For each task \(k\), the prediction head produces a binary probability:
\[
\hat{y}_k
=
\operatorname{sigmoid}\!\left(w_k^{\top}z_k+\beta_k\right),
\qquad k=1,\ldots,K .
\]

**m2g_net_v3 option (`V3_INTERACTION_SKIP`, default off).** When enabled, the interaction
representation also enters each logit directly, bypassing the gate and the blend:
\[
\hat{y}_k
=
\operatorname{sigmoid}\!\left(w_k^{\top}z_k+\beta_k+s_k^{\top}h_{\mathrm{int}}\right),
\qquad s_k\in\mathbb{R}^{d},\; s_k=0 \text{ at initialisation}.
\]
With \(s_k=0\) the model equals the formula above. The option requires the interaction
and gated paths (not the `no_interaction` or `early_fusion` ablations).

## 7. Training Objective

For sample \(n\) and task \(k\), the binary cross-entropy loss is
\[
\ell_k^{(n)}
=
-y_k^{(n)}\log \hat{y}_k^{(n)} -
\left(1-y_k^{(n)}\right)
\log\left(1-\hat{y}_k^{(n)}\right).
\]
The default model is trained using a masked uncertainty-weighted multi-task
objective adapted from Kendall et al.~\cite{kendall2018multi}. For numerical
stability, the implementation learns
\[
s_k=\log \tau_k^2,
\]
where \(\tau_k>0\) is the task uncertainty. Define the observed-label count
and masked mean task loss in a mini-batch as
\[
M_k=\sum_{n=1}^{N}m_k^{(n)},
\qquad
\bar{\ell}_k=
\begin{cases}
\frac{1}{M_k}\sum_{n=1}^{N}m_k^{(n)}\ell_k^{(n)}, & M_k>0,\\
\text{undefined}, & M_k=0.
\end{cases}
\]
The implemented loss skips task \(k\) in mini-batches where \(M_k=0\):
\[
\mathcal{L}
=
\sum_{k:M_k>0}
\left[
\frac{1}{2}\exp(-s_k)
\bar{\ell}_k

+ \frac{1}{2}s_k
  \right],
  \]
  with \(\exp(-s_k)=1/\tau_k^2\). A fully unnormalized masked objective would be
  \[
  \mathcal{L}_{\mathrm{sum}}
  =
  \sum_{n=1}^{N}\sum_{k=1}^{K}
  m_k^{(n)}
  \left[
  \frac{1}{2}\exp(-s_k)\ell_k^{(n)}
+ \frac{1}{2}s_k
  \right].
  \]
  This is not generally equivalent to the implemented normalized objective,
  because it scales each task by \(M_k\) and gives the uncertainty term one
  contribution per observed label rather than one contribution per active task.
  Focal loss is enabled and positive-class weighting is disabled in the current
  configuration. The `m2g_net_v3` variant keeps every
  equation in Sections 1-6 and 8, replaces the focal term by the plain binary
  cross-entropy \(\ell_k^{(n)}\) above (`V3_USE_FOCAL_LOSS=False`), and averages
  \(M=\)`V3_ENSEMBLE_SIZE` independently initialized members in probability space:
  \[
  \hat y_k^{\mathrm{v3}}=\frac1M\sum_{m=1}^{M}\hat y_k^{(m)},\qquad
  \operatorname{seed}_m=\operatorname{seed}+m\cdot\texttt{V3\_MEMBER\_SEED\_OFFSET}.
  \]
  Each member early-stops on its own validation macro ROC-AUC; the ensemble
  validation AUC is reported but not used to select among variants. Targets are indexed by the observation mask before evaluating
  BCE/focal loss so missing NaNs cannot contaminate the objective.

### Controlled retrained ablations

The default architecture and objective above are unchanged. Every ablation is
trained from a fresh initialization on matched train/validation/test rows:

- `no_interaction`: omit the interaction input and its projection from the gate,
  leaving five normalized gate weights; retain early fusion over five views.
- `no_site`: replace the site representation with zero before both interaction
  and early fusion, and omit that view from the gate (five remaining inputs).
- `shared_gate`: tie the gate across tasks; preserve task-specific output heads.
- `early_fusion`: use only the early-fusion path, without gated fusion.
- `no_uncertainty`: replace the uncertainty-weighted sum with the sum of masked
  mean task losses, with no learned uncertainty contribution.
- `complete_case`: retain only fully observed training and validation records;
  evaluate on the original masked test set. This comparator never relabels
  missing outcomes as observed negatives and also changes sample size.

Calibration temperature is fitted on observed validation outcomes only.
Decision thresholds are selected from those calibrated validation probabilities,
then held fixed when transforming and classifying test probabilities. Both raw
and calibrated metrics are reported. Site-out validation uses sites disjoint
from training and disables site-specific embeddings in validation and test.

## 8. Optimization

The objective is optimized with Adam and weight decay. The learning rate and
gate temperature are scheduled using cosine annealing:
\[
q(e)
=
q_{\min} +
\frac{1}{2}
\left(q_{\max}-q_{\min}\right)
\left(1+\cos\frac{\pi e}{E}\right),
\]
where \(e\) is the current epoch, \(E\) is the total number of epochs, and
\(q\) is either the learning rate or the gate temperature.

## 9. Evaluation

The primary evaluation metric is macro-averaged ROC-AUC:
\[
\operatorname{AUC}_{\mathrm{macro}}
=
\frac{1}{K}
\sum_{k=1}^{K}\operatorname{AUC}_k .
\]
Here \(\operatorname{AUC}_k\) denotes the ROC-AUC for binary task \(k\),
computed only over test samples with \(m_k^{(n)}=1\). If a task has no observed
labels or only one class in the evaluated split, its metric is reported as
undefined and omitted from finite macro averages.

For thresholded metrics, the predicted label for task \(k\) is
\[
\tilde{y}_k^{(n)}(\theta_k)
=
\mathbf{1}\{\hat{y}_k^{(n)}\ge \theta_k\}.
\]
The default threshold is \(\theta_k=0.5\). The tuned threshold is selected on
validation predictions by grid search:
\[
\theta_k^\star
=
\arg\max_{\theta\in\Theta}
F_1\!\left(y_k,\tilde{y}_k(\theta)\right),
\qquad
\Theta=\{0.01,0.02,\ldots,0.99\}.
\]
ROC-AUC and PR-AUC remain threshold-free; tuned thresholds affect only
decision metrics such as F1, sensitivity, specificity, and balanced accuracy.
For a binary confusion matrix with true positives \(TP\), false positives
\(FP\), true negatives \(TN\), and false negatives \(FN\),
\[
F_1=\frac{2TP}{2TP+FP+FN},
\qquad
\operatorname{BalAcc}
=
\frac{1}{2}
\left(
\frac{TP}{TP+FN} +
\frac{TN}{TN+FP}
\right).
\]
The Brier score and expected calibration error are
\[
\operatorname{Brier}_k
=
\frac{1}{M_k}
\sum_{n:m_k^{(n)}=1}
\left(\hat{y}_k^{(n)}-y_k^{(n)}\right)^2,
\]
and, for calibration bins \(B_b\),
\[
\operatorname{ECE}_k
=
\sum_b
\frac{|B_b|}{M_k}
\left|
\operatorname{acc}(B_b)-\operatorname{conf}(B_b)
\right|,
\]
where \(\operatorname{conf}(B_b)\) is the mean predicted probability in bin
\(B_b\), and \(\operatorname{acc}(B_b)\) is the empirical positive rate in that
bin.

For \(R\) repeated runs with paired scores \(a_r\) for M2G-Net and \(b_r\) for
a baseline, the reported mean and 95% confidence interval are
\[
\bar{a}=\frac{1}{R}\sum_{r=1}^{R}a_r,
\qquad
\bar{a}\pm
t_{0.975,R-1}\frac{s_a}{\sqrt{R}},
\]
where \(s_a\) is the sample standard deviation across runs. Statistical
significance is tested with a paired \(t\)-test on run-wise differences
\(d_r=a_r-b_r\):
\[
t=
\frac{\bar{d}}{s_d/\sqrt{R}},
\qquad
\bar{d}=\frac{1}{R}\sum_{r=1}^{R}d_r.
\]
This test is appropriate only when runs are paired by the same seed/split and
the run-wise differences are reasonably symmetric. With very small \(R\), the
\(p\)-value should be treated as supporting evidence rather than a definitive
claim.

For site generalization, leave-site-out evaluation trains a fresh model for
each held-out site \(s\). Let \(\mathcal{D}_{-s}\) be all samples except site
\(s\), and \(\mathcal{D}_{s}\) the held-out samples from site \(s\). The model
is trained on \(\mathcal{D}_{-s}\), evaluated on \(\mathcal{D}_s\), and the
site-specific random component is disabled:
\[
h_{\mathrm{rand}}=\mathbf{0}
\quad\text{on}\quad
\mathcal{D}_{s}.
\]
If \(G_s\) is the macro ROC-AUC on held-out site \(s\), the leave-site-out
summary is
\[
\bar{G}_{\mathrm{LSO}}
=
\frac{1}{|\mathcal{S}_{\mathrm{eval}}|}
\sum_{s\in\mathcal{S}_{\mathrm{eval}}}G_s,
\qquad
s_G
=
\sqrt{
\frac{1}{|\mathcal{S}_{\mathrm{eval}}|-1}
\sum_{s\in\mathcal{S}_{\mathrm{eval}}}
\left(G_s-\bar{G}_{\mathrm{LSO}}\right)^2
}.
\]
The site-generalization gap can be reported against the ordinary random-split
macro ROC-AUC \(G_{\mathrm{random}}\):
\[
\Delta_{\mathrm{site}}
=
G_{\mathrm{random}}-\bar{G}_{\mathrm{LSO}}.
\]
A larger positive gap indicates more performance loss when transferring to
unseen sites.

## 10. Code Mapping

| Component               | Implementation       |
| ----------------------- | -------------------- |
| View-specific encoders  | `src/views.py`       |
| Site-aware encoder      | `src/views.py`       |
| Cross-level interaction | `src/interaction.py` |
| Gated/residual fusion   | `src/fusion.py`      |
| Task prediction heads   | `src/model.py`       |
| Multi-task objective    | `src/loss.py`        |
| Evaluation metrics      | `src/metrics.py`     |

## BibTeX

```bibtex
@inproceedings{kendall2018multi,
  title     = {Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics},
  author    = {Kendall, Alex and Gal, Yarin and Cipolla, Roberto},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages     = {7482--7491},
  year      = {2018}
}
```

