# Mathematical Formulation: M2G-Net v2

This file is the final mathematical reference for M2G-Net v2. It is
intended to be the single source of truth for the current model formulation.
The repository data files are synthetic proof-of-concept datasets unless
replaced by real observational data.

## Notation

Let
\[
\mathcal{D}=\{(x^{(n)},y^{(n)})\}\_{n=1}^{N}
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

Let \(S*{\mathrm{train}}\) be the number of sites observed during training.V
The encoded train-time site identifier satisfies
\[
s\in\{0,\ldots,S*{\mathrm{train}}-1\}.
\]
The implementation reserves one additional unknown-site id,
\(s*{\mathrm{unk}}=S*{\mathrm{train}}\), for sites not seen by the train-time
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
h*i = f_i(x_i),
\qquad
h_i\in\mathbb{R}^{d},
\quad i=1,\ldots,V .
\]
The individual and contextual representations are
\[
h*{\mathrm{ind}} = [h_{\mathrm{role}};h_{\mathrm{traits}}],
\qquad
h*{\mathrm{ctx}} = [h*{\mathrm{road}};h*{\mathrm{env}};h*{\mathrm{site}}].
\]

## 2. Site-Aware Representation

Let \(s\) denote the site identifier and let \(h*{\mathrm{obs}}\) denote the
encoded observed site-level covariates from the site-observed view. The
site-aware representation uses a regularized site-specific embedding:
\[
h*{\mathrm{site}}
=
\phi\!\left([h_{\mathrm{obs}};h_{\mathrm{rand}}]\right),
\qquad
h*{\mathrm{rand}} = E*{\mathrm{site}}(s),
\]
where \(E*{\mathrm{site}}\) is a learnable site embedding trained with weight
decay. This component acts as a site-specific intercept-like representation,
but it should not be interpreted as a full probabilistic mixed-effects model.
For true out-of-site evaluation, the site-specific component is disabled:
\[
h*{\mathrm{rand}}=\mathbf{0}.
\]
The site projection function is
\[
\phi:\mathbb{R}^{d*{\mathrm{obs}}+d*{\mathrm{site}}}\rightarrow
\mathbb{R}^{d}.
\]

## 3. Cross-Level Interaction

The model captures interactions between individual-level and contextual-level
factors using projected additive and multiplicative terms:
\[
p*{\mathrm{ind}} = W*{\mathrm{ind}}h*{\mathrm{ind}},
\qquad
p*{\mathrm{ctx}} = W*{\mathrm{ctx}}h*{\mathrm{ctx}},
\]
with
\[
W*{\mathrm{ind}}\in\mathbb{R}^{d\times 2d},
\qquad
W*{\mathrm{ctx}}\in\mathbb{R}^{d\times 3d}.
\]
\[
h*{\mathrm{int}}
=
\psi\!\left(
p*{\mathrm{ind}} + p*{\mathrm{ctx}} + W*{\mathrm{int}}
\left(p*{\mathrm{ind}}\odot p*{\mathrm{ctx}}\right)
\right),
\qquad
h*{\mathrm{int}}\in\mathbb{R}^{d},
\]
where \(\odot\) denotes element-wise multiplication.
Here
\[
W*{\mathrm{int}}\in\mathbb{R}^{d\times d},
\qquad
\psi:\mathbb{R}^{d}\rightarrow\mathbb{R}^{d}.
\]

## 4. Task-Specific Gated Fusion

For task \(k\), gated attention is computed over the view representations and
the cross-level interaction representation. Let
\[
\tilde{h}_j=h_j,\quad j=1,\ldots,V,
\qquad
\tilde{h}_{V+1}=P*{\mathrm{int}}h*{\mathrm{int}},
\]
where
\[
P*{\mathrm{int}}\in\mathbb{R}^{d\times d}.
\]
All \(\tilde{h}\_j\) therefore have the same dimensionality \(d\). Define
\[
r=[\tilde{h}\_1;\ldots;\tilde{h}*{V+1}]\in\mathbb{R}^{(V+1)d}.
\]
The task-specific attention weights are
\[
\alpha*k^{\mathrm{raw}}
=
\operatorname{softmax}
\left(
\frac{A_k r + c_k}{T}
\right),
\qquad
\alpha_k
=
\frac{\alpha_k^{\mathrm{raw}}+\lambda p_0}{1+\lambda},
\]
where
\[
A_k\in\mathbb{R}^{(V+1)\times (V+1)d},
\qquad
c_k\in\mathbb{R}^{V+1}.
\]
The uniform prior used by smoothing is
\[
p_0=\frac{1}{V+1}\mathbf{1}*{V+1},
\qquad
\lambda\ge 0.
\]
This post-softmax prior smoothing keeps the attention distribution normalized
while giving every gate input a lower bound:
\[
\alpha*{k,j}\ge
\frac{\lambda}{(1+\lambda)(V+1)}.
\]
The gate temperature \(T>0\) remains active and controls the sharpness of the
raw attention distribution before smoothing.
The gated representation for task \(k\) is
\[
z*{\mathrm{gated}}^{(k)}
=
\sum*{j=1}^{V+1}\alpha*{k,j}\tilde{h}\_j .
\]

## 5. Residual Early-Fusion Blending

The early-fusion baseline representation is
\[
z*{\mathrm{early}} = g([h_1;\ldots;h_V]).
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
\operatorname{sigmoid}(a)z*{\mathrm{gated}}^{(k)} +
\left(1-\operatorname{sigmoid}(a)\right)z\_{\mathrm{early}},
\]
where \(a\in\mathbb{R}\) is a shared scalar learned jointly with the model
parameters.

## 6. Task Prediction

For each task \(k\), the prediction head produces a binary probability:
\[
\hat{y}\_k
=
\operatorname{sigmoid}\!\left(w_k^{\top}z_k+\beta_k\right),
\qquad k=1,\ldots,K .
\]

## 7. Training Objective

For sample \(n\) and task \(k\), the binary cross-entropy loss is
\[
\ell*k^{(n)}
=
-y_k^{(n)}\log \hat{y}\_k^{(n)} -
\left(1-y_k^{(n)}\right)
\log\left(1-\hat{y}\_k^{(n)}\right).
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
M_k=\sum*{n=1}^{N}m*k^{(n)},
\qquad
\bar{\ell}\_k=
\begin{cases}
\frac{1}{M_k}\sum*{n=1}^{N}m*k^{(n)}\ell_k^{(n)}, & M_k>0,\\
\text{undefined}, & M_k=0.
\end{cases}
\]
The implemented loss skips task \(k\) in mini-batches where \(M_k=0\):
\[
\mathcal{L}
=
\sum*{k:M\*k>0}
\left[
\frac{1}{2}\exp(-s_k)
\bar{\ell}\_k

- \frac{1}{2}s*k
  \right],
  \]
  with \(\exp(-s_k)=1/\tau_k^2\). A fully unnormalized masked objective would be
  \[
  \mathcal{L}*{\mathrm{sum}}
  =
  \sum\_{n=1}^{N}\sum\*{k=1}^{K}
  m_k^{(n)}
  \left[
  \frac{1}{2}\exp(-s_k)\ell_k^{(n)}
- \frac{1}{2}s_k
  \right].
  \]
  This is not generally equivalent to the implemented normalized objective,
  because it scales each task by \(M_k\) and gives the uncertainty term one
  contribution per observed label rather than one contribution per active task.
  Optional focal loss and positive-class weighting are implemented but disabled
  by default in the current configuration.

## 8. Optimization

The objective is optimized with Adam and weight decay. The learning rate and
gate temperature are scheduled using cosine annealing:
\[
q(e)
=
q*{\min} +
\frac{1}{2}
\left(q*{\max}-q\_{\min}\right)
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
\sum_{k=1}^{K}\operatorname{AUC}\_k .
\]
Here \(\operatorname{AUC}\_k\) denotes the ROC-AUC for binary task \(k\),
computed only over test samples with \(m_k^{(n)}=1\). If a task has no observed
labels or only one class in the evaluated split, its metric is reported as
undefined and omitted from finite macro averages.

For thresholded metrics, the predicted label for task \(k\) is
\[
\tilde{y}_k^{(n)}(\theta_k)
=
\mathbf{1}\{\hat{y}\_k^{(n)}\ge \theta_k\}.
\]
The default threshold is \(\theta_k=0.5\). The tuned threshold is selected on
validation predictions by grid search:
\[
\theta_k^\star
=
\arg\max_{\theta\in\Theta}
F*1\!\left(y_k,\tilde{y}\_k(\theta)\right),
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
\operatorname{Brier}\_k
=
\frac{1}{M_k}
\sum*{n:m_k^{(n)}=1}
\left(\hat{y}\_k^{(n)}-y_k^{(n)}\right)^2,
\]
and, for calibration bins \(B_b\),
\[
\operatorname{ECE}\_k
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
