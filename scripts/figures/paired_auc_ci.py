"""95% t-intervals of paired macro-AUC deltas (M2G-NET minus comparator), 20 datasets -> paired-auc-ci.csv."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

D = Path(__file__).resolve().parents[2] / "docs/paper/extension-v3"
d = pd.read_csv(D / "per-set-locked.csv")
w = d.pivot_table(index=["mechanism", "generator_seed"], columns="model", values="roc_auc")
rows = []
for c in ["single_task_mlp", "early_fusion_mlp", "logistic_regression", "catboost"]:
    for mech, g in w.groupby(level="mechanism"):
        x = (g["m2g_net_v3"] - g[c]).to_numpy()
        assert len(x) == 20
        h = stats.t.ppf(0.975, 19) * x.std(ddof=1) / np.sqrt(20)
        rows.append((c, mech, x.mean(), x.mean() - h, x.mean() + h))
out = pd.DataFrame(rows, columns=["comparator", "mechanism", "mean", "ci_lo", "ci_hi"])
old = pd.read_csv(D / "paired-auc-ci.csv")
assert np.allclose(out.sort_values(["comparator", "mechanism"])[["mean", "ci_lo", "ci_hi"]].to_numpy(),
                   old.sort_values(["comparator", "mechanism"])[["mean", "ci_lo", "ci_hi"]].to_numpy())
out.to_csv(D / "paired-auc-ci.csv", index=False)
print("ok", len(out))
