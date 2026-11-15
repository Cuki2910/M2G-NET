"""Condensed main tables (T1-T3) and supplement tables (S1-S3) from the frozen result CSVs.

Writes docs/paper/tables-main.md and docs/paper/tables-supplement.md. Reads CSVs only; does not touch tables.md.
"""
from pathlib import Path
import pandas as pd

P = Path(__file__).resolve().parents[2] / "docs/paper"
NAMES = {
    "m2g_net": "M2G-NET", "no_interaction": "No interaction", "shared_gate": "Shared gate",
    "early_fusion": "Early fusion", "single_task_mlp": "Independent MLP", "early_fusion_mlp": "Shared one-hot MLP",
    "logistic_regression": "Logistic regression", "catboost": "CatBoost", "complete_case": "Complete case",
}
TASKS = {"red_light_running": "Red-light", "no_turn_signal": "Turn-signal", "helmet_nonuse": "Helmet", "mobile_phone_use": "Phone"}
MECHS = ["additive", "interaction", "partial"]
ORDER = list(NAMES)

res = pd.read_csv(P / "results.csv")
summ = pd.read_csv(P / "results-summary.csv")
dsum = pd.read_csv(P / "paired-summary.csv")
bud = pd.read_csv(P / "fit-budgets.csv")


def md(caption, headers, rows):
    out = [caption, "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out) + "\n"


def rng(s, nd=0):
    lo, hi = s.min(), s.max()
    return f"{lo:.0f}" if lo == hi else f"{lo:.0f}–{hi:.0f}"


# T1: denominators (range over the three generator datasets)
t1 = []
for m in MECHS:
    for t, label in TASKS.items():
        g = res[(res.mechanism == m) & (res.task == t)]
        t1.append([m.capitalize(), label, rng(g.n_eligible), rng(g.n_positive), rng(g.n_observed)])
T1 = md("Table 1. Eligible, positive and observed test outcomes per task (range over three generated datasets; "
        "600 test encounters each).", ["Mechanism", "Task", "Eligible", "Positive", "Observed"], t1)

# T2: wide layout, one row per procedure, mechanism blocks of AUC / AP / Brier; best per block column in bold
macro = summ[summ.task == "macro"]
METRIC = [("roc_auc_mean", "AUC", max), ("average_precision_mean", "AP", max), ("brier_mean", "Brier", min)]
g = {m: macro[macro.mechanism == m].set_index("model") for m in MECHS}
t2 = []
for model in ORDER:
    cells = []
    for m in MECHS:
        for col, _, best in METRIC:
            if model not in g[m].index:
                cells.append("–")
                continue
            v = round(g[m].loc[model, col], 3)
            s = f"{v:.3f}"
            cells.append(f"**{s}**" if v == round(best(g[m][col]), 3) else s)
    t2.append([NAMES[model], *cells])
CAP2 = ("Table 2. Macro performance on eligible test outcomes: mean over three generated datasets (each averaged over two "
        "training seeds). Bold: best per mechanism and metric at three decimals. Higher AUC and AP, lower Brier are better.")
head1 = "| | " + " | ".join(f"{m.capitalize()} | |" if False else f"{m.capitalize()} | | " for m in MECHS)
head1 = "| " + " | ".join(["", *[x for m in MECHS for x in (m.capitalize(), "", "")]]) + " |"
head2 = "| " + " | ".join(["Procedure", *["AUC", "AP", "Brier"] * 3]) + " |"
T2 = "\n".join([CAP2, "", head1, "| " + " | ".join(["---"] * 10) + " |", head2,
               *["| " + " | ".join(r) + " |" for r in t2]]) + "\n"

# T3: compact computational comparison
t3 = []
for model in ORDER:
    g = bud[bud.model == model]
    par = g.trainable_model_parameters.dropna()
    p = f"{par.iloc[0] / 1000:.1f}k" if len(par) else "n/a"
    t3.append([NAMES[model], p, int(g.candidate_fits.min()), f"{g.wall_seconds.mean():.1f}"])
T3 = md("Table 3. Computational comparison. Parameters: trainable, including dormant instantiated modules (classical "
        "models: not comparable). Search: candidate fits per unit. Runtime: mean fit-process seconds per fit.",
        ["Procedure", "Parameters", "Search candidates", "Runtime (s)"], t3)

# S1: per-task AUC
s1 = []
for m in MECHS:
    for model in ORDER:
        g = summ[(summ.mechanism == m) & (summ.model == model)].set_index("task")
        if len(g):
            s1.append([m.capitalize(), NAMES[model], *[f"{g.loc[t, 'roc_auc_mean']:.6f}" for t in TASKS]])
S1 = md("Table S1. Per-task mean eligible AUC over three generated datasets; negative comparisons retained.",
        ["Mechanism", "Procedure", *TASKS.values()], s1)

# S2: paired macro AUC differences
d = dsum[dsum.task == "macro"]
s2 = [[r.mechanism.capitalize(), NAMES[r.model], *[f"{r[c]:+.6f}" for c in
       ("roc_auc_mean", "roc_auc_sd", "roc_auc_min", "roc_auc_max", "roc_auc_mcse")]]
      for _, r in d.assign(o=d.model.map(ORDER.index)).sort_values(["mechanism", "o"]).iterrows()]
S2 = md("Table S2. Paired macro AUC differences (M2G-NET minus comparator; positive favors M2G-NET). SD, min, max and "
        "descriptive MCSE use three replicate-level, training-seed-averaged deltas.",
        ["Mechanism", "Comparator", "Mean", "SD", "Min", "Max", "MCSE"], s2)

# S3: macro metrics, full precision with dataset SD and training-seed SD
s3 = [[m.capitalize(), NAMES[model], *[f"{r[c]:.6f}" for metric in ("roc_auc", "average_precision", "brier")
       for c in (f"{metric}_mean", f"{metric}_sd", f"{metric}_mean_training_sd")]]
      for m in MECHS for model in ORDER
      for r in [macro[(macro.mechanism == m) & (macro.model == model)].squeeze()] if not isinstance(r, pd.DataFrame) and len(r)]
S3 = md("Table S3. Macro endpoints at full precision: mean, dataset SD (three generated datasets) and mean within-dataset "
        "training-seed SD (two seeds).", ["Mechanism", "Procedure", "AUC", "AUC SD", "AUC train SD", "AP", "AP SD",
                                          "AP train SD", "Brier", "Brier SD", "Brier train SD"], s3)

(P / "tables-main.md").write_text("# Main tables\n\n" + "\n".join([T1, T2, T3]))
(P / "tables-supplement.md").write_text("# Supplementary tables\n\n" + "\n".join([S1, S2, S3]))
print("wrote tables-main.md, tables-supplement.md")
