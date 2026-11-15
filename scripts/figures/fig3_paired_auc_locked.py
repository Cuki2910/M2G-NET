"""Fig 3: macro-AUC paired delta (M2G-NET minus comparator), 20 dots = independent generator datasets (evaluation seeds 1711-1730)."""
from pathlib import Path
import altair as alt
import pandas as pd
import vl_convert as vlc

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "docs/paper/extension-v3/per-set-locked.csv"
OUT = ROOT / "docs/paper/fjcai/figures"
XMIN, XMAX = -0.02, 0.04  # shared axis; larger deltas shown as labelled arrows (no dots)
NAMES = {"single_task_mlp": "Independent MLP", "early_fusion_mlp": "Shared one-hot MLP",
         "logistic_regression": "Logistic regression", "catboost": "CatBoost"}
ORDER = list(NAMES.values())
MODEL = "m2g_net_v3"  # internal result label of the evaluated M2G-NET configuration
from style import BLUE, ORANGE, INK

d = pd.read_csv(SRC)
wide = d.pivot_table(index=["mechanism", "generator_seed"], columns="model", values="roc_auc")
rows = [(m, s, NAMES[k], wide.loc[(m, s), MODEL] - wide.loc[(m, s), k])
        for (m, s) in wide.index for k in NAMES]
rep = pd.DataFrame(rows, columns=["mechanism", "generator_seed", "comparator", "roc_auc"])
counts = rep.groupby(["mechanism", "comparator"]).size()
assert len(counts) == 3 * len(NAMES) and counts.eq(20).all(), "Expected twenty datasets per mechanism and comparator"
assert rep.roc_auc.notna().all()
rep["mechanism"] = rep.mechanism.str.capitalize()
mean = rep.groupby(["mechanism", "comparator"], as_index=False).roc_auc.mean()
mean["label"] = mean.roc_auc.map(lambda v: f"{v:+.4f}")
off = mean[mean.roc_auc > XMAX].assign(x=XMAX)
key = ["mechanism", "comparator"]
rep = rep[~rep.set_index(key).index.isin(off.set_index(key).index)]
mean = mean[mean.roc_auc <= XMAX]

y = alt.Y("comparator:N", sort=ORDER, title=None)
col = alt.condition(alt.datum.roc_auc > 0, alt.value(BLUE), alt.value(ORANGE))
TITLE = "ΔAUC = AUC(M2G-NET) − AUC(comparator)     ← comparator better | M2G-NET better →"


def panel(m, last):
    x = alt.X("roc_auc:Q", scale=alt.Scale(domain=[XMIN, XMAX], clamp=True),
              axis=alt.Axis(format="+.2f", tickCount=4, title=TITLE if last else None, labels=True))
    f = lambda df: df[df.mechanism == m]
    return alt.layer(
        alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=INK, strokeWidth=1).encode(x="z:Q"),
        alt.Chart(f(rep)).mark_circle(size=26, opacity=0.8).encode(x=x, y=y, color=col),
        alt.Chart(f(mean)).mark_tick(thickness=2.5, size=13, color=INK).encode(x="roc_auc:Q", y=y),
        alt.Chart(f(off)).mark_text(align="right", dx=-3, fontSize=8, color=INK).encode(x="x:Q", y=y, text="label:N"),
        alt.Chart(f(off)).mark_text(align="left", dx=-1, text="▶", fontSize=7, color=INK).encode(x="x:Q", y=y),
    ).properties(width=337, height=alt.Step(13),
                 title=alt.TitleParams(m, anchor="start", fontSize=9, fontWeight="bold"))


mechs = ["Additive", "Interaction", "Partial"]
fig = (alt.vconcat(*[panel(m, m == mechs[-1]) for m in mechs], spacing=8)
       .resolve_scale(y="independent")
       .configure(font="Helvetica", background="white")
       .configure_axis(labelFontSize=8, titleFontSize=8, labelColor=INK, titleColor=INK, titleFontWeight="normal",
                       gridColor="#e6e6e6", gridWidth=0.5, domainColor="#999999", tickColor="#999999")
       .configure_view(stroke=None))
spec = fig.to_dict()
(OUT / "fig3_paired_auc.png").write_bytes(vlc.vegalite_to_png(spec, scale=300 / 72))
print("wrote", OUT / "fig3_paired_auc.png")
