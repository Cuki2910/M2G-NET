"""Fig 3: macro-AUC paired delta (M2G-NET minus comparator), 3 dots = generator replicates."""
import sys
from pathlib import Path
import altair as alt
import pandas as pd
import vl_convert as vlc

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "docs/paper"
OUT = PAPER / "figures"
XMIN, XMAX = -0.01, 0.02  # shared axis; larger deltas shown as labelled arrows (no dots)
NAMES = {
    "no_interaction": "No interaction", "shared_gate": "Shared gate",
    "early_fusion": "Early fusion", "single_task_mlp": "Independent MLP",
    "early_fusion_mlp": "Shared one-hot MLP", "logistic_regression": "Logistic regression",
    "catboost": "CatBoost", "complete_case": "Complete case",
}
ORDER = list(NAMES.values())
from style import BLUE, ORANGE, INK

d = pd.read_csv(PAPER / "paired-deltas.csv")
d = d[d.task == "macro"]
rep = d.groupby(["mechanism", "model", "generator_seed"], as_index=False).roc_auc.mean()
rep["comparator"] = rep.model.map(NAMES)
rep["mechanism"] = rep.mechanism.str.capitalize()
mean = rep.groupby(["mechanism", "comparator"], as_index=False).roc_auc.mean()
mean["label"] = mean.roc_auc.map(lambda v: f"{v:+.4f}")
off = mean[mean.roc_auc > XMAX].assign(x=XMAX)
rep = rep[~rep.set_index(["mechanism", "comparator"]).index.isin(off.set_index(["mechanism", "comparator"]).index)]
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
        alt.Chart(f(rep)).mark_circle(size=34, opacity=0.85).encode(x=x, y=y, color=col),
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
OUT.mkdir(exist_ok=True)
(OUT / "fig3_paired_auc.svg").write_text(vlc.vegalite_to_svg(spec))
(OUT / "fig3_paired_auc.png").write_bytes(vlc.vegalite_to_png(spec, scale=300 / 72))
print("wrote fig3_paired_auc.{svg,png}")
