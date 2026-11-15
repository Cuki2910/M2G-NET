"""Fig 4: macro AUC vs Brier per mechanism (mean of 3 generator replicates; y reversed so up-right = better)."""
from pathlib import Path
import altair as alt
import pandas as pd
import vl_convert as vlc
from style import BLUE, GREY, INK, LIGHT, FONT, FS, WIDTH_IN, DPI

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/paper/figures"
NAMES = {
    "m2g_net": "M2G-NET", "no_interaction": "No interaction", "shared_gate": "Shared gate",
    "early_fusion": "Early fusion", "single_task_mlp": "Independent MLP",
    "early_fusion_mlp": "Shared one-hot MLP", "logistic_regression": "Logistic regression",
    "catboost": "CatBoost", "complete_case": "Complete case",
}
r = pd.read_csv(ROOT / "docs/paper/results.csv")
r = r[r.task == "macro"]
d = (r.groupby(["mechanism", "model", "generator_seed"], as_index=False)[["roc_auc", "brier"]].mean()
      .groupby(["mechanism", "model"], as_index=False)[["roc_auc", "brier"]].mean())
d["name"] = d.model.map(NAMES)
d["mechanism"] = d.mechanism.str.capitalize()
d["ours"] = d.model == "m2g_net"


# manual label offsets (dx, dy, align) where neighbours collide
OFFSET = {("Additive", "Shared one-hot MLP"): (-7, 0, "right"), ("Partial", "M2G-NET"): (-9, 0, "right")}


def panel(m, last):
    f = d[d.mechanism == m]
    # label only M2G-NET, best-AUC and best-Brier model; the rest are grey context
    keep = {f.roc_auc.idxmax(), f.brier.idxmin()} | set(f.index[f.ours])
    lab = f.loc[sorted(keep)]
    pad = lambda lo, hi, a, b: [lo - a * (hi - lo), hi + b * (hi - lo)]
    xd = pad(f.roc_auc.min(), f.roc_auc.max(), 0.08, 0.45)
    yd = pad(f.brier.min(), f.brier.max(), 0.15, 0.15)
    x = alt.X("roc_auc:Q", scale=alt.Scale(domain=xd), axis=alt.Axis(format=".3f", tickCount=4,
              title="Macro AUC (higher is better) →" if last else None))
    y = alt.Y("brier:Q", scale=alt.Scale(domain=yd, reverse=True), axis=alt.Axis(format=".3f", tickCount=3,
              title="Brier (lower is better) →"))
    c = alt.condition("datum.ours", alt.value(BLUE), alt.value(GREY))
    pts = alt.Chart(f).mark_circle(opacity=0.9).encode(
        x=x, y=y, color=c, size=alt.condition("datum.ours", alt.value(70), alt.value(34)))
    lab = lab.assign(pos=[OFFSET.get((m, n), (9, 9, "left") if o else (7, 0, "left")) for n, o in zip(lab.name, lab.ours)])
    txts = [alt.Chart(g).mark_text(
        fontSize=FS - 1, baseline="middle", align=al, dx=dx, dy=dy,
        color=BLUE if o else INK, fontWeight="bold" if o else "normal"
    ).encode(x="roc_auc:Q", y="brier:Q", text="name:N")
        for (dx, dy, al, o), g in lab.assign(k=lab.pos.map(lambda t: t + (False,)), o=lab.ours).groupby(
            lab.pos.combine(lab.ours, lambda t, o: t + (bool(o),)))]
    return alt.layer(pts, *txts).properties(
        width=337, height=100, title=alt.TitleParams(m, anchor="start", fontSize=FS + 1, fontWeight="bold"))


mechs = ["Additive", "Interaction", "Partial"]
fig = (alt.vconcat(*[panel(m, m == mechs[-1]) for m in mechs], spacing=8)
       .resolve_scale(x="independent", y="independent")
       .configure(font=FONT, background="white")
       .configure_axis(labelFontSize=FS, titleFontSize=FS, labelColor=INK, titleColor=INK, titleFontWeight="normal",
                       gridColor=LIGHT, gridWidth=0.5, domainColor="#999999", tickColor="#999999")
       .configure_view(stroke=None))
spec = fig.to_dict()
(OUT / "fig4_auc_brier.svg").write_text(vlc.vegalite_to_svg(spec))
(OUT / "fig4_auc_brier.png").write_bytes(vlc.vegalite_to_png(spec, scale=DPI / 72))
print("wrote fig4_auc_brier.{svg,png}")
