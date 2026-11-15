"""Fig 2: eligibility-aware outcome states (conceptual; no data)."""
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from style import BLUE, GREY, INK, FONT, FS, WIDTH_IN, DPI

OUT = Path(__file__).resolve().parents[2] / "docs/paper/figures"
mpl.rcParams.update({"font.family": FONT, "font.size": FS, "hatch.linewidth": 0.7, "mathtext.fontset": "custom",
                     "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:bold",
                     "mathtext.cal": FONT})
fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.35))
ax.set_xlim(0, 6.8); ax.set_ylim(-0.05, 2.6); ax.axis("off")


def box(cx, cy, w, h, text, fc="white", ec=INK, tc=INK, ls="-", hatch=None, hc=None, bold=False):
    xy = (cx - w / 2, cy - h / 2)
    if hatch:  # light hatch drawn separately so it never fights with the text
        ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0,rounding_size=0.06", fc="white", ec=hc, lw=0,
                                    hatch=hatch))
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0,rounding_size=0.06", fc="none" if hatch else fc,
                                ec=ec, lw=1.1, ls=ls))
    ax.text(cx, cy, text, ha="center", va="center", color=tc, fontweight="bold" if bold else "normal", linespacing=1.3,
            bbox=dict(fc="white", ec="none", pad=1.5) if hatch else None)


def path(pts, head=True):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] if head else xs, ys[:-1] if head else ys, color=INK, lw=1, solid_capstyle="butt")
    if head:
        ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=INK, lw=1,
                                                                    shrinkA=0, shrinkB=0, mutation_scale=7))


def tag(x, y, s, ha="center"):
    ax.text(x, y, s, ha=ha, va="center", fontweight="bold", color=INK)


box(0.55, 1.5, 0.9, 0.5, "Encounter\n$i$, task $k$")
box(1.95, 1.5, 1.2, 0.5, "Task eligible?\n$E_{ik}$")
box(3.0, 0.8, 1.4, 0.5, "Label observed?\n$R_{ik}$")
box(4.85, 2.2, 1.5, 0.5, "Not applicable\n(NA)", ec=GREY, tc=GREY, hatch="////", hc="#d0d0d0")
box(4.85, 1.3, 1.5, 0.5, "Eligible,\nunobserved", ec=BLUE, tc=BLUE, ls=(0, (3, 2)))
box(4.85, 0.3, 1.5, 0.5, "Outcome 0 / 1", fc=BLUE, ec=BLUE, tc="white", bold=True)

path([(1.0, 1.5), (1.35, 1.5)])
path([(1.95, 1.75), (1.95, 2.2), (4.1, 2.2)]); tag(3.0, 2.32, "No")
path([(1.95, 1.25), (1.95, 0.8), (2.3, 0.8)]); tag(1.86, 1.02, "Yes", "right")
path([(3.0, 1.05), (3.0, 1.3), (4.1, 1.3)]); tag(2.91, 1.17, "No", "right")
path([(3.0, 0.55), (3.0, 0.3), (4.1, 0.3)]); tag(2.91, 0.43, "Yes", "right")

# brackets: scoring = all eligible (unobserved + outcome); training = observed outcomes only
bx = 5.75
path([(bx - 0.07, 0.03), (bx, 0.03), (bx, 1.57), (bx - 0.07, 1.57)], head=False)
ax.plot([bx - 0.07, bx, bx, bx - 0.07], [0.03, 0.03, 0.57, 0.57], color=BLUE, lw=2.4, solid_capstyle="butt")
ax.text(bx + 0.12, 1.05, "Scoring:\nall $E_{ik}=1$", ha="left", va="center", linespacing=1.3)
ax.text(bx + 0.12, 0.3, "Training:\n$M_{ik}=1$", ha="left", va="center", color=BLUE, linespacing=1.3)
fig.savefig(OUT / "fig2_eligibility.svg", bbox_inches="tight", pad_inches=0.05)
fig.savefig(OUT / "fig2_eligibility.png", dpi=DPI, bbox_inches="tight", pad_inches=0.05)
print("wrote fig2_eligibility.{svg,png}")
