"""Fig 1: M2G-NET architecture, left to right. Masks are shown in Fig 2, not here."""
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from style import BLUE, GREY, INK, FONT, FS, WIDTH_IN, DPI

OUT = Path(__file__).resolve().parents[2] / "docs/paper/figures"
mpl.rcParams.update({"font.family": FONT, "font.size": FS, "mathtext.fontset": "custom", "mathtext.rm": FONT,
                     "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:bold"})
fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.75))
ax.set_xlim(0, 13.2); ax.set_ylim(0.35, 5.4); ax.axis("off")
LIGHT_BLUE = "#e8f1f8"


def box(cx, cy, w, h, text, fc="white", ec=INK, tc=INK, bold=False):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(cx, cy, text, ha="center", va="center", linespacing=1.35, color=tc, fontweight="bold" if bold else "normal")


def line(pts, head=True):
    xs, ys = zip(*pts)
    n = -1 if head else None
    ax.plot(xs[:n], ys[:n], color=INK, lw=1, solid_capstyle="butt")
    if head:
        ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=INK, lw=1, shrinkA=0,
                                                                    shrinkB=0, mutation_scale=7))


views = ["Role", "Traits", "Road", "Environment", "Site"]
ys = [4.9, 4.05, 3.2, 2.35, 1.5]
for v, y in zip(views, ys):
    box(1.0, y, 1.9, 0.62, v, fc="#f4f4f4", ec=GREY)
ax.text(1.0, 0.75, "View encoders\n$h_v\\in\\mathbb{R}^{16}$", ha="center", va="center", color=GREY, linespacing=1.35)
bx = 2.4
line([(bx, ys[0]), (bx, ys[-1])], head=False)
for y in ys:
    line([(1.95, y), (bx, y)], head=False)

IY, EY, GY = 4.6, 1.5, 3.2
box(4.1, IY, 2.9, 1.0, "Interaction\nindividual \u00d7 context", fc=LIGHT_BLUE, ec=BLUE)
box(4.1, EY, 2.9, 1.0, "Shared early\nfusion", fc=LIGHT_BLUE, ec=BLUE)
line([(bx, IY), (2.65, IY)])
line([(bx, EY), (2.65, EY)])

box(7.1, GY, 2.1, 1.7, "Task-specific\ngates (\u00d74)\nsparsemax", fc=LIGHT_BLUE, ec=BLUE)
line([(bx, GY), (6.05, GY)])
ax.text(4.1, GY + 0.1, "five views", ha="center", va="bottom")
line([(5.55, IY), (5.8, IY), (5.8, 3.7), (6.05, 3.7)])

box(9.6, GY, 2.3, 1.7, "Residual blend\n$\\beta\\,z^{gated}_k$\n$+(1-\\beta)\\,z^{early}$", fc=LIGHT_BLUE, ec=BLUE)
line([(8.15, GY), (8.45, GY)])
line([(5.55, EY), (9.6, EY), (9.6, 2.35)])
ax.text(7.5, EY + 0.1, "$z^{early}$", ha="center", va="bottom")

box(11.95, GY, 2.0, 1.7, "Four heads\nsigmoid\n$\\hat p_k$", fc=BLUE, ec=BLUE, tc="white", bold=True)
line([(10.75, GY), (10.95, GY)])
fig.savefig(OUT / "fig1_architecture.svg", bbox_inches="tight", pad_inches=0.05)
fig.savefig(OUT / "fig1_architecture.png", dpi=DPI, bbox_inches="tight", pad_inches=0.05)
print("wrote fig1_architecture.{svg,png}")
