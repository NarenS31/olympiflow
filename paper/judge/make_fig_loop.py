"""Figure: faithfulness F1 by correction round, real evidence vs random evidence.

Data comes straight from the two run summaries -- no numbers are typed here:
  real   evaluation/results/active_grounding/active_grounding_summary.json
  random evaluation/results/raw/<grounding run>/part2_summary.json

Design notes (why it looks the way it does):
  * ONE axis. F1 only. The fraction-reaching-threshold lives in the prose; putting
    it on a second y-scale would be a dual-axis chart, which misleads about
    relative magnitude.
  * Two series, Wong blue #0072B2 and vermillion #D55E00 -- validated
    colorblind-safe (worst adjacent CVD dE 21.9 protan, 31.2 normal vision).
  * Colour is not the only channel: the series also differ in dash pattern and
    marker, so the figure survives greyscale printing.
  * Legend present (two series), axis/grid recessive, no embedded title -- the
    caption carries it, as NeurIPS expects.
  * Vector PDF, 8pt text to match the body font at this reduction.

    python3 paper/judge/make_fig_loop.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RID = "20260825T201433Z__grounding_without_information__f193e764__a1b0fc36"
REAL = os.path.join(REPO, "xtraffic", "evaluation", "results", "active_grounding",
                    "active_grounding_summary.json")
RAND = os.path.join(REPO, "xtraffic", "evaluation", "results", "raw", RID,
                    "part2_summary.json")

BLUE, VERM, INK, MUTED = "#0072B2", "#D55E00", "#222222", "#777777"


def main() -> None:
    real = json.load(open(REAL))
    rand = json.load(open(RAND))
    thr = real["f1_threshold"]

    rx = [r["round"] for r in real["per_round"]]
    ry = [r["mean_f1"] for r in real["per_round"]]
    nx = [r["round"] for r in rand["per_round"]]
    ny = [r["mean_f1"] for r in rand["per_round"]]

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                         "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(5.4, 2.05))

    ax.axhline(thr, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax.text(1.5, thr + 0.006, "threshold {:.2f}".format(thr), color=MUTED,
            va="bottom", ha="center", fontsize=7)

    ax.plot(rx, ry, color=BLUE, lw=1.8, marker="o", ms=4.5, ls="-",
            label="real evidence (A)", zorder=3)
    ax.plot(nx, ny, color=VERM, lw=1.8, marker="s", ms=4.2, ls="--",
            label="random evidence (A_rand)", zorder=3)

    # The point of the figure: the gap exists at round 0 and is gone after.
    ax.annotate("", xy=(0, ry[0]), xytext=(0, ny[0]),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7,
                                shrinkA=2.5, shrinkB=2.5))
    ax.text(0.13, (ry[0] + ny[0]) / 2, "round-0 gap", fontsize=7,
            color=INK, va="center", ha="left")

    ax.set_xlabel("correction round")
    ax.set_ylabel("faithfulness $F_1$")
    ax.set_xticks(rx)
    ax.set_xlim(-0.28, 3.25)
    ax.set_ylim(0.62, 0.95)
    ax.grid(axis="y", color="#e8e8e8", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, length=3, width=0.8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)
    ax.legend(frameon=False, loc="lower right", handlelength=2.4)

    fig.tight_layout(pad=0.3)
    out = os.path.join(HERE, "fig_loop.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote {}".format(out))
    print("  real   round0 {:.4f} -> round3 {:.4f}".format(ry[0], ry[-1]))
    print("  random round0 {:.4f} -> round3 {:.4f}".format(ny[0], ny[-1]))


if __name__ == "__main__":
    main()
