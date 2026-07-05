"""Render an explanation as a publication-quality figure (Figure 2 of the paper).

Run:
  python -m xtraffic.evaluation.visualize_explanation \
      --explanation evaluation/results/explanations/rush_hour_am.json \
      --dataset metr_la

Draws the sensor graph in geographic (lon, lat) coordinates:
  * every sensor as a faint grey dot (the road network context),
  * the explanation's top influencing sensors coloured by importance,
  * the target sensor as a star,
  * the propagation path as directed arrows target<-...<-source,
  * a labelled colorbar; colorblind-safe 'viridis'; vector PDF output.

Design choices called out for the paper:
  * viridis is perceptually uniform and colorblind-safe.
  * geographic coordinates (not a spring layout) so the figure is a real MAP a
    planner can read against the city.
Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")               # headless: no display needed, just write files
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from ..models.gnn.loaders import load_node_meta
from ..utils.io_utils import PKG_ROOT


def visualize(explanation_path: str, dataset: str, out_path: str = None) -> str:
    if not os.path.isabs(explanation_path):
        explanation_path = os.path.join(PKG_ROOT, explanation_path)
    with open(explanation_path) as f:
        exp = json.load(f)
    meta = load_node_meta(dataset)
    latlon = np.array(meta["latlon"])              # [N, 2] = (lat, lon)
    lon, lat = latlon[:, 1], latlon[:, 0]

    fig, ax = plt.subplots(figsize=(8, 7))
    # (1) all sensors as context.
    ax.scatter(lon, lat, s=8, c="0.82", zorder=1, label="other sensors")

    # (2) top nodes coloured by importance.
    top = exp["top_nodes"]
    if top:
        imps = np.array([n["importance"] for n in top])
        norm = Normalize(vmin=float(imps.min()), vmax=float(imps.max()) or 1.0)
        cmap = plt.get_cmap("viridis")
        for n in top:
            i = n["node_id"]
            ax.scatter(lon[i], lat[i], s=140, c=[cmap(norm(n["importance"]))],
                       edgecolors="black", linewidths=0.5, zorder=3)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("explanation importance", rotation=270, labelpad=15)

    # (3) propagation path as directed arrows.
    path = exp["propagation_path"]
    for a, b in zip(path[:-1], path[1:]):
        ax.annotate("", xy=(lon[b], lat[b]), xytext=(lon[a], lat[a]),
                    arrowprops=dict(arrowstyle="->", color="crimson", lw=1.8),
                    zorder=2)

    # (4) target sensor as a star.
    tgt = exp["prediction"]["node_id"]
    ax.scatter(lon[tgt], lat[tgt], s=380, marker="*", c="gold",
               edgecolors="black", linewidths=1.0, zorder=4, label="target sensor")

    p = exp["prediction"]
    ax.set_title(f"{exp['meta'].get('scenario', '')}: {p['node_name']}\n"
                 f"{p['current_speed_mph']} -> {p['predicted_speed_mph']} mph "
                 f"at {p['horizon_minutes']} min "
                 f"(conf {exp['explanation_confidence']}, "
                 f"lag {exp['propagation_lag_minutes']} min)", fontsize=10)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    if out_path is None:
        base = os.path.splitext(os.path.basename(explanation_path))[0]
        out_dir = os.path.join(PKG_ROOT, "evaluation", "results", "figures")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"explanation_{base}.pdf")
    fig.savefig(out_path)                           # vector PDF
    plt.close(fig)
    print(f"[viz] -> {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--explanation", required=True)
    ap.add_argument("--dataset", default="metr_la")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    visualize(args.explanation, args.dataset, args.out)


if __name__ == "__main__":
    main()
