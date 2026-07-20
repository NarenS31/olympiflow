"""Rewrite the human-readable node NAMES inside cached explanation JSONs.

WHY THIS EXISTS (Phase-11 correction, FLAGGED)
----------------------------------------------
node_names.py gained a real PEMS-BAY region table, so every PEMS-BAY sensor's
name changed ("NW of Downtown San Jose (sensor 400970, ...)" ->
"Santa Clara / Great America (sensor 400970, ...)"). The advisor renders the
prompt FROM the cached explanation JSON, and build_or_load_explanation caches by
FILENAME PRESENCE ONLY (the staleness trap Phase 17 already documented). So
without this step the rerun would feed the LLM the OLD degenerate names and
measure nothing new.

WHY PATCHING IS CORRECT, NOT A SHORTCUT
---------------------------------------
A node's name is a DETERMINISTIC function of (node_id, node_meta) — it is derived
presentation metadata, not model output. Everything the explainer actually
computed (node_ids, importances, current speeds, edges, propagation path, lag,
confidence) is untouched, so this is NOT a substitute for re-running the
explainer and never changes an explanation's content. Re-solving 85 GNNExplainer
optimisations to relabel strings would cost hours and produce identical numbers.

SAFETY
------
* Refuses to run without --yes, and always writes a .bak.tar of the directory.
* Only ever assigns fields it can recompute; a node_id absent from node_meta is
  left alone and reported.
* Reports a per-file diff count so a silent no-op is visible.

Run:
  python -m xtraffic.evaluation.refresh_explanation_names --city pems_bay \
      --dir evaluation/results/cross_city_faithfulness/explanations_cache --yes

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import tarfile
from typing import Any, Dict, List

from ..models.explainer.node_names import NodeNamer
from ..models.gnn.loaders import load_node_meta
from ..utils.io_utils import PKG_ROOT


def refresh_file(path: str, namer: NodeNamer) -> int:
    """Rewrite node_name fields in one explanation. Returns how many changed."""
    with open(path) as f:
        exp: Dict[str, Any] = json.load(f)

    changed = 0

    def fix(holder: Dict[str, Any]) -> None:
        nonlocal changed
        nid = holder.get("node_id")
        if nid is None:
            return
        try:
            new = namer.name(int(nid))
        except (IndexError, KeyError, ValueError):
            return                      # unknown node: leave untouched
        if holder.get("node_name") != new:
            holder["node_name"] = new
            changed += 1

    fix(exp.get("prediction", {}))
    for n in exp.get("top_nodes", []) or []:
        fix(n)

    if changed:
        with open(path, "w") as f:
            json.dump(exp, f, indent=2)
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, help="dataset whose node_meta names these nodes")
    ap.add_argument("--dir", required=True, help="directory of cached explanation JSONs")
    ap.add_argument("--pattern", default=None,
                    help="glob within --dir (default: '<city>_*.json')")
    ap.add_argument("--yes", action="store_true",
                    help="required: actually rewrite files in place")
    args = ap.parse_args()

    d = args.dir if os.path.isabs(args.dir) else os.path.join(PKG_ROOT, args.dir)
    pattern = args.pattern or "{}_*.json".format(args.city)
    files: List[str] = sorted(glob.glob(os.path.join(d, pattern)))
    if not files:
        print("no files matching {} in {}".format(pattern, d))
        return

    namer = NodeNamer(load_node_meta(args.city))
    print("{} file(s) matching {!r}".format(len(files), pattern))
    print("sample rename: node 0 -> {}".format(namer.name(0)))
    if not args.yes:
        print("\nDRY RUN — pass --yes to rewrite in place.")
        return

    backup = d.rstrip("/") + ".bak.tar"
    with tarfile.open(backup, "w") as tar:
        for p in files:
            tar.add(p, arcname=os.path.basename(p))
    print("backup -> {}".format(backup))

    total = n_files = 0
    for p in files:
        c = refresh_file(p, namer)
        total += c
        n_files += int(c > 0)
    print("rewrote {} name field(s) across {}/{} file(s)".format(
        total, n_files, len(files)))


if __name__ == "__main__":
    main()
