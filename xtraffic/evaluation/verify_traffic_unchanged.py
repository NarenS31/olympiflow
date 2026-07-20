"""Phase 19 REGRESSION GATE — prove the domain refactor changed NOTHING for traffic.

WHY THIS FILE EXISTS
--------------------
Phase 19 parameterised the advisor's vocabulary (models/advisor/domains.py) and
gave the faithfulness resolver a non-geographic ladder. Both touched code that
every committed number in Phases 4, 5, 11, 12, 13, 15b, 16, 17 and 18 was produced
by. CLAUDE.md's rule is "never silently change something we built earlier" — the
honest way to keep that promise is not a careful reading, it is an ASSERTION.

This script re-renders the traffic prompts and re-runs the traffic resolver, and
compares them byte-for-byte against `golden_traffic.json`, which was captured
BEFORE the refactor. If a single character of a METR-LA prompt moved, this fails
and the committed numbers are suspect until it is explained.

Run:
  python -m xtraffic.evaluation.verify_traffic_unchanged           # check
  python -m xtraffic.evaluation.verify_traffic_unchanged --capture # re-baseline

--capture is deliberately separate and loud: re-baselining is how you would hide
a regression, so it must be a conscious act, never a side effect of running the
check.

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List

from ..models.advisor import advisor as A
from ..models.gnn.loaders import load_node_meta
from ..utils.io_utils import PKG_ROOT
from .faithfulness import NodeTable, resolve_location

GOLDEN = os.path.join(PKG_ROOT, "evaluation", "golden_traffic.json")

# Locations chosen to exercise EVERY rung of the traffic ladder: exact region,
# fuzzy, gazetteer alias, explicit sensor id, and a deliberate miss.
_PROBE_LOCATIONS = [
    "Downtown LA", "glendale", "San Fernando Valley", "sensor 772167",
    "near downtown", "the Westside", "I-10 corridor", "Pasadena",
    "nowhere at all", "East Los Angeles", "South LA", "Long Beach",
]
_PROBE_CHICAGO = ["The Loop", "south loop", "Hyde Park", "nowhere"]


def _explanation_files(limit: int = 6) -> List[str]:
    """A stable sample of committed METR-LA explanations to render prompts from."""
    files: List[str] = []
    for pat in ("evaluation/results/faithfulness/explanations_cache/*.json",
                "evaluation/results/explanations/*.json"):
        files += sorted(glob.glob(os.path.join(PKG_ROOT, pat)))[:limit]
    return files


def snapshot() -> Dict[str, Any]:
    """Render every traffic artifact this refactor could possibly have moved."""
    out: Dict[str, Any] = {}
    for path in _explanation_files():
        try:
            with open(path) as f:
                exp = json.load(f)
        except (OSError, ValueError):
            continue
        if "top_nodes" not in exp or "prediction" not in exp:
            continue
        key = os.path.basename(path)
        out[key] = {
            "render": A.render_explanation_text(exp),
            "predonly": A.render_prediction_only_text(exp),
            "promptA": A.build_prompt_condition(exp, "KB BLOCK", "A"),
            "promptB": A.build_prompt_condition(exp, "KB BLOCK", "B"),
            "promptC": A.build_prompt_condition(exp, "KB BLOCK", "C"),
        }
    # The module-level constants other phases import directly.
    out["_constants"] = {"SYSTEM": A._SYSTEM, "TASK": A._TASK,
                         "TASK_NO_EXPLANATION": A._TASK_NO_EXPLANATION}
    # The resolver, on both traffic cities that have committed numbers.
    for city, probes in (("metr_la", _PROBE_LOCATIONS), ("chicago", _PROBE_CHICAGO)):
        try:
            table = NodeTable(load_node_meta(city))
        except (OSError, KeyError):
            continue                      # dataset not built locally; skip, don't fail
        res: Dict[str, Any] = {}
        for loc in probes:
            r = resolve_location(loc, table)
            res[loc] = {"method": r.method, "n": len(r.node_ids),
                        "ids": sorted(r.node_ids)[:8], "matched": r.matched}
        out["_resolver_" + city] = res
    return out


def _diff(golden: Dict[str, Any], now: Dict[str, Any]) -> List[str]:
    """Every key whose value moved, as human-readable lines."""
    problems: List[str] = []
    for key in sorted(set(golden) | set(now)):
        if key not in now:
            problems.append("MISSING now: {}".format(key)); continue
        if key not in golden:
            continue                      # new coverage is fine, not a regression
        g, n = golden[key], now[key]
        if g == n:
            continue
        if isinstance(g, dict) and isinstance(n, dict):
            for sub in sorted(set(g) | set(n)):
                if g.get(sub) != n.get(sub):
                    problems.append("CHANGED: {}::{}".format(key, sub))
                    gs, ns = str(g.get(sub)), str(n.get(sub))
                    for i, (a, b) in enumerate(zip(gs, ns)):
                        if a != b:
                            lo = max(0, i - 60)
                            problems.append("   golden: ...{!r}".format(gs[lo:i + 60]))
                            problems.append("   now   : ...{!r}".format(ns[lo:i + 60]))
                            break
                    else:
                        problems.append("   length {} -> {}".format(len(gs), len(ns)))
        else:
            problems.append("CHANGED: {}".format(key))
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true",
                    help="OVERWRITE the golden baseline. Only do this when you "
                         "have consciously decided a traffic-side change is correct.")
    args = ap.parse_args()

    now = snapshot()
    if args.capture:
        with open(GOLDEN, "w") as f:
            json.dump(now, f, indent=1, sort_keys=True)
        print("CAPTURED baseline -> {} ({} entries)".format(GOLDEN, len(now)))
        return

    if not os.path.exists(GOLDEN):
        print("No golden baseline at {}.\nRun with --capture first.".format(GOLDEN))
        sys.exit(2)
    with open(GOLDEN) as f:
        golden = json.load(f)

    problems = _diff(golden, now)
    n_prompts = sum(1 for k in golden if not k.startswith("_"))
    if problems:
        print("TRAFFIC REGRESSION GATE: FAIL")
        for p in problems:
            print("  " + p)
        sys.exit(1)
    print("TRAFFIC REGRESSION GATE: PASS")
    print("  {} explanation(s) x 5 rendered artifacts, the 3 prompt constants, "
          "and the resolver on {} city/cities are BYTE-IDENTICAL to pre-Phase-19."
          .format(n_prompts, sum(1 for k in golden if k.startswith("_resolver_"))))


if __name__ == "__main__":
    main()
