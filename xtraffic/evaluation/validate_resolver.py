"""Validate the entity resolver against the hand-labeled set (Phase-5 gate).

WHY this exists: the faithfulness metric is only as trustworthy as the resolver
underneath it. If the resolver mis-maps the LLM's cited locations to node_ids,
every precision/recall number is garbage — and a reviewer WILL check this. So we
hold out 30 human-labeled cases (evaluation/resolver_labels.json) and report the
resolver's accuracy against them. Gate: >= 90%.

A case passes when the resolver's node-id set equals the set the label implies:
  * region label   -> table.region_to_nodes[label]
  * "sensor:<id>"  -> {that sensor's node id}
  * null           -> empty set (correctly refuses to resolve)

Run:
  python -m xtraffic.evaluation.validate_resolver

Python 3.9 compatible.
"""
from __future__ import annotations

import json
import os
from typing import Set

from ..models.gnn.loaders import load_node_meta
from ..utils.io_utils import PKG_ROOT
from .faithfulness import FUZZ_BACKEND, NodeTable, resolve_location


def _expected_set(expected, table: NodeTable) -> Set[int]:
    """Turn a label's `expected` field into the node-id set it implies."""
    if expected is None:
        return set()
    if isinstance(expected, str) and expected.startswith("sensor:"):
        sid = expected.split(":", 1)[1]
        return {table.sid_to_node[sid]} if sid in table.sid_to_node else set()
    # otherwise it is a region label
    return set(table.region_to_nodes.get(expected, set()))


def main() -> None:
    labels_path = os.path.join(PKG_ROOT, "evaluation", "resolver_labels.json")
    with open(labels_path) as f:
        spec = json.load(f)

    table = NodeTable(load_node_meta(spec["city"]))
    cases = spec["cases"]

    n_pass = 0
    print("Fuzzy backend: {}\n".format(FUZZ_BACKEND))
    print("{:38s} {:14s} {:6s} {}".format("LOCATION", "METHOD", "OK", "NOTE"))
    print("-" * 78)
    for case in cases:
        res = resolve_location(case["location"], table)
        want = _expected_set(case["expected"], table)
        ok = res.node_ids == want
        n_pass += int(ok)
        note = ""
        if not ok:
            note = "got {} nodes, wanted {}".format(len(res.node_ids), len(want))
        print("{:38s} {:14s} {:6s} {}".format(
            case["location"][:38], res.method, "PASS" if ok else "FAIL", note))

    acc = n_pass / len(cases)
    print("-" * 78)
    print("Resolver accuracy: {}/{} = {:.1%}".format(n_pass, len(cases), acc))
    print("GATE (>= 90%): {}".format("PASS" if acc >= 0.90 else "FAIL"))

    # Persist for the paper's methods section (CLAUDE.md: results -> JSON+CSV).
    out = os.path.join(PKG_ROOT, "evaluation", "results", "resolver_accuracy.json")
    with open(out, "w") as f:
        json.dump({"n_cases": len(cases), "n_pass": n_pass, "accuracy": acc,
                   "threshold": 0.90, "passed": acc >= 0.90,
                   "fuzz_backend": FUZZ_BACKEND}, f, indent=2)
    print("Wrote {}".format(out))


if __name__ == "__main__":
    main()
