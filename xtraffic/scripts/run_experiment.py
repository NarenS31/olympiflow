"""Phase 1 — the single entry point for a provenance-stamped experiment run.

    python -m xtraffic.scripts.run_experiment --config configs/base.yaml \
           --experiment smoke [--dry-run] [--check-determinism]

WHAT IT DOES
------------
  1. Load + overlay config (base.yaml <- experiment overlay <- CLI overrides).
  2. Seed everything (reproducibility/seeds.py) and record what could NOT be
     enforced.
  3. Capture provenance: git SHA + dirty flag, Python, package-vs-requirements
     mismatches, platform, checkpoint/data sha256, Ollama model DIGEST, the
     entity resolver's active fuzzy backend.
  4. Create an append-only run directory that REFUSES to overwrite.
  5. Optionally measure LLM reproducibility (never assume it).
  6. Hand the prepared run to the study callable.
  7. Write manifest + per-file hashes on completion — including on FAILURE,
     because a failed run is data.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not run any study on its own. Phase 1 builds the harness; the studies
land in later phases. `--dry-run` exercises every step except the study body,
which is how the Phase-1 gate is checked without burning LLM compute.

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional

import yaml

from ..reproducibility import llm_log, provenance, run_dir, seeds as seeds_mod

# Repo root = two levels up from this file (xtraffic/scripts/ -> xtraffic/ -> repo)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PKG_ROOT = os.path.join(REPO_ROOT, "xtraffic")


# ---------------------------------------------------------------------------
# Config loading + overlay
# ---------------------------------------------------------------------------
def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge. Overlay wins; nested dicts merge rather than
    replace, so an overlay can change `llm.sampling.seed` without having to
    restate the whole `llm` block."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve(path: str) -> str:
    """Config paths are written relative to the package root (configs/...,
    models/...), matching every committed config."""
    return path if os.path.isabs(path) else os.path.join(PKG_ROOT, path)


def load_config(config_path: str, overlays: Optional[List[str]] = None,
                overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    """base <- overlays (in order) <- `--set a.b.c=value` CLI overrides."""
    with open(_resolve(config_path), "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    for ov in (overlays or []):
        with open(_resolve(ov), "r", encoding="utf-8") as fh:
            cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})

    for item in (overrides or []):
        if "=" not in item:
            raise SystemExit("--set expects key.path=value, got {!r}".format(item))
        key, _, raw = item.partition("=")
        # Parse through YAML so numbers/bools/lists behave as written.
        try:
            val = yaml.safe_load(raw)
        except Exception:
            val = raw
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val

    return cfg


# ---------------------------------------------------------------------------
# The artifacts a run should hash
# ---------------------------------------------------------------------------
def artifacts_for(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Every input file whose bytes could change a result."""
    arts: Dict[str, str] = {}

    ckpt = (cfg.get("model") or {}).get("checkpoint")
    if ckpt:
        arts["checkpoint"] = _resolve(ckpt)

    ds = (cfg.get("dataset") or {}).get("name")
    split = (cfg.get("dataset") or {}).get("split", "test")
    if ds:
        base = os.path.join(PKG_ROOT, "data", "processed", ds)
        for label, fname in (("data_split", "{}.npz".format(split)),
                             ("adjacency", "adjacency.npy"),
                             ("node_meta", "node_meta.json"),
                             ("scaler", "scaler.json")):
            p = os.path.join(base, fname)
            if os.path.exists(p):
                arts["{}_{}".format(ds, label)] = p

    arts["config_advisor"] = os.path.join(PKG_ROOT, "configs", "advisor.yaml")
    return arts


def prompt_templates() -> Dict[str, str]:
    """Hash the prompt constants so a prompt edit is detectable in the manifest
    without diffing whole strings. These are the exact constants the traffic
    regression gate pins."""
    try:
        from ..models.advisor import advisor as A
        return {
            "SYSTEM": A._SYSTEM,
            "TASK": A._TASK,
            "TASK_NO_EXPLANATION": A._TASK_NO_EXPLANATION,
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def preflight(cfg: Dict[str, Any], prov: Dict[str, Any]) -> List[str]:
    """Checks that must pass before compute is spent. Returns fatal errors.

    Non-fatal problems are already in `prov["warnings"]` and are printed; these
    are the ones that make a run scientifically void, so they abort instead.
    """
    fatal: List[str] = []

    # Pinned checkpoint hash must match, if pinned.
    want = (cfg.get("model") or {}).get("expect_sha256")
    if want:
        got = (prov.get("artifacts", {}).get("checkpoint", {}) or {}).get("sha256")
        if got != want:
            fatal.append(
                "checkpoint sha256 mismatch:\n  expected {}\n  found    {}\n"
                "  The checkpoint at this path is NOT the one this config pins. "
                "This is the exact failure mode that made epoch-34 and epoch-54 "
                "results indistinguishable (audit §5.5).".format(want, got))

    # Pinned Ollama digest must match, if pinned.
    want_dig = (cfg.get("llm") or {}).get("expect_digest")
    if want_dig:
        got_dig = (prov.get("llm", {}) or {}).get("model_digest")
        if got_dig != want_dig:
            fatal.append(
                "Ollama model digest mismatch:\n  expected {}\n  found    {}"
                .format(want_dig, got_dig))

    # A confirmatory run has a higher bar: it must be reproducible from a commit.
    if (cfg.get("experiment") or {}).get("mode") == "confirmatory":
        if prov.get("git", {}).get("dirty"):
            fatal.append(
                "confirmatory run refused: git working tree is DIRTY, so the "
                "result would not be reproducible from its commit. Commit first.")
        if not prov.get("packages", {}).get("environment_matches_requirements"):
            fatal.append(
                "confirmatory run refused: environment does not match "
                "requirements.txt ({} mismatch(es)). Fix the environment or "
                "update the pins deliberately.".format(
                    prov.get("packages", {}).get("n_mismatches")))

    return fatal


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(cfg: Dict[str, Any], experiment: str,
        study_fn: Optional[Callable[[Dict[str, Any], run_dir.RunDir,
                                     Optional[llm_log.LLMLogger]], Any]] = None,
        dry_run: bool = False,
        check_determinism: bool = False) -> run_dir.RunDir:
    """Prepare, stamp, and execute one run. Returns the RunDir."""
    exp_cfg = cfg.get("experiment") or {}
    mode = exp_cfg.get("mode", "exploratory")

    # -- 1. seeds ----------------------------------------------------------
    master = int((cfg.get("seeds") or {}).get("master", 42))
    strict = bool((cfg.get("seeds") or {}).get("strict_determinism", False))
    seed_rec = seeds_mod.set_all_seeds(master, strict=strict)

    # -- 2. provenance -----------------------------------------------------
    llm_cfg = cfg.get("llm") or {}
    prov = provenance.capture(
        repo_root=REPO_ROOT,
        config=cfg,
        artifacts=artifacts_for(cfg),
        ollama_host=llm_cfg.get("host"),
        ollama_model=llm_cfg.get("model"),
        prompt_templates=prompt_templates(),
        extra={"seeds": seed_rec, "rng_state": seeds_mod.describe(),
               "argv": sys.argv})

    print("=" * 72)
    print("XTraffic run: {}   [{}]".format(experiment, mode.upper()))
    print("=" * 72)
    for line in provenance.summary_lines(prov):
        print("  " + line)

    fatal = preflight(cfg, prov)
    if fatal:
        print("\nPREFLIGHT FAILED:")
        for f in fatal:
            print("  - " + f)
        raise SystemExit(2)

    if mode == "exploratory":
        print("  note      : EXPLORATORY run. Results from this run may not be "
              "reported as confirmatory.")

    # -- 3. run directory --------------------------------------------------
    run = run_dir.RunDir.create(
        repo_root=REPO_ROOT, experiment=experiment, config=cfg,
        artifacts=artifacts_for(cfg),
        ollama_host=llm_cfg.get("host"), ollama_model=llm_cfg.get("model"),
        prompt_templates=prompt_templates(),
        notes=exp_cfg.get("notes") or None)
    print("\n  run_id    : {}".format(run.run_id))
    print("  path      : {}".format(run.path))

    logger: Optional[llm_log.LLMLogger] = None
    if (cfg.get("output") or {}).get("log_llm_calls", True):
        logger = llm_log.LLMLogger(run.file(run_dir.LLM_CALLS), run.run_id)

    run.start()

    try:
        # -- 4. determinism measurement ------------------------------------
        det_cfg = llm_cfg.get("determinism_check") or {}
        if check_determinism or det_cfg.get("enabled"):
            if (prov.get("llm", {}) or {}).get("model_present"):
                print("\n  measuring LLM reproducibility "
                      "(seeded, identical prompt)...")
                det = llm_log.verify_determinism(
                    host=llm_cfg["host"], model=llm_cfg["model"],
                    prompt=("Return ONLY this JSON object and nothing else: "
                            '{"ok": true, "n": 7}'),
                    n_draws=int(det_cfg.get("n_draws", 5)),
                    sampling=llm_cfg.get("sampling"), logger=logger,
                    timeout=float(llm_cfg.get("timeout_seconds", 180)))
                run.write_json("determinism_check.json", det)
                print("    reproducibility_rate = {:.3f}  "
                      "({} distinct output(s) over {} draw(s))".format(
                          det["reproducibility_rate"], det["distinct_outputs"],
                          det["n_draws"]))
                if not det["byte_identical"]:
                    print("    NOTE: seeded output is NOT byte-identical on this "
                          "host. This is a measured property, reported as such; "
                          "the analysis plan treats DRAW as a random factor.")
            else:
                print("\n  skipping determinism check: model not available")

        # -- 5. the study ---------------------------------------------------
        if dry_run or study_fn is None:
            print("\n  DRY RUN: harness exercised, no study executed.")
            run.write_json("dry_run.json",
                           {"dry_run": True,
                            "note": "harness verification only; no study body ran",
                            "seeds": seed_rec})
            result: Any = {"dry_run": True}
        else:
            result = study_fn(cfg, run, logger)
            if isinstance(result, dict):
                run.write_json("result.json", result)

        # -- 6. close out ---------------------------------------------------
        summary: Dict[str, Any] = {"mode": mode}
        if logger is not None:
            summary["llm"] = llm_log.summarise_log(logger.read())
        run.complete(**summary)
        print("\n  status    : COMPLETED")
        if logger is not None and summary.get("llm", {}).get("n_calls"):
            s = summary["llm"]
            print("  llm calls : {} ({} failed, {} parse failures, "
                  "{} truncated)".format(s["n_calls"], s["n_transport_failed"],
                                         s["n_parse_failures"], s["n_truncated"]))
        return run

    except BaseException as exc:
        # A crashed run keeps its partial outputs and records WHY it stopped.
        # Deleting failed runs is how a project loses track of what it tried.
        run.fail("{}: {}\n{}".format(type(exc).__name__, exc,
                                     traceback.format_exc()))
        print("\n  status    : FAILED ({}: {})".format(type(exc).__name__, exc))
        raise


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run a provenance-stamped XTraffic experiment.")
    ap.add_argument("--config", default="configs/base.yaml",
                    help="base config (relative to the xtraffic package root)")
    ap.add_argument("--overlay", action="append", default=[],
                    help="overlay config(s), applied in order")
    ap.add_argument("--set", action="append", default=[], dest="overrides",
                    metavar="KEY.PATH=VALUE", help="config override")
    ap.add_argument("--experiment", default=None,
                    help="experiment name (default: experiment.name from config)")
    ap.add_argument("--dry-run", action="store_true",
                    help="exercise the whole harness without running a study")
    ap.add_argument("--check-determinism", action="store_true",
                    help="measure seeded LLM reproducibility before the study")
    ap.add_argument("--print-config", action="store_true",
                    help="print the resolved config and exit")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overlay, args.overrides)
    if args.print_config:
        print(json.dumps(cfg, indent=2, default=str))
        return 0

    experiment = args.experiment or (cfg.get("experiment") or {}).get("name", "run")
    run(cfg, experiment, study_fn=None, dry_run=args.dry_run,
        check_determinism=args.check_determinism)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
