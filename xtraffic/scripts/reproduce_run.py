"""Phase 1 — re-execute a past run and DIFF it against the original.

    python -m xtraffic.scripts.reproduce_run --run-id <id> [--execute]
    python -m xtraffic.scripts.reproduce_run --list

WHY A DIFF, NOT A PASS/FAIL
---------------------------
A reproduction script that prints "PASS" teaches you nothing when the answer is
"almost". This one reports, field by field, what changed:

  environment   git commit, dirty flag, Python, package versions
  inputs        checkpoint / data / config / prompt-template hashes
  llm           model digest, sampling options
  outputs       per-file sha256

Anything that differs is named, with the reason it matters. That is the useful
artefact: "reproduces except the checkpoint hash changed" is a diagnosis;
"FAIL" is not.

WHAT THIS CAN AND CANNOT VERIFY
-------------------------------
It can verify that inputs are byte-identical and that deterministic outputs
match. It CANNOT promise identical LLM text: Ollama is a separate process and
seeded reproducibility holds only for the same server version, model digest, and
loaded context. Where LLM output differs, the report says so and quantifies it
rather than declaring failure — see reproducibility/llm_log.verify_determinism.

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from ..reproducibility import llm_log, provenance, run_dir

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _find_run(run_id: str) -> str:
    base = os.path.join(REPO_ROOT, "xtraffic", run_dir.RAW)
    exact = os.path.join(base, run_id)
    if os.path.isdir(exact):
        return exact
    if os.path.isdir(base):
        hits = [n for n in os.listdir(base) if n.startswith(run_id)]
        if len(hits) == 1:
            return os.path.join(base, hits[0])
        if len(hits) > 1:
            raise SystemExit("ambiguous run id {!r}; matches:\n  {}".format(
                run_id, "\n  ".join(sorted(hits))))
    raise SystemExit("no run found for {!r} under {}".format(run_id, base))


# ---------------------------------------------------------------------------
# Diffing
# ---------------------------------------------------------------------------
class Diff:
    """One recorded difference, with why it matters."""

    def __init__(self, field: str, before: Any, after: Any, severity: str,
                 why: str):
        self.field = field
        self.before = before
        self.after = after
        self.severity = severity        # CRITICAL | WARNING | INFO
        self.why = why

    def as_dict(self) -> Dict[str, Any]:
        return {"field": self.field, "before": self.before, "after": self.after,
                "severity": self.severity, "why": self.why}

    def __str__(self) -> str:
        return "[{}] {}\n    before: {}\n    after : {}\n    why   : {}".format(
            self.severity, self.field, self.before, self.after, self.why)


def _cmp(diffs: List[Diff], field: str, before: Any, after: Any,
         severity: str, why: str) -> None:
    """Record a difference, distinguishing CHANGED from NEVER RECORDED.

    A field that is absent from the ORIGINAL manifest but present now has not
    "changed" — the original run simply did not capture it. Reporting that as
    CRITICAL is a false alarm, and a diff tool that cries wolf gets ignored,
    which defeats its purpose. The asymmetry is deliberate: absent-then-present
    is INFO (the original predates the field, or captured a narrower artifact
    set), while present-then-absent stays at the caller's severity, because
    losing provenance that once existed is a genuine regression.
    """
    if before == after:
        return
    if before is None and after is not None:
        diffs.append(Diff(
            field, None, after, "INFO",
            "not recorded by the original run (it captured a narrower "
            "provenance set), so there is nothing to compare against — this is "
            "a coverage gap, not a detected change"))
        return
    diffs.append(Diff(field, before, after, severity, why))


def diff_environment(old: Dict[str, Any], new: Dict[str, Any]) -> List[Diff]:
    diffs: List[Diff] = []
    og, ng = old.get("git", {}), new.get("git", {})
    _cmp(diffs, "git.commit", og.get("commit"), ng.get("commit"), "WARNING",
         "code changed between runs; output differences may be code, not noise")
    _cmp(diffs, "git.dirty", og.get("dirty"), ng.get("dirty"), "WARNING",
         "a dirty tree is not reproducible from its commit alone")

    _cmp(diffs, "python", old.get("interpreter", {}).get("python_version"),
         new.get("interpreter", {}).get("python_version"), "WARNING",
         "interpreter version affects dict/set iteration and float formatting")

    _cmp(diffs, "platform.system", old.get("platform", {}).get("system"),
         new.get("platform", {}).get("system"), "WARNING",
         "different OS/device changes float reduction order; bitwise identity "
         "is not achievable across CPU/MPS/CUDA")

    op = {p["package"]: p for p in old.get("packages", {}).get("mismatches", [])}
    npk = {p["package"]: p for p in new.get("packages", {}).get("mismatches", [])}
    for pkg in sorted(set(op) | set(npk)):
        _cmp(diffs, "package.{}".format(pkg),
             (op.get(pkg) or {}).get("installed"),
             (npk.get(pkg) or {}).get("installed"), "WARNING",
             "package version differs from the original run")

    # The resolver backend is the one that silently rewrites the metric.
    _cmp(diffs, "metric.fuzzy_backend",
         old.get("metric", {}).get("fuzzy_backend"),
         new.get("metric", {}).get("fuzzy_backend"), "CRITICAL",
         "the entity resolver's similarity FUNCTION changed while its threshold "
         "(82) stayed fixed. rapidfuzz.token_set_ratio and "
         "difflib.SequenceMatcher are different functions, so the same citation "
         "can resolve under one and not the other. Every faithfulness number "
         "depends on this.")
    _cmp(diffs, "metric.fuzzy_threshold",
         old.get("metric", {}).get("fuzzy_threshold"),
         new.get("metric", {}).get("fuzzy_threshold"), "CRITICAL",
         "resolver acceptance threshold changed")
    return diffs


def diff_inputs(old: Dict[str, Any], new: Dict[str, Any]) -> List[Diff]:
    diffs: List[Diff] = []
    oa, na = old.get("artifacts", {}) or {}, new.get("artifacts", {}) or {}
    for label in sorted(set(oa) | set(na)):
        o, n = oa.get(label, {}), na.get(label, {})
        _cmp(diffs, "artifact.{}.sha256".format(label),
             o.get("sha256"), n.get("sha256"), "CRITICAL",
             "input file CONTENT changed. If this is the checkpoint, the two "
             "runs used different models under the same filename — the exact "
             "failure that made epoch-34 and epoch-54 results indistinguishable "
             "(audit §5.5).")
        _cmp(diffs, "artifact.{}.exists".format(label),
             o.get("exists"), n.get("exists"), "CRITICAL",
             "input file appeared or disappeared")

    _cmp(diffs, "config_sha256", old.get("config_sha256"),
         new.get("config_sha256"), "CRITICAL",
         "the resolved configuration differs; this is not the same experiment")

    ot, nt = old.get("prompt_templates", {}) or {}, new.get("prompt_templates", {}) or {}
    for name in sorted(set(ot) | set(nt)):
        _cmp(diffs, "prompt_template.{}".format(name),
             (ot.get(name) or {}).get("sha256"), (nt.get(name) or {}).get("sha256"),
             "CRITICAL", "a prompt template changed; conditions are not comparable")
    return diffs


def diff_llm(old: Dict[str, Any], new: Dict[str, Any]) -> List[Diff]:
    diffs: List[Diff] = []
    ol, nl = old.get("llm", {}) or {}, new.get("llm", {}) or {}
    _cmp(diffs, "llm.model", ol.get("requested_model"), nl.get("requested_model"),
         "CRITICAL", "different model")
    _cmp(diffs, "llm.model_digest", ol.get("model_digest"), nl.get("model_digest"),
         "CRITICAL",
         "the model TAG resolved to different WEIGHTS. A tag is a mutable "
         "pointer; re-pulling can change what it points at.")
    _cmp(diffs, "llm.server_version", ol.get("server_version"),
         nl.get("server_version"), "WARNING",
         "Ollama server version affects seeded sampling reproducibility")
    return diffs


def diff_outputs(old_run: run_dir.RunDir, new_run: Optional[run_dir.RunDir]
                 ) -> Tuple[List[Diff], Dict[str, Any]]:
    """Compare per-file hashes. Returns (diffs, stats)."""
    diffs: List[Diff] = []
    old_files = {f["file"]: f for f in old_run.read_manifest().get("outputs", [])}
    if new_run is None:
        return diffs, {"compared": False,
                       "note": "no re-execution requested (--execute omitted)"}

    new_files = {f["file"]: f for f in new_run.list_outputs()}
    identical = 0
    for name in sorted(set(old_files) | set(new_files)):
        o, n = old_files.get(name), new_files.get(name)
        if o is None:
            diffs.append(Diff("output.{}".format(name), None, "present", "INFO",
                              "new run produced a file the original did not"))
        elif n is None:
            diffs.append(Diff("output.{}".format(name), "present", None, "WARNING",
                              "original produced a file the new run did not"))
        elif o.get("sha256") != n.get("sha256"):
            sev = "INFO" if name.endswith(".jsonl") and "llm" in name else "WARNING"
            diffs.append(Diff(
                "output.{}".format(name), o.get("sha256", "")[:16],
                n.get("sha256", "")[:16], sev,
                "output bytes differ" + (
                    " — expected for an LLM call log, which contains timings "
                    "and timestamps even when the text matches" if sev == "INFO"
                    else "")))
        else:
            identical += 1
    return diffs, {"compared": True, "n_identical": identical,
                   "n_old": len(old_files), "n_new": len(new_files)}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_report(old_run: run_dir.RunDir, new_prov: Dict[str, Any],
                 new_run: Optional[run_dir.RunDir]) -> Dict[str, Any]:
    old_man = old_run.read_manifest()
    old_prov = old_man.get("provenance", {})

    diffs = (diff_environment(old_prov, new_prov)
             + diff_inputs(old_prov, new_prov)
             + diff_llm(old_prov, new_prov))
    out_diffs, out_stats = diff_outputs(old_run, new_run)
    diffs += out_diffs

    by_sev: Dict[str, int] = {"CRITICAL": 0, "WARNING": 0, "INFO": 0}
    for d in diffs:
        by_sev[d.severity] = by_sev.get(d.severity, 0) + 1

    if by_sev["CRITICAL"]:
        verdict = "NOT_REPRODUCIBLE"
        expl = ("At least one input, config, prompt, or model differs. The two "
                "runs are not the same experiment.")
    elif by_sev["WARNING"]:
        verdict = "REPRODUCIBLE_WITH_CAVEATS"
        expl = ("Inputs match. Environment or outputs differ in ways that are "
                "expected to perturb results without invalidating them. Read "
                "the warnings before quoting either run.")
    else:
        verdict = "REPRODUCIBLE"
        expl = "Environment, inputs, and outputs all match."

    return {
        "original_run_id": old_man.get("run_id"),
        "original_status": old_man.get("status"),
        "original_experiment": old_man.get("experiment"),
        "original_mode": (old_man.get("config", {}).get("experiment", {})
                          .get("mode")),
        "new_run_id": new_run.run_id if new_run else None,
        "verdict": verdict,
        "verdict_explanation": expl,
        "counts": by_sev,
        "outputs": out_stats,
        "diffs": [d.as_dict() for d in diffs],
    }


def print_report(rep: Dict[str, Any]) -> None:
    print("=" * 72)
    print("REPRODUCTION REPORT")
    print("=" * 72)
    print("  original : {} [{}] ({})".format(
        rep["original_run_id"], rep.get("original_status"),
        rep.get("original_mode")))
    if rep.get("new_run_id"):
        print("  new      : {}".format(rep["new_run_id"]))
    print("  verdict  : {}".format(rep["verdict"]))
    print("             {}".format(rep["verdict_explanation"]))
    c = rep["counts"]
    print("  diffs    : {} critical, {} warning, {} info".format(
        c.get("CRITICAL", 0), c.get("WARNING", 0), c.get("INFO", 0)))
    if rep["diffs"]:
        print("\nDIFFERENCES")
        print("-" * 72)
        order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
        for d in sorted(rep["diffs"], key=lambda x: order.get(x["severity"], 9)):
            print(str(Diff(d["field"], d["before"], d["after"], d["severity"],
                           d["why"])))
            print()
    else:
        print("\n  No differences detected.")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Re-execute a past run and diff it against the original.")
    ap.add_argument("--run-id", help="run id (or unique prefix)")
    ap.add_argument("--list", action="store_true", help="list runs and exit")
    ap.add_argument("--experiment", help="filter --list by experiment name")
    ap.add_argument("--execute", action="store_true",
                    help="actually re-run the study; without it, compare "
                         "environment and inputs only (cheap and safe)")
    ap.add_argument("--out", help="write the report JSON here")
    args = ap.parse_args(argv)

    if args.list:
        rows = run_dir.list_runs(REPO_ROOT, args.experiment)
        if not rows:
            print("no runs found under xtraffic/{}".format(run_dir.RAW))
            return 0
        print("{:<58} {:<12} {:<10} {}".format(
            "RUN_ID", "STATUS", "GIT", "EXPERIMENT"))
        for r in rows:
            if "error" in r:
                print("{:<58} {}".format(r["run_id"], r["error"]))
                continue
            print("{:<58} {:<12} {:<10} {}{}".format(
                r["run_id"][:57], r.get("status", "?"), r.get("git", "?") or "?",
                r.get("experiment", "?"),
                "  [SUPERSEDED]" if r.get("superseded_by") else ""))
        return 0

    if not args.run_id:
        ap.error("--run-id is required (or use --list)")

    old_run = run_dir.RunDir.open(_find_run(args.run_id), REPO_ROOT)
    old_man = old_run.read_manifest()
    cfg = old_man.get("config", {})

    # Capture the CURRENT environment under the ORIGINAL config, so the
    # comparison isolates what moved.
    from .run_experiment import artifacts_for, prompt_templates
    llm_cfg = cfg.get("llm") or {}
    new_prov = provenance.capture(
        repo_root=REPO_ROOT, config=cfg, artifacts=artifacts_for(cfg),
        ollama_host=llm_cfg.get("host"), ollama_model=llm_cfg.get("model"),
        prompt_templates=prompt_templates())

    new_run: Optional[run_dir.RunDir] = None
    if args.execute:
        from .run_experiment import run as run_experiment
        print("Re-executing {} ...\n".format(old_man.get("experiment")))
        new_run = run_experiment(cfg, old_man.get("experiment", "reproduce"),
                                 study_fn=None, dry_run=True)

    rep = build_report(old_run, new_prov, new_run)
    print_report(rep)

    out = args.out or os.path.join(old_run.path, "reproduction_report.json")
    # The original run is append-only, so a repeat check writes a numbered file
    # rather than overwriting the earlier verdict.
    if os.path.exists(out):
        i = 2
        stem, ext = os.path.splitext(out)
        while os.path.exists("{}_{}{}".format(stem, i, ext)):
            i += 1
        out = "{}_{}{}".format(stem, i, ext)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, default=str)
    print("report -> {}".format(out))

    return 0 if rep["verdict"] != "NOT_REPRODUCIBLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
