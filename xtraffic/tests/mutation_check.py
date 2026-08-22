"""Do the Phase-3 tests actually have teeth?

    python -m xtraffic.tests.mutation_check

A green test suite proves nothing on its own — a test can pass because the code
is right, or because the test never reaches the code. This script sabotages one
critical branch at a time and checks the suite NOTICES.

It earned its keep immediately. The first run reported that stubbing out
`_direction`'s reversal detection — the single capability the pre-Phase-3 metric
lacked, and the headline justification for this whole module — left all 33 tests
green. The reversal test was being satisfied by the propagation-path ordering
fallback instead, so the branch it claimed to cover never executed. The fixture
gained an edge (3->1) with neither endpoint on the propagation path, which
isolates the branch.

SURVIVED = the suite did not notice the sabotage = a coverage gap worth fixing.

This is a diagnostic, not a gate: some mutants are legitimately equivalent, and
the right response to a survivor is to look at it, not to reflexively add a test.

Python 3.9 compatible.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from typing import List, Tuple

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(PKG)

# (file, description, find, replace)
MUTANTS: List[Tuple[str, str, str, str]] = [
    ("evaluation/graph_claim_verifier.py",
     "reversal detection disabled (the old metric's blind spot)",
     "reversed_hits = [(a, b) for a, b in pairs if (b, a) in self.edges]",
     "reversed_hits = []"),

    ("evaluation/graph_claim_verifier.py",
     "self-attribution check disabled (Phase-13 failure mode)",
     "        if self.target is not None and ids == {self.target}:",
     "        if False:"),

    ("evaluation/graph_claim_verifier.py",
     "unresolved location treated as merely unsupported, not contradicted",
     '            return Verdict(\n                claim, CONTRADICTED,\n'
     '                "cited location resolves to no node in this graph; it names "',
     '            return Verdict(\n                claim, UNSUPPORTED,\n'
     '                "cited location resolves to no node in this graph; it names "'),

    ("evaluation/graph_claim_verifier.py",
     "numeric tolerance widened 10% -> 100% (accepts almost anything)",
     "NUMERIC_TOLERANCE = 0.10",
     "NUMERIC_TOLERANCE = 1.00"),

    ("evaluation/graph_claim_verifier.py",
     "confidence contradiction downgraded to supported",
     'if {tier, actual} == {"high", "low"}:',
     "if False:"),

    ("evaluation/graph_claim_verifier.py",
     "path traversability check disabled",
     "traversable = (self.adj is not None and all(",
     "traversable = (self.adj is None and all("),

    ("evaluation/claim_parser.py",
     "unparsed sentences silently dropped (the honesty mechanism)",
     "            unparsed.append(sent)",
     "            pass"),

    ("evaluation/claim_parser.py",
     "refusal detection disabled (breaks the no-answer control)",
     "    if isinstance(reasoning, str) and _REFUSAL_RE.search(reasoning):",
     "    if False:"),

    ("evaluation/claim_parser.py",
     "forecast-horizon guard disabled (Phase-13 mislabelling bug)",
     "        if is_horizon and not has_lag_language:",
     "        if False:"),
]


def run_suite() -> Tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "unittest",
         "xtraffic.tests.test_claim_verifier"],
        cwd=REPO, capture_output=True, text=True)
    return proc.returncode == 0, (proc.stderr or "") + (proc.stdout or "")


def main() -> int:
    ok, _ = run_suite()
    if not ok:
        print("Baseline suite is RED. Fix it before mutation testing.")
        return 1
    print("baseline: {} tests GREEN\n".format(
        run_suite()[1].strip().splitlines()[-3].split()[1]
        if "Ran" in run_suite()[1] else "?"))
    print("{:<62} {}".format("MUTANT", "RESULT"))
    print("-" * 76)

    killed, survived = 0, []
    tmpdir = tempfile.mkdtemp(prefix="xtraffic_mut_")
    try:
        for rel, desc, find, repl in MUTANTS:
            path = os.path.join(PKG, rel)
            backup = os.path.join(tmpdir, os.path.basename(rel))
            shutil.copy2(path, backup)
            src = open(path, encoding="utf-8").read()
            if find not in src:
                print("{:<62} {}".format(desc[:60], "SKIP (pattern not found)"))
                continue
            open(path, "w", encoding="utf-8").write(src.replace(find, repl, 1))
            try:
                passed, _ = run_suite()
            finally:
                shutil.copy2(backup, path)
            if passed:
                survived.append(desc)
                print("{:<62} {}".format(desc[:60], "SURVIVED  <-- gap"))
            else:
                killed += 1
                print("{:<62} {}".format(desc[:60], "killed"))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("-" * 76)
    print("killed {} / {}".format(killed, killed + len(survived)))
    if survived:
        print("\nSURVIVING MUTANTS — the suite does not notice these changes:")
        for s in survived:
            print("  - " + s)
        print("\nEach is either a missing test or a genuinely equivalent "
              "mutant. Inspect before adding tests reflexively.")
    ok_after, _ = run_suite()
    print("\nsuite restored and {}".format("GREEN" if ok_after else "RED"))
    return 0 if ok_after else 1


if __name__ == "__main__":
    raise SystemExit(main())
