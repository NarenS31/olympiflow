"""Phase 1 gate — MEASURE the reproducibility of the LLM layer, before and after.

    python -m xtraffic.scripts.measure_llm_determinism [--n-draws 4] [--model ...]

WHY THIS EXISTS
---------------
`docs/REPOSITORY_AUDIT.md` §5.2 claims that every committed LLM number is one
unreplicable draw, because `advisor.py` sent only `{"temperature": 0.1}` and no
seed. That is an assertion about the system's behaviour, and assertions about
behaviour should be measured, not argued.

This script runs the SAME real advisory prompt under two payload regimes:

  LEGACY  exactly the pre-Phase-1 payload: options={"temperature": 0.1}, no seed
          key sent at all. This is what produced every committed result.
  PHASE1  the new payload: explicit seed + pinned top_p/top_k/num_predict/num_ctx.

and reports how many DISTINCT outputs each produces over N identical requests.

A DEVELOPMENT NOTE WORTH KEEPING
--------------------------------
The first version of this check was WRONG and looked like a positive result.
It tried to express "unseeded" as `sampling={"seed": None}`, but `build_options`
treats None as "use the default", so it silently sent seed=42 — and the legacy
regime appeared perfectly deterministic. The bug was caught only because a
follow-up test asked whether the seed did anything at all, and found that
"unseeded" output matched seed=42 exactly, which is too tidy to be real.

`build_options(..., omit=["seed"])` now exists specifically so the legacy regime
can be reproduced honestly. The lesson generalises: a reproducibility check that
returns the answer you hoped for deserves more suspicion than one that does not.

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from ..reproducibility import llm_log, run_dir

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PKG_ROOT = os.path.join(REPO_ROOT, "xtraffic")

# The exact options dict pre-Phase-1 `advisor.py:_call_ollama` built.
LEGACY_OPTIONS = {"temperature": 0.1}


def load_real_prompt(scenario: str = "high_congestion") -> str:
    """A REAL committed advisory prompt, not a toy.

    Prompt length and structure matter: a two-token prompt can look
    deterministic purely because one continuation dominates. The committed
    scenarios are the actual measurement surface.
    """
    from ..models.advisor.advisor import Advisor, build_prompt
    path = os.path.join(PKG_ROOT, "evaluation", "results", "explanations",
                        "{}.json".format(scenario))
    with open(path, "r", encoding="utf-8") as fh:
        exp = json.load(fh)
    adv = Advisor("metr_la")
    kb_block = adv.kb.retrieve(exp, adv.top_k, adv.max_context_chars) \
        if hasattr(adv.kb, "retrieve") else ""
    if not isinstance(kb_block, str):
        kb_block = ""
    return build_prompt(exp, kb_block)


def measure(host: str, model: str, prompt: str, n_draws: int,
            logger: Optional[llm_log.LLMLogger]) -> Dict[str, Any]:
    """Both regimes, same prompt, same server, back to back."""
    legacy = llm_log.verify_determinism(
        host=host, model=model, prompt=prompt, n_draws=n_draws,
        sampling=LEGACY_OPTIONS, logger=logger,
        # The whole point: send NO seed, exactly as the committed code did.
        omit_options=["seed", "top_p", "top_k", "repeat_penalty",
                      "num_predict", "num_ctx"])

    phase1 = llm_log.verify_determinism(
        host=host, model=model, prompt=prompt, n_draws=n_draws, logger=logger)

    return {
        "prompt_n_chars": len(prompt),
        "prompt_n_words": len(prompt.split()),
        "n_draws": n_draws,
        "model": model,
        "legacy_pre_phase1": legacy,
        "phase1_seeded": phase1,
        "verdict": {
            "legacy_deterministic": legacy["byte_identical"],
            "phase1_deterministic": phase1["byte_identical"],
            "legacy_distinct_outputs": legacy["distinct_outputs"],
            "phase1_distinct_outputs": phase1["distinct_outputs"],
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--n-draws", type=int, default=4)
    ap.add_argument("--scenario", default="high_congestion")
    ap.add_argument("--no-run-dir", action="store_true",
                    help="print only; do not create a run directory")
    args = ap.parse_args(argv)

    prompt = load_real_prompt(args.scenario)
    cfg = {"experiment": {"name": "llm_determinism", "mode": "exploratory"},
           "llm": {"host": args.host, "model": args.model,
                   "n_draws": args.n_draws},
           "scenario": args.scenario}

    run: Optional[run_dir.RunDir] = None
    logger: Optional[llm_log.LLMLogger] = None
    if not args.no_run_dir:
        # Capture the SAME provenance surface run_experiment does, so a later
        # reproduce_run diff has something to compare against instead of a hole.
        from .run_experiment import artifacts_for, prompt_templates
        run = run_dir.RunDir.create(
            repo_root=REPO_ROOT, experiment="llm_determinism", config=cfg,
            artifacts=artifacts_for(cfg), prompt_templates=prompt_templates(),
            ollama_host=args.host, ollama_model=args.model,
            notes="Phase-1 gate: legacy (unseeded) vs Phase-1 (seeded) payload.")
        logger = llm_log.LLMLogger(run.file(run_dir.LLM_CALLS), run.run_id)
        run.start()
        print("run_id: {}\n".format(run.run_id))

    print("Prompt: {} chars / {} words (scenario={})".format(
        len(prompt), len(prompt.split()), args.scenario))
    print("Model : {}   draws per regime: {}\n".format(args.model, args.n_draws))

    res = measure(args.host, args.model, prompt, args.n_draws, logger)

    leg, ph1 = res["legacy_pre_phase1"], res["phase1_seeded"]
    print("-" * 68)
    print("{:<34} {:>14} {:>14}".format("", "LEGACY", "PHASE 1"))
    print("{:<34} {:>14} {:>14}".format(
        "seed sent", "none", str(ph1.get("seed"))))
    print("{:<34} {:>14} {:>14}".format(
        "distinct outputs / draws",
        "{}/{}".format(leg["distinct_outputs"], leg["n_successful"]),
        "{}/{}".format(ph1["distinct_outputs"], ph1["n_successful"])))
    print("{:<34} {:>14} {:>14}".format(
        "reproducibility rate",
        "{:.3f}".format(leg["reproducibility_rate"]),
        "{:.3f}".format(ph1["reproducibility_rate"])))
    print("{:<34} {:>14} {:>14}".format(
        "byte-identical", str(leg["byte_identical"]), str(ph1["byte_identical"])))
    print("{:<34} {:>14} {:>14}".format(
        "output length range",
        "{}-{}".format(min(leg["n_chars"]), max(leg["n_chars"])),
        "{}-{}".format(min(ph1["n_chars"]), max(ph1["n_chars"]))))
    print("-" * 68)

    if leg["byte_identical"]:
        print("\nLEGACY path was byte-identical here. Do NOT generalise: this is "
              "one prompt on one host in one session. Report the measurement.")
    else:
        print("\nCONFIRMED: the pre-Phase-1 payload is NONDETERMINISTIC — {} "
              "distinct outputs from {} identical requests.\nEvery committed LLM "
              "number is therefore ONE ARBITRARY DRAW from this distribution, "
              "not a reproducible value.".format(
                  leg["distinct_outputs"], leg["n_successful"]))
    if ph1["byte_identical"]:
        print("CONFIRMED: the Phase-1 seeded payload is byte-identical across "
              "{} draws on this host.\nScope of the claim: same server version, "
              "same model digest, same session. Cross-session and cross-version\n"
              "reproducibility is NOT established by this test and is not "
              "claimed.".format(ph1["n_draws"]))
    else:
        print("The Phase-1 seeded payload was NOT byte-identical ({} distinct). "
              "Report this rate; do not claim determinism.".format(
                  ph1["distinct_outputs"]))

    if run is not None:
        run.write_json("determinism_comparison.json", res)
        run.complete(**{"llm": llm_log.summarise_log(logger.read())})
        print("\nsaved -> {}".format(run.path))
    else:
        print(json.dumps(res["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
