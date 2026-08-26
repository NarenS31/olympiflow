"""Shared plumbing for the four parts of the grounding-without-information run.

Every part writes into ONE run directory (the brief: "run_dir only"), so they all
need the same three things: find that directory, log every LLM call into it, and
resume without redoing work. Keeping them here means no part script re-implements
resumability slightly differently.

Python 3.9 compatible.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

from ..reproducibility import run_dir
from ..utils.io_utils import PKG_ROOT

REPO_ROOT = os.path.dirname(PKG_ROOT)
EXPERIMENT = "grounding_without_information"


def resolve_run(run_id: Optional[str] = None) -> run_dir.RunDir:
    """Open the experiment's run directory (newest if not named).

    list_runs returns newest-first, so [0] is the current run.
    """
    if run_id:
        return run_dir.RunDir.open(
            os.path.join(PKG_ROOT, run_dir.RAW, run_id), REPO_ROOT)
    runs = run_dir.list_runs(REPO_ROOT, experiment=EXPERIMENT)
    if not runs:
        raise SystemExit(
            "no {} run directory found. Run "
            "`python -m xtraffic.scripts.init_grounding_noise_run` first — it "
            "writes the pre-registration, which must exist before any result "
            "does.".format(EXPERIMENT))
    return run_dir.RunDir.open(runs[0]["path"], REPO_ROOT)


# ---------------------------------------------------------------------------
# LLM call logging
# ---------------------------------------------------------------------------
def logged_call_fn(run: run_dir.RunDir, part: str, host: str, model: str,
                   temperature: float, timeout: float) -> Callable[..., Dict[str, Any]]:
    """A `call_fn` for Advisor that logs every call into the run directory.

    This is NOT reproducibility.llm_log.make_logged_caller. That one pins seed,
    top_p, top_k, num_predict and num_ctx, which changes generated text relative
    to the committed A / B / XTRAFFIC runs (REPOSITORY_AUDIT 5.2) — and those
    committed numbers are exactly what this experiment compares against. So we
    send the SAME payload the inline path sends (model, prompt, stream=False,
    format=json, options={temperature}) and add logging around it, nothing else.

    The consequence is stated in predictions.md: LLM output here is auditable and
    fully recorded, but not byte-reproducible on re-run.
    """
    import requests

    # Transport-level retries. WHY THIS EXISTS: without it, ONE transient
    # ReadTimeout kills a multi-hour study. Part 2 died at scenario 26 of 93
    # after 40 minutes because the advisor's `_call_ollama` turns any transport
    # failure into a RuntimeError and nothing upstream catches it. sim_eval's own
    # `_ollama_choose` already retries 3x with backoff for exactly this reason;
    # this path simply did not, which was an inconsistency, not a decision.
    #
    # A retry is safe here: a timed-out generate produced no result we kept, so
    # re-asking is not double-counting. The seeded/scored semantics are unchanged
    # because the prompt is identical and nothing was recorded for the failed try.
    max_transport_tries = 4

    def _post_with_retries(prompt: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        last: Optional[Exception] = None
        for attempt in range(max_transport_tries):
            try:
                resp = requests.post("{}/api/generate".format(host.rstrip("/")),
                                     json=payload, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.HTTPError) as exc:
                last = exc
                run.append_jsonl("llm_calls.jsonl", {
                    "ok": False, "part": part, "event": "transport_retry",
                    "attempt": attempt, "error_type": type(exc).__name__,
                    "error": str(exc)[:300], "prompt_chars": len(prompt)})
                time.sleep(5 * (attempt + 1))          # 5s, 10s, 15s
        raise last if last else RuntimeError("unreachable")

    def _call(prompt: str, retry_index: int = 0, **_: Any) -> Dict[str, Any]:
        payload = {"model": model, "prompt": prompt, "stream": False,
                   "format": "json", "options": {"temperature": temperature}}
        t0 = time.time()
        try:
            body = _post_with_retries(prompt, payload)
            rec: Dict[str, Any] = {
                "ok": True,
                "part": part,
                # Caller-set tag (condition, scenario, round) so the log can be
                # sliced later without re-deriving which call was which.
                "label": getattr(_call, "label", None),
                "model": model,
                "temperature": temperature,
                "retry_index": retry_index,
                "prompt": prompt,
                "prompt_chars": len(prompt),
                "raw_response": body.get("response", "").strip(),
                # prompt_eval_count is the tokenised prompt length straight from
                # the server: the number Part 1's 5% token-parity guard needs.
                "prompt_eval_count": body.get("prompt_eval_count"),
                "eval_count": body.get("eval_count"),
                "seconds": round(time.time() - t0, 2),
            }
        except Exception as exc:                          # noqa: BLE001
            rec = {"ok": False, "part": part,
                   "label": getattr(_call, "label", None), "model": model,
                   "retry_index": retry_index, "prompt": prompt,
                   "prompt_chars": len(prompt),
                   "error_type": type(exc).__name__, "error": str(exc),
                   "seconds": round(time.time() - t0, 2)}
        run.append_jsonl("llm_calls.jsonl", rec)
        return rec

    _call.label = None                                    # type: ignore[attr-defined]
    return _call


def count_prompt_tokens(prompt: str, host: str, model: str,
                        timeout: float = 180.0) -> Optional[int]:
    """Tokenised length of `prompt`, from the server that will consume it.

    num_predict=1, NOT 0. This Ollama build IGNORES num_predict=0 and decodes to
    its default length anyway — measured: num_predict=0 returned eval_count=123
    on a prompt we asked for zero tokens on, num_predict=1 returned eval_count=1.
    Asking for a single token is the cheapest honest way to make the server
    tokenise a prompt and tell us how long it was.

    Used for condition A, whose advisories are reused from the committed run and
    so have no logged token count of their own. A_rand / A_mismatch get theirs
    free from `prompt_eval_count` on their real advisory calls.
    """
    import requests

    resp = requests.post(
        "{}/api/generate".format(host.rstrip("/")),
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.1, "num_predict": 1}},
        timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("prompt_eval_count")


# ---------------------------------------------------------------------------
# Resumable per-item caching inside the run directory
# ---------------------------------------------------------------------------
class PartCache:
    """One JSON per completed item, under <run>/<part>_cache/.

    A 4-hour LLM study that cannot resume is a 4-hour study you only get to run
    once. Keys are caller-chosen and must encode everything that would change the
    result (scenario, condition, seed).
    """

    def __init__(self, run: run_dir.RunDir, part: str):
        self.dir = os.path.join(run.path, "{}_cache".format(part))
        os.makedirs(self.dir, exist_ok=True)

    def path(self, key: str) -> str:
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in key)
        return os.path.join(self.dir, safe + ".json")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        p = self.path(key)
        if not os.path.exists(p):
            return None
        with open(p) as fh:
            return json.load(fh)

    def put(self, key: str, value: Dict[str, Any]) -> None:
        with open(self.path(key), "w") as fh:
            json.dump(value, fh, indent=2, default=str)

    def all(self) -> List[Dict[str, Any]]:
        out = []
        for name in sorted(os.listdir(self.dir)):
            if name.endswith(".json"):
                with open(os.path.join(self.dir, name)) as fh:
                    out.append(json.load(fh))
        return out
