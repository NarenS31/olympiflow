"""Every LLM call, persisted in full. The single most important Phase-1 module.

TWO PROBLEMS THIS SOLVES (docs/REPOSITORY_AUDIT.md §5.2 and §5.3)
------------------------------------------------------------------
1. NO SEED. `models/advisor/advisor.py` built its Ollama payload as

       {"model": ..., "prompt": ..., "options": {"temperature": 0.1}}

   No seed, no num_predict, no top_p/top_k/num_ctx. Temperature 0.1 is not 0, so
   decoding is stochastic and no committed LLM number is reproducible. The
   project measured this itself in Phase 20: the same scenario, same prompt, same
   model gave F1 0.857 on one run and 0.333 on another, with a mean round-0
   |diff| of 0.078 across 20 scenarios — the same order as several REPORTED
   EFFECTS (the Phase-12 GNNExplainer-vs-SHAP gap is +0.05).

2. NO RAW OUTPUT. Of roughly 4,000 generations across all studies, about 100
   survive as text. `sim_eval`'s 2,736 decisions store the chosen action and
   nothing else. So no new metric can ever be applied retrospectively: the
   Phase-3 claim verifier CANNOT be run on the existing corpus, because the
   corpus does not exist. The project already paid this once — diagnosing the
   Phase-11 outlier "required re-running the LLM".

THE RULE THIS MODULE ENFORCES
-----------------------------
An LLM call that is not fully logged did not happen. Every call writes one JSONL
record containing the prompt, the raw response string, the parsed object, the
exact sampling parameters, the model digest, retry index, and timings. That
record is sufficient to re-score the call under any future metric without
touching the LLM again.

ON DETERMINISM — READ THIS BEFORE CLAIMING IT
---------------------------------------------
Passing `seed` to Ollama makes sampling reproducible on the SAME server version,
SAME model digest, and SAME loaded context. It is NOT a guarantee across model
reloads, server upgrades, or different hardware, and llama.cpp has known
batching-related nondeterminism. So:

  * we pass the seed and record it;
  * `verify_determinism()` MEASURES the achieved reproducibility rate;
  * the manifest records the measured rate, not an assumption.

If the measured rate is below 1.0, that is a finding to report, not a bug to
paper over. The Phase-11 analysis plan therefore treats DRAW as a nested random
factor regardless of what the rate turns out to be.

Python 3.9 compatible.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from . import provenance

# Default sampling parameters. `temperature` matches the committed
# configs/advisor.yaml value (0.1) so behaviour is unchanged where it is not
# deliberately overridden. Everything else was previously LEFT TO THE SERVER
# DEFAULT and unrecorded, which is exactly the problem.
DEFAULT_SAMPLING: Dict[str, Any] = {
    "seed": 42,
    "temperature": 0.1,
    "top_p": 0.9,             # llama.cpp default; pinned so it cannot drift
    "top_k": 40,              # llama.cpp default; pinned
    "repeat_penalty": 1.1,    # llama.cpp default; pinned
    "num_predict": 1024,      # HARD OUTPUT CAP — the token-budget control the
                              # matched ablation matrix (Phase 5) requires. The
                              # advisory JSON contract fits comfortably; a model
                              # that runs away gets truncated and the truncation
                              # is recorded rather than silently changing length
                              # across conditions.
    "num_ctx": 8192,          # input context window; pinned so a long-context
                              # adversarial condition cannot silently truncate
                              # differently from a short one.
}


class LLMLogger:
    """Append-only JSONL log of LLM calls.

    One logger per run. `path` normally lives inside a RunDir, so the log is
    part of the immutable run record.
    """

    def __init__(self, path: str, run_id: Optional[str] = None):
        self.path = path
        self.run_id = run_id
        self._n = 0
        d = os.path.dirname(path)
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Append one record, stamping run id and sequence number."""
        record = dict(record)
        record.setdefault("run_id", self.run_id)
        record["call_index"] = self._n
        record.setdefault("logged_utc", datetime.now(timezone.utc).isoformat())
        self._n += 1
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        return record

    def read(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        out: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue          # truncated tail from a killed run
        return out

    @property
    def n_calls(self) -> int:
        return self._n


def build_options(sampling: Optional[Dict[str, Any]] = None,
                  omit: Optional[List[str]] = None) -> Dict[str, Any]:
    """Merge caller overrides onto the pinned defaults.

    Returned dict goes verbatim into the Ollama payload AND verbatim into the
    log record, so what was requested and what was logged cannot diverge.

    CAREFUL — `None` means "use the default", NOT "do not send this key". This
    footgun bit the Phase-1 determinism check during development: passing
    `{"seed": None}` intending "unseeded" silently kept the default seed of 42,
    making an unseeded run look deterministic when it had simply been seeded.
    To genuinely omit a key — which is the only way to reproduce the
    pre-Phase-1 behaviour — pass it in `omit`:

        build_options({"temperature": 0.1}, omit=["seed"])   # truly unseeded
    """
    opts = dict(DEFAULT_SAMPLING)
    if sampling:
        opts.update({k: v for k, v in sampling.items() if v is not None})
    for key in (omit or []):
        opts.pop(key, None)
    return opts


def generate(host: str, model: str, prompt: str,
             sampling: Optional[Dict[str, Any]] = None,
             force_json: bool = True,
             timeout: float = 180.0,
             logger: Optional[LLMLogger] = None,
             context: Optional[Dict[str, Any]] = None,
             retry_index: int = 0,
             model_digest: Optional[str] = None,
             omit_options: Optional[List[str]] = None) -> Dict[str, Any]:
    """One Ollama generate call, fully parameterised and fully logged.

    Args:
        host:         Ollama base URL.
        model:        model tag, e.g. "llama3.1:8b".
        prompt:       the fully rendered prompt.
        sampling:     overrides for DEFAULT_SAMPLING.
        force_json:   set Ollama's `format="json"` constrained decoding.
        timeout:      seconds.
        logger:       where to persist. Omitting it is allowed for interactive
                      use but MUST NOT happen inside a study.
        context:      experiment bookkeeping (scenario_id, condition, domain,
                      draw index, ...). Copied into the record so the log is
                      self-describing and can be grouped without a join.
        retry_index:  which attempt this is (the advisor retries on malformed
                      JSON; each attempt is its own record, never overwritten).
        model_digest: resolved digest, if the caller already looked it up.

    Returns:
        A record dict containing `ok`, `raw_response`, `parsed`, `error`, timings
        and token counts. NEVER raises on a transport or decode failure — the
        failure is recorded and returned, because a study that dies on one bad
        call loses hours of work, and because failures are data.
    """
    import requests

    opts = build_options(sampling, omit=omit_options)
    payload: Dict[str, Any] = {"model": model, "prompt": prompt,
                               "stream": False, "options": opts}
    if force_json:
        payload["format"] = "json"

    rec: Dict[str, Any] = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "host": host,
        "model": model,
        "model_digest": model_digest,
        "options": opts,
        "force_json": force_json,
        "retry_index": retry_index,
        "prompt": prompt,
        "prompt_sha256": provenance.sha256_text(prompt),
        "prompt_n_chars": len(prompt),
        # Cheap, stable proxy for prompt length, used by the Phase-5 matched
        # ablation matrix to check that conditions are length-comparable. The
        # TRUE token count comes back from Ollama below when the call succeeds;
        # this is here so the check still works on failed calls.
        "prompt_n_words": len(prompt.split()),
    }
    if context:
        rec.update({k: v for k, v in context.items() if k not in rec})

    t0 = time.time()
    try:
        resp = requests.post(host.rstrip("/") + "/api/generate",
                             json=payload, timeout=timeout)
        rec["http_status"] = resp.status_code
        resp.raise_for_status()
        body = resp.json()
        raw = body.get("response", "")
        rec.update({
            "ok": True,
            "raw_response": raw,
            "raw_sha256": provenance.sha256_text(raw),
            "raw_n_chars": len(raw),
            # Ollama's own accounting. `eval_count` is the completion-token count
            # the matched-token-budget requirement needs.
            "prompt_eval_count": body.get("prompt_eval_count"),
            "eval_count": body.get("eval_count"),
            "total_duration_ns": body.get("total_duration"),
            "done_reason": body.get("done_reason"),
        })
        # A truncated completion changes output length in a way that would
        # otherwise silently confound a length-matched comparison. Flag it.
        if body.get("done_reason") == "length":
            rec["truncated"] = True
            rec["warning"] = ("completion hit num_predict={} and was truncated"
                              .format(opts.get("num_predict")))
        try:
            rec["parsed"] = json.loads(raw)
            rec["parse_ok"] = True
        except Exception as exc:
            rec["parsed"] = None
            rec["parse_ok"] = False
            rec["parse_error"] = str(exc)
    except Exception as exc:
        rec.update({"ok": False, "error": str(exc),
                    "error_type": type(exc).__name__,
                    "raw_response": None, "parsed": None, "parse_ok": False})

    rec["latency_ms"] = round((time.time() - t0) * 1000.0, 2)

    if logger is not None:
        logger.log(rec)
    return rec


def verify_determinism(host: str, model: str, prompt: str,
                       n_draws: int = 5,
                       sampling: Optional[Dict[str, Any]] = None,
                       logger: Optional[LLMLogger] = None,
                       timeout: float = 180.0,
                       omit_options: Optional[List[str]] = None) -> Dict[str, Any]:
    """Measure — do not assume — how reproducible this LLM actually is.

    Issues the SAME prompt with the SAME seed `n_draws` times and reports the
    fraction of draws whose raw output matches the first byte-for-byte.

    This is the Phase-1 gate. If it returns 1.0, seeded reproducibility holds on
    this host for this model and can be stated. If it returns less, the honest
    move is to REPORT THE MEASURED RATE and design the statistics around draw
    variance — which is what `analysis/statistical_plan.md` does regardless.
    """
    hashes: List[str] = []
    records: List[Dict[str, Any]] = []
    for i in range(n_draws):
        rec = generate(host, model, prompt, sampling=sampling, logger=logger,
                       timeout=timeout, omit_options=omit_options,
                       context={"purpose": "determinism_check", "draw": i,
                                "omitted_options": omit_options})
        records.append(rec)
        hashes.append(rec.get("raw_sha256") or "ERROR")

    ok = [h for h in hashes if h != "ERROR"]
    identical = sum(1 for h in ok if ok and h == ok[0])
    return {
        "n_draws": n_draws,
        "n_successful": len(ok),
        "n_identical_to_first": identical,
        "reproducibility_rate": (identical / len(ok)) if ok else 0.0,
        "distinct_outputs": len(set(ok)),
        "byte_identical": len(set(ok)) == 1 and len(ok) == n_draws,
        "seed": build_options(sampling, omit=omit_options).get("seed"),
        "omitted_options": omit_options,
        "hashes": hashes,
        "n_chars": [r.get("raw_n_chars") for r in records],
        "latency_ms": [r.get("latency_ms") for r in records],
    }


def make_logged_caller(host: str, model: str, logger: LLMLogger,
                       sampling: Optional[Dict[str, Any]] = None,
                       force_json: bool = True, timeout: float = 180.0,
                       model_digest: Optional[str] = None
                       ) -> Callable[..., Dict[str, Any]]:
    """Bind host/model/logger into a one-argument callable.

    This is the seam `models/advisor/advisor.py` plugs into: the Advisor accepts
    an optional `call_fn`, and when one is supplied every call it makes is
    logged with no other change to its behaviour. When it is not supplied the
    Advisor uses its original code path, so nothing pre-Phase-1 changes.
    """
    def _call(prompt: str, retry_index: int = 0,
              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return generate(host=host, model=model, prompt=prompt,
                        sampling=sampling, force_json=force_json,
                        timeout=timeout, logger=logger, context=context,
                        retry_index=retry_index, model_digest=model_digest)
    return _call


def summarise_log(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a call log: counts, failures, truncations, tokens, latency.

    `n_truncated` and `n_parse_failures` are the two a study must not ignore —
    both silently distort output length and content across conditions.
    """
    n = len(records)
    ok = [r for r in records if r.get("ok")]
    parse_ok = [r for r in ok if r.get("parse_ok")]
    lat = [r.get("latency_ms") for r in records if r.get("latency_ms") is not None]
    comp = [r.get("eval_count") for r in ok if r.get("eval_count") is not None]
    prom = [r.get("prompt_eval_count") for r in ok
            if r.get("prompt_eval_count") is not None]

    def _mean(xs: List[Any]) -> Optional[float]:
        vals = [float(x) for x in xs if x is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "n_calls": n,
        "n_transport_ok": len(ok),
        "n_transport_failed": n - len(ok),
        "n_parse_ok": len(parse_ok),
        "n_parse_failures": len(ok) - len(parse_ok),
        "n_truncated": sum(1 for r in records if r.get("truncated")),
        "n_retries": sum(1 for r in records if (r.get("retry_index") or 0) > 0),
        "mean_latency_ms": _mean(lat),
        "total_latency_s": (sum(float(x) for x in lat) / 1000.0) if lat else 0.0,
        "mean_prompt_tokens": _mean(prom),
        "mean_completion_tokens": _mean(comp),
        "total_completion_tokens": (int(sum(float(x) for x in comp))
                                    if comp else 0),
        "distinct_models": sorted({r.get("model") for r in records
                                   if r.get("model")}),
        "distinct_prompt_hashes": len({r.get("prompt_sha256")
                                       for r in records if r.get("prompt_sha256")}),
    }
