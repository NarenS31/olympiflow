"""Capture WHAT produced a result, so a number can be traced to its inputs.

THE PROBLEM THIS SOLVES (docs/REPOSITORY_AUDIT.md §5.5)
-------------------------------------------------------
Before Phase 1, an explanation recorded:

    "meta": {"model_checkpoint": "metr_la_best.pt", ...}

a FILENAME. That file was overwritten three times (3-epoch placeholder -> epoch
34 -> epoch 54), so no committed artifact can say which model produced it. The
project hit the consequence directly in Phase 17: six cached explanations
silently predated the checkpoint they claimed, because
`build_or_load_explanation` caches on filename presence alone.

Content addressing fixes this permanently. A checkpoint is identified by the
sha256 of its bytes; a stale cache becomes detectable instead of invisible.

WHAT IS CAPTURED
----------------
  code        git commit, branch, dirty flag, diff hash if dirty
  interpreter python version, implementation, executable path
  packages    installed versions of everything requirements.txt pins, PLUS an
              explicit MISMATCH list (see below)
  platform    OS, machine, processor, torch device availability
  artifacts   sha256 of every checkpoint / data file / config the run reads
  llm         Ollama server version + the resolved model DIGEST (not the tag —
              "llama3.1:8b" is a mutable pointer, the digest is not)
  metric      the entity resolver's active fuzzy backend
  timing      UTC start/end, duration

WHY THE PACKAGE MISMATCH LIST MATTERS
-------------------------------------
`requirements.txt` pins torch==2.2.2 and torch-geometric==2.5.3 and
rapidfuzz==3.6.1. On the machine this module was written on, torch 2.8.0 was
installed and BOTH torch-geometric and rapidfuzz were absent. So the pinned file
described an environment that did not exist. rapidfuzz's absence is not cosmetic:
`evaluation/faithfulness.py` silently falls back to `difflib.SequenceMatcher`, a
DIFFERENT similarity function compared against the SAME threshold of 82. The
resolver — and therefore every faithfulness number — depends on which backend is
installed, and nothing recorded which one it was.

`compare_to_requirements()` makes that visible on every single run.

Everything here degrades gracefully: a missing git binary, an absent Ollama
server, or an unreadable file produces a recorded ERROR STRING, never an
exception and never a silently omitted field. A provenance module that crashes
the experiment it is documenting is worse than useless.

Python 3.9 compatible.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

# Read in 1 MiB blocks so a 100 MB tensor file does not land in memory at once.
_HASH_BLOCK = 1024 * 1024

# Packages we care about pinning. Kept explicit rather than scraping the whole
# environment: a full `pip freeze` is also captured, but this shortlist is what
# gets diffed against requirements.txt and shown in the mismatch report.
_TRACKED_PACKAGES = [
    "torch", "torch-geometric", "numpy", "pandas", "scipy", "rapidfuzz",
    "pandapower", "shap", "pyyaml", "requests", "matplotlib",
    "fastapi", "uvicorn", "python-multipart",
]


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def sha256_file(path: str) -> str:
    """sha256 of a file's bytes, or an "ERROR: ..." string if unreadable.

    Never raises. A provenance failure must not take down the experiment; it
    must be RECORDED so the manifest shows a hole rather than a false value.
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while True:
                block = fh.read(_HASH_BLOCK)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()
    except Exception as exc:
        return "ERROR: {}".format(exc)


def sha256_text(text: str) -> str:
    """sha256 of a string (UTF-8). Used for prompt templates and rendered
    prompts, so a prompt change is detectable without storing the prompt twice."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj: Any) -> str:
    """sha256 of a JSON-serialisable object under a CANONICAL encoding
    (sorted keys, no incidental whitespace), so logically-equal configs hash
    equal regardless of key order. This is what makes a run id stable."""
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                  default=str))


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------
def _git(args: Sequence[str], repo: str) -> Optional[str]:
    """Run a git command, returning stripped stdout or None on any failure."""
    try:
        out = subprocess.run(["git"] + list(args), cwd=repo, capture_output=True,
                             text=True, timeout=15)
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except Exception:
        return None


def git_provenance(repo: str) -> Dict[str, Any]:
    """Commit, branch, and — critically — whether the tree was DIRTY.

    A result produced from a dirty tree is not reproducible from its commit
    alone, so we additionally hash the diff. Two dirty runs with the same
    `diff_sha256` came from the same uncommitted state; different hashes mean
    the code moved underneath them.
    """
    commit = _git(["rev-parse", "HEAD"], repo)
    if commit is None:
        return {"available": False,
                "note": "git unavailable or not a repository at {}".format(repo)}

    status = _git(["status", "--porcelain"], repo)
    dirty = bool(status)
    rec: Dict[str, Any] = {
        "available": True,
        "commit": commit,
        "commit_short": commit[:12],
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], repo),
        "dirty": dirty,
        "commit_utc": _git(["show", "-s", "--format=%cI", "HEAD"], repo),
        "subject": _git(["show", "-s", "--format=%s", "HEAD"], repo),
    }
    if dirty:
        diff = _git(["diff", "HEAD"], repo) or ""
        rec["dirty_files"] = [ln[3:] for ln in (status or "").splitlines()][:100]
        rec["diff_sha256"] = sha256_text(diff)
        rec["warning"] = ("working tree was DIRTY: this run is not reproducible "
                          "from the commit alone")
    return rec


# ---------------------------------------------------------------------------
# Packages
# ---------------------------------------------------------------------------
def _installed_version(pkg: str) -> Optional[str]:
    """Installed version of one distribution, or None if absent."""
    try:
        from importlib import metadata as importlib_metadata  # py3.8+
    except Exception:                                         # pragma: no cover
        return None
    try:
        return importlib_metadata.version(pkg)
    except Exception:
        return None


def parse_requirements(path: str) -> Dict[str, str]:
    """Parse `name==version` pins out of a requirements file, ignoring comments,
    blank lines, and any non-`==` requirement form."""
    pins: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.split("#", 1)[0].strip()
                if not line or "==" not in line:
                    continue
                name, _, ver = line.partition("==")
                pins[name.strip().lower()] = ver.strip()
    except Exception:
        pass
    return pins


def compare_to_requirements(requirements_path: str) -> Dict[str, Any]:
    """Installed versions vs the pins, with an explicit mismatch list.

    Three outcomes per package, all reported:
      match     installed == pinned
      MISMATCH  installed != pinned  (a silent scientific hazard)
      MISSING   pinned but not installed (the rapidfuzz case — changes results)
    """
    pins = parse_requirements(requirements_path)
    installed: Dict[str, Optional[str]] = {
        p: _installed_version(p) for p in _TRACKED_PACKAGES}

    mismatches: List[Dict[str, Any]] = []
    for name, pinned in sorted(pins.items()):
        got = _installed_version(name)
        if got is None:
            mismatches.append({"package": name, "pinned": pinned,
                               "installed": None, "status": "MISSING"})
        elif got != pinned:
            mismatches.append({"package": name, "pinned": pinned,
                               "installed": got, "status": "MISMATCH"})

    return {
        "requirements_path": requirements_path,
        "requirements_sha256": sha256_file(requirements_path),
        "pinned": pins,
        "installed": installed,
        "mismatches": mismatches,
        "n_mismatches": len(mismatches),
        "environment_matches_requirements": len(mismatches) == 0,
    }


def pip_freeze() -> List[str]:
    """Full environment listing, for the record. Best-effort."""
    try:
        from importlib import metadata as importlib_metadata
        out = []
        for dist in importlib_metadata.distributions():
            try:
                out.append("{}=={}".format(dist.metadata["Name"], dist.version))
            except Exception:
                continue
        return sorted(out)
    except Exception:                              # pragma: no cover
        return []


# ---------------------------------------------------------------------------
# Interpreter / platform
# ---------------------------------------------------------------------------
def interpreter_provenance() -> Dict[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    }


def platform_provenance() -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "node": platform.node(),
    }
    try:
        import torch
        rec["torch_version"] = torch.__version__
        rec["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            rec["cuda_version"] = torch.version.cuda
            rec["gpu_name"] = torch.cuda.get_device_name(0)
        mps = getattr(torch.backends, "mps", None)
        rec["mps_available"] = bool(mps is not None and mps.is_available())
    except Exception as exc:
        rec["torch_error"] = str(exc)
    return rec


# ---------------------------------------------------------------------------
# The metric's own configuration
# ---------------------------------------------------------------------------
def metric_provenance() -> Dict[str, Any]:
    """Which fuzzy backend the entity resolver will actually use, and its
    threshold.

    THIS IS NOT COSMETIC. `faithfulness.py` prefers `rapidfuzz.token_set_ratio`
    (order-insensitive, tolerant of extra words) and silently falls back to
    `difflib.SequenceMatcher` (a character-matching ratio). Both are compared
    against the SAME threshold of 82, but they are different functions, so the
    same citation can resolve under one and not the other. Every faithfulness
    number therefore depends on which backend was installed — and until now,
    nothing recorded it.
    """
    rec: Dict[str, Any] = {}
    try:
        from ..evaluation import faithfulness as F
        rec["fuzzy_backend"] = F.FUZZ_BACKEND
        rec["fuzzy_threshold"] = F.FUZZY_THRESHOLD
        rec["numeric_tolerance"] = F.NUMERIC_TOLERANCE
        rec["rapidfuzz_installed"] = (_installed_version("rapidfuzz") is not None)
        if not rec["rapidfuzz_installed"]:
            rec["warning"] = (
                "rapidfuzz is NOT installed; the resolver is running on the "
                "difflib fallback, which is a DIFFERENT similarity function "
                "against the same threshold. Results are not comparable to a "
                "rapidfuzz run.")
    except Exception as exc:
        rec["error"] = str(exc)
    return rec


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
def ollama_provenance(host: str, model: Optional[str] = None,
                      timeout: float = 5.0) -> Dict[str, Any]:
    """Ollama server version + the DIGEST of the requested model.

    A tag like "llama3.1:8b" is a mutable pointer — pulling again can change what
    it resolves to. The digest is content-addressed and is what belongs in a
    manifest. Absence of a running server is recorded, never raised, so an
    offline analysis run still produces a complete manifest.
    """
    rec: Dict[str, Any] = {"host": host, "requested_model": model}
    try:
        import requests
    except Exception as exc:                       # pragma: no cover
        rec["error"] = "requests unavailable: {}".format(exc)
        return rec

    try:
        r = requests.get(host.rstrip("/") + "/api/version", timeout=timeout)
        rec["server_version"] = r.json().get("version") if r.ok else None
    except Exception as exc:
        rec["server_version"] = None
        rec["server_error"] = str(exc)

    try:
        r = requests.get(host.rstrip("/") + "/api/tags", timeout=timeout)
        if not r.ok:
            rec["error"] = "GET /api/tags -> HTTP {}".format(r.status_code)
            return rec
        models = r.json().get("models", []) or []
        rec["available_models"] = sorted(m.get("name", "") for m in models)
        if model:
            hit = next((m for m in models if m.get("name") == model), None)
            if hit is None:
                rec["model_present"] = False
                rec["warning"] = (
                    "requested model {!r} is NOT installed on this Ollama "
                    "server".format(model))
            else:
                det = hit.get("details", {}) or {}
                rec.update({
                    "model_present": True,
                    "model_digest": hit.get("digest"),
                    "model_size_bytes": hit.get("size"),
                    "model_modified_at": hit.get("modified_at"),
                    "model_family": det.get("family"),
                    "model_parameter_size": det.get("parameter_size"),
                    "model_quantization": det.get("quantization_level"),
                })
    except Exception as exc:
        rec["error"] = str(exc)
    return rec


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
def artifact_provenance(paths: Dict[str, str]) -> Dict[str, Any]:
    """Hash a named set of input files: checkpoints, tensors, configs, KBs.

    Args:
        paths: label -> filesystem path, e.g. {"checkpoint": ".../metr_la_best.pt"}

    Each entry records existence, size, mtime, and sha256. A missing file is
    recorded as `exists: False` rather than skipped, so the manifest shows the
    hole. `sha256` is what makes checkpoint identity survive a filename reuse.
    """
    out: Dict[str, Any] = {}
    for label, path in sorted(paths.items()):
        if not path:
            out[label] = {"path": path, "exists": False, "note": "empty path"}
            continue
        exists = os.path.exists(path)
        rec: Dict[str, Any] = {"path": path, "exists": exists}
        if exists:
            try:
                st = os.stat(path)
                rec["size_bytes"] = st.st_size
                rec["mtime_utc"] = datetime.fromtimestamp(
                    st.st_mtime, tz=timezone.utc).isoformat()
            except Exception as exc:               # pragma: no cover
                rec["stat_error"] = str(exc)
            rec["sha256"] = sha256_file(path)
        out[label] = rec
    return out


# ---------------------------------------------------------------------------
# The full record
# ---------------------------------------------------------------------------
def capture(repo_root: str,
            config: Optional[Dict[str, Any]] = None,
            artifacts: Optional[Dict[str, str]] = None,
            ollama_host: Optional[str] = None,
            ollama_model: Optional[str] = None,
            prompt_templates: Optional[Dict[str, str]] = None,
            extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble the complete provenance record for one run.

    Args:
        repo_root:        directory containing the git repository.
        config:           the run's resolved config (hashed AND stored verbatim).
        artifacts:        label -> path for every input file to hash.
        ollama_host:      e.g. "http://localhost:11434"; None skips the LLM block.
        ollama_model:     the model tag whose digest to resolve.
        prompt_templates: name -> template text; each is hashed so a prompt
                          change is detectable without diffing whole strings.
        extra:            anything caller-specific.

    Never raises.
    """
    req_path = os.path.join(repo_root, "xtraffic", "requirements.txt")
    if not os.path.exists(req_path):
        req_path = os.path.join(repo_root, "requirements.txt")

    rec: Dict[str, Any] = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": repo_root,
        "git": git_provenance(repo_root),
        "interpreter": interpreter_provenance(),
        "platform": platform_provenance(),
        "packages": compare_to_requirements(req_path),
        "metric": metric_provenance(),
    }

    if config is not None:
        rec["config_sha256"] = sha256_obj(config)
        rec["config"] = config

    if artifacts:
        rec["artifacts"] = artifact_provenance(artifacts)

    if prompt_templates:
        rec["prompt_templates"] = {
            name: {"sha256": sha256_text(text), "n_chars": len(text)}
            for name, text in sorted(prompt_templates.items())
        }

    if ollama_host:
        rec["llm"] = ollama_provenance(ollama_host, ollama_model)

    if extra:
        rec["extra"] = extra

    # Surface the things a reader must not miss, collected in one place so a
    # reviewer does not have to hunt through nested blocks for them.
    warnings: List[str] = []
    if rec["git"].get("dirty"):
        warnings.append("git working tree was dirty")
    if rec["packages"].get("n_mismatches"):
        warnings.append("{} package(s) differ from requirements.txt".format(
            rec["packages"]["n_mismatches"]))
    if rec["metric"].get("warning"):
        warnings.append(rec["metric"]["warning"])
    if rec.get("llm", {}).get("warning"):
        warnings.append(rec["llm"]["warning"])
    rec["warnings"] = warnings

    return rec


def summary_lines(rec: Dict[str, Any]) -> List[str]:
    """Short human-readable digest for console output at run start."""
    g = rec.get("git", {})
    lines = [
        "git       : {} ({}){}".format(
            g.get("commit_short", "n/a"), g.get("branch", "n/a"),
            "  [DIRTY]" if g.get("dirty") else ""),
        "python    : {}".format(rec.get("interpreter", {}).get("python_version")),
        "platform  : {} {}".format(rec.get("platform", {}).get("system"),
                                   rec.get("platform", {}).get("machine")),
        "packages  : {} mismatch(es) vs requirements.txt".format(
            rec.get("packages", {}).get("n_mismatches", "?")),
        "resolver  : {}".format(rec.get("metric", {}).get("fuzzy_backend", "n/a")),
    ]
    llm = rec.get("llm")
    if llm:
        dig = llm.get("model_digest")
        lines.append("llm       : {} digest={}".format(
            llm.get("requested_model"), (dig[:12] + "...") if dig else "n/a"))
    for w in rec.get("warnings", []):
        lines.append("WARNING   : {}".format(w))
    return lines
