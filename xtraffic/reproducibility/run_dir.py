"""Versioned, append-only run directories. Never overwrite a result.

THE PROBLEM (docs/REPOSITORY_AUDIT.md §5.8)
--------------------------------------------
Results landed at fixed paths — `results/faithfulness/faithfulness_summary.json`
— and collisions were avoided by ad-hoc suffixes (`_15b`, `_dcontra`, `_smoke`)
chosen by hand. There was no run id, no manifest, and no guarantee that a rerun
would not clobber a committed number. `--smoke` writes `*_smoke.pt` specifically
because a smoke run once destroyed a real checkpoint.

Worse: `faithfulness_summary.json` still holds the SUPERSEDED pre-correction
numbers, and `make_paper_artifacts.py:424` still reads it. Nothing in the file
says it is superseded. This module makes that class of error structural rather
than a matter of remembering.

THE DESIGN
----------
    results/raw/<run_id>/            immutable; written once, never edited
        manifest.json                config + provenance + status
        llm_calls.jsonl              every LLM call (see llm_log.py)
        <study outputs>
    results/processed/<run_id>/      derived tables/figures; safe to regenerate

    run_id = <UTC timestamp>__<experiment>__<config-hash-8>__<git-sha-8>

Two runs of the same config on the same commit differ only in timestamp, so they
sort chronologically and never collide. The config hash in the name means you can
tell at a glance whether two runs used the same settings.

`create()` REFUSES to write into an existing directory. There is no `force`
flag and that is deliberate: the correct response to "I want to redo that run" is
a new run directory plus, if genuinely wanted, a `superseded_by` pointer written
into the old one's manifest (`mark_superseded`).

Python 3.9 compatible.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import provenance

RAW = os.path.join("evaluation", "results", "raw")
PROCESSED = os.path.join("evaluation", "results", "processed")

MANIFEST = "manifest.json"
LLM_CALLS = "llm_calls.jsonl"

# Status values a manifest can carry. `created` -> `running` -> one terminal
# state. A run left in `running` after the process exits means it CRASHED, which
# is itself information worth preserving rather than tidying away.
STATUS_CREATED = "created"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_SUPERSEDED = "superseded"


def make_run_id(experiment: str, config: Dict[str, Any],
                git_commit: Optional[str] = None,
                now: Optional[datetime] = None) -> str:
    """Build a sortable, self-describing run id.

    Timestamp first so plain lexical sort is chronological. Config hash and git
    sha are truncated to 8 chars — enough to distinguish runs by eye, with the
    full values always present in the manifest.
    """
    ts = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    cfg8 = provenance.sha256_obj(config)[:8]
    git8 = (git_commit or "nogit")[:8]
    safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in experiment)
    return "{}__{}__{}__{}".format(ts, safe, cfg8, git8)


class RunDir:
    """One immutable run directory.

    Usage:
        run = RunDir.create(repo_root, "ablation_matrix", cfg,
                            ollama_host=..., ollama_model=...)
        run.write_json("summary.json", summary)
        run.complete()
    """

    def __init__(self, path: str, run_id: str, repo_root: str):
        self.path = path
        self.run_id = run_id
        self.repo_root = repo_root

    # -- construction -------------------------------------------------------
    @classmethod
    def create(cls, repo_root: str, experiment: str, config: Dict[str, Any],
               artifacts: Optional[Dict[str, str]] = None,
               ollama_host: Optional[str] = None,
               ollama_model: Optional[str] = None,
               prompt_templates: Optional[Dict[str, str]] = None,
               root: Optional[str] = None,
               notes: Optional[str] = None) -> "RunDir":
        """Create a fresh run directory and write its manifest.

        Raises:
            FileExistsError: if the directory already exists. Intentional and
                not overridable — see the module docstring.
        """
        prov = provenance.capture(
            repo_root=repo_root, config=config, artifacts=artifacts,
            ollama_host=ollama_host, ollama_model=ollama_model,
            prompt_templates=prompt_templates)

        commit = prov.get("git", {}).get("commit")
        run_id = make_run_id(experiment, config, commit)
        base = root or os.path.join(repo_root, "xtraffic", RAW)
        path = os.path.join(base, run_id)

        if os.path.exists(path):
            raise FileExistsError(
                "run directory already exists: {}\n"
                "Runs are append-only. Start a new run rather than overwriting; "
                "use mark_superseded() on the old one if it is being replaced."
                .format(path))

        os.makedirs(path)
        run = cls(path, run_id, repo_root)
        run._write_manifest({
            "run_id": run_id,
            "experiment": experiment,
            "status": STATUS_CREATED,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
            "config": config,
            "provenance": prov,
        })
        return run

    @classmethod
    def open(cls, path: str, repo_root: Optional[str] = None) -> "RunDir":
        """Open an existing run directory for reading (or appending outputs)."""
        mpath = os.path.join(path, MANIFEST)
        if not os.path.exists(mpath):
            raise FileNotFoundError("no manifest at {}".format(mpath))
        with open(mpath, "r", encoding="utf-8") as fh:
            man = json.load(fh)
        return cls(path, man.get("run_id", os.path.basename(path)),
                   repo_root or man.get("provenance", {}).get("repo_root", ""))

    # -- manifest -----------------------------------------------------------
    @property
    def manifest_path(self) -> str:
        return os.path.join(self.path, MANIFEST)

    def read_manifest(self) -> Dict[str, Any]:
        with open(self.manifest_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write_manifest(self, man: Dict[str, Any]) -> None:
        # Atomic replace: write a temp file then rename, so a crash mid-write
        # cannot leave a truncated manifest that makes the whole run
        # unreadable.
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(man, fh, indent=2, default=str)
        os.replace(tmp, self.manifest_path)

    def update_manifest(self, **fields: Any) -> Dict[str, Any]:
        man = self.read_manifest()
        man.update(fields)
        man["updated_utc"] = datetime.now(timezone.utc).isoformat()
        self._write_manifest(man)
        return man

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        self.update_manifest(status=STATUS_RUNNING,
                             started_utc=datetime.now(timezone.utc).isoformat())

    def complete(self, **summary: Any) -> None:
        self.update_manifest(status=STATUS_COMPLETED,
                             finished_utc=datetime.now(timezone.utc).isoformat(),
                             summary=summary or None,
                             outputs=self.list_outputs())

    def fail(self, error: str) -> None:
        """Record a failure IN the run directory. A failed run is data: it keeps
        its partial outputs and its manifest says why it stopped. Deleting
        failed runs is how a project loses track of what it actually tried."""
        self.update_manifest(status=STATUS_FAILED,
                             finished_utc=datetime.now(timezone.utc).isoformat(),
                             error=error, outputs=self.list_outputs())

    def mark_superseded(self, by_run_id: str, reason: str) -> None:
        """Mark this run superseded by a later one, IN THE ARTIFACT.

        This is the fix for the audit's §5.4: the superseded Phase-5 summary
        carries no marker of its own, so `make_paper_artifacts.py` reads it
        without complaint and the only warning lives in prose. A superseded run
        must announce itself."""
        self.update_manifest(
            status=STATUS_SUPERSEDED, superseded_by=by_run_id,
            superseded_reason=reason,
            superseded_utc=datetime.now(timezone.utc).isoformat())

    # -- outputs ------------------------------------------------------------
    def file(self, name: str) -> str:
        """Absolute path for an output file inside this run (parents created)."""
        p = os.path.join(self.path, name)
        d = os.path.dirname(p)
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        return p

    def write_json(self, name: str, obj: Any) -> str:
        path = self.file(name)
        if os.path.exists(path):
            raise FileExistsError(
                "refusing to overwrite {} inside an append-only run "
                "directory".format(path))
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, default=str)
        return path

    def write_text(self, name: str, text: str) -> str:
        path = self.file(name)
        if os.path.exists(path):
            raise FileExistsError(
                "refusing to overwrite {} inside an append-only run "
                "directory".format(path))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def append_jsonl(self, name: str, record: Dict[str, Any]) -> str:
        """Append one record. The ONLY sanctioned way to add to an existing file
        in a run directory — append-only is compatible with immutability, and it
        is what makes long LLM studies resumable."""
        path = self.file(name)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        return path

    def read_jsonl(self, name: str) -> List[Dict[str, Any]]:
        path = os.path.join(self.path, name)
        if not os.path.exists(path):
            return []
        out: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        # A truncated final line (killed mid-write) is expected
                        # in a resumable study. Skip it; do not lose the file.
                        continue
        return out

    def list_outputs(self) -> List[Dict[str, Any]]:
        """Every file in the run, with size and sha256 — the run's own integrity
        record, so later tampering or truncation is detectable."""
        out: List[Dict[str, Any]] = []
        for dirpath, _, names in os.walk(self.path):
            for n in sorted(names):
                if n in (MANIFEST, MANIFEST + ".tmp"):
                    continue
                full = os.path.join(dirpath, n)
                rel = os.path.relpath(full, self.path)
                try:
                    size = os.path.getsize(full)
                except OSError:                    # pragma: no cover
                    size = None
                out.append({"file": rel, "size_bytes": size,
                            "sha256": provenance.sha256_file(full)})
        return out


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def list_runs(repo_root: str, experiment: Optional[str] = None,
              root: Optional[str] = None) -> List[Dict[str, Any]]:
    """All runs, newest first. Unreadable manifests are reported, not hidden."""
    base = root or os.path.join(repo_root, "xtraffic", RAW)
    if not os.path.isdir(base):
        return []
    rows: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(base), reverse=True):
        path = os.path.join(base, name)
        if not os.path.isdir(path):
            continue
        try:
            with open(os.path.join(path, MANIFEST), "r", encoding="utf-8") as fh:
                man = json.load(fh)
        except Exception as exc:
            rows.append({"run_id": name, "path": path,
                         "error": "unreadable manifest: {}".format(exc)})
            continue
        if experiment and man.get("experiment") != experiment:
            continue
        rows.append({
            "run_id": man.get("run_id", name),
            "path": path,
            "experiment": man.get("experiment"),
            "status": man.get("status"),
            "created_utc": man.get("created_utc"),
            "git": man.get("provenance", {}).get("git", {}).get("commit_short"),
            "dirty": man.get("provenance", {}).get("git", {}).get("dirty"),
            "config_sha256": man.get("provenance", {}).get("config_sha256"),
            "superseded_by": man.get("superseded_by"),
            "n_warnings": len(man.get("provenance", {}).get("warnings", [])),
        })
    return rows


def latest_run(repo_root: str, experiment: str,
               root: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Most recent COMPLETED, non-superseded run of an experiment.

    Deliberately skips failed, running, and superseded runs: "the latest result"
    should never silently be a crashed one or a retracted one.
    """
    for row in list_runs(repo_root, experiment, root):
        if row.get("status") == STATUS_COMPLETED and not row.get("superseded_by"):
            return row
    return None
