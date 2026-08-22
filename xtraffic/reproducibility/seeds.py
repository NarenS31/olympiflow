"""Deterministic seed control across every RNG the pipeline touches.

WHAT WAS WRONG BEFORE (docs/REPOSITORY_AUDIT.md §5.7)
-----------------------------------------------------
`models/gnn/train.py:41` was the only `set_seed()` in the repository and it
covered four RNGs:

    random.seed / np.random.seed / torch.manual_seed / torch.cuda.manual_seed_all

Missing: cuDNN determinism flags, `torch.use_deterministic_algorithms`, the
CUBLAS workspace variable that CUDA >= 10.2 needs for deterministic matmuls, and
DataLoader worker seeding. `evaluation/cross_city.py:59` reimplemented a WEAKER
version (no CUDA). `evaluation/explainer_metrics.py:87` hardcoded seed 1234.

THE HONESTY CLAUSE
------------------
Full bitwise determinism is NOT achievable in this project and we say so in the
manifest rather than implying otherwise:

  * Training ran on Apple MPS locally and CUDA/T4 on Colab. MPS falls back to CPU
    for `einsum`, and float reduction order differs across all three backends.
    Cross-device bitwise identity is impossible, not merely difficult.
  * `torch.use_deterministic_algorithms(True)` makes some ops RAISE rather than
    silently pick a nondeterministic kernel. That is the point — but it can break
    code that previously ran. It is therefore OPT-IN via `strict=True`, and
    whether it was enabled is recorded.
  * The LLM is a separate process (Ollama). Seeding it is llm_log's job, not
    this module's, and even then Ollama does not guarantee reproducibility across
    model reloads or server versions.

`describe()` returns exactly what was and was not enforced, so the manifest
carries the truth instead of a claim.

Python 3.9 compatible.
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

try:
    import numpy as np
except Exception:                                  # pragma: no cover
    np = None                                      # type: ignore

try:
    import torch
except Exception:                                  # pragma: no cover
    torch = None                                   # type: ignore


# The CUBLAS workspace configuration CUDA >= 10.2 requires before
# `use_deterministic_algorithms(True)` will permit deterministic matmuls.
# ":4096:8" is the value the PyTorch docs prescribe. Set BEFORE the first CUDA
# context is created, which is why it lives here and not in a shell script.
_CUBLAS_WORKSPACE = ":4096:8"


def set_all_seeds(seed: int, strict: bool = False) -> Dict[str, Any]:
    """Seed every RNG in the process. Returns the record of what was enforced.

    Args:
        seed:   the integer seed. Applied to Python, NumPy, torch (CPU + all
                CUDA devices), and exported as PYTHONHASHSEED for child
                processes (the ablation harness shells out via subprocess).
        strict: if True, additionally request deterministic algorithms from
                torch and disable cuDNN autotuning. This can make some ops RAISE
                instead of silently running a nondeterministic kernel — which is
                the desired behaviour for a confirmatory run, but is off by
                default so it can never silently break an exploratory one.

    Returns:
        A dict for the run manifest. `enforced` lists what was actually applied;
        `not_enforceable` names what cannot be guaranteed on this hardware, so
        the manifest never overclaims.
    """
    rec: Dict[str, Any] = {
        "seed": int(seed),
        "strict": bool(strict),
        "enforced": [],
        "not_enforceable": [],
        "warnings": [],
    }

    # --- Python's own RNG + hash randomisation ------------------------------
    random.seed(seed)
    rec["enforced"].append("python.random")

    # PYTHONHASHSEED only takes effect at INTERPRETER STARTUP, so setting it here
    # cannot affect THIS process — it affects children (evaluation/ablations.py
    # trains each variant via subprocess). Recorded honestly as child-only.
    os.environ["PYTHONHASHSEED"] = str(seed)
    rec["enforced"].append("PYTHONHASHSEED (child processes only)")

    # --- NumPy --------------------------------------------------------------
    if np is not None:
        np.random.seed(seed)
        rec["enforced"].append("numpy.random (legacy global)")
        # NOTE: the studies mostly use explicit np.random.RandomState(seed) /
        # random.Random(seed) INSTANCES, which is the better pattern and is
        # unaffected by (and immune to) this global. Seeding the global is a
        # backstop for any library that reaches for it.
    else:                                          # pragma: no cover
        rec["warnings"].append("numpy not importable; global numpy seed skipped")

    # --- torch --------------------------------------------------------------
    if torch is not None:
        torch.manual_seed(seed)
        rec["enforced"].append("torch.manual_seed")

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            rec["enforced"].append("torch.cuda.manual_seed_all")
            if strict:
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE)
                rec["enforced"].append(
                    "CUBLAS_WORKSPACE_CONFIG=" + _CUBLAS_WORKSPACE)
        else:
            rec["not_enforceable"].append("cuda (no CUDA device on this host)")

        # MPS has no seed-all API and no determinism guarantee. Record it rather
        # than pretending. This is the local training backend, so it matters.
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            rec["not_enforceable"].append(
                "mps: no deterministic-algorithm guarantee; einsum falls back to "
                "CPU, so MPS<->CUDA<->CPU results differ in float reduction order")

        if strict:
            cudnn = getattr(torch.backends, "cudnn", None)
            if cudnn is not None:
                cudnn.deterministic = True
                cudnn.benchmark = False
                rec["enforced"].append("cudnn.deterministic=True, benchmark=False")
            try:
                torch.use_deterministic_algorithms(True)
                rec["enforced"].append("torch.use_deterministic_algorithms(True)")
            except Exception as exc:               # pragma: no cover
                # Do NOT swallow this. If determinism was requested and refused,
                # the manifest must say so or it is a false claim.
                rec["warnings"].append(
                    "use_deterministic_algorithms refused: {}".format(exc))
    else:                                          # pragma: no cover
        rec["warnings"].append("torch not importable; torch seeds skipped")

    # Always true, always stated.
    rec["not_enforceable"].append(
        "LLM sampling: Ollama runs out-of-process; see reproducibility/llm_log.py")
    rec["not_enforceable"].append(
        "float reduction order across devices (CPU/MPS/CUDA) is not identical")

    return rec


def worker_init_fn(worker_id: int) -> None:
    """DataLoader `worker_init_fn`. Each worker gets a DISTINCT but DERIVED seed.

    Without this, every worker inherits the parent's NumPy state and can emit
    IDENTICAL random draws — the classic silent DataLoader bug. `initial_seed()`
    is already offset per worker by torch, so deriving from it gives workers that
    differ from each other but are reproducible across runs.
    """
    if torch is None:                              # pragma: no cover
        return
    base = torch.initial_seed() % (2 ** 32)
    random.seed(base + worker_id)
    if np is not None:
        np.random.seed((base + worker_id) % (2 ** 32))


def make_generator(seed: int) -> Optional["torch.Generator"]:
    """A seeded `torch.Generator` for DataLoader(generator=...).

    Controls SHUFFLE ORDER, which `torch.manual_seed` alone does not pin once
    workers are involved. Returns None if torch is unavailable so callers can
    pass it through unconditionally.
    """
    if torch is None:                              # pragma: no cover
        return None
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def describe() -> Dict[str, Any]:
    """Current RNG/device state, for the manifest. Read-only; seeds nothing."""
    out: Dict[str, Any] = {
        "pythonhashseed_env": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if torch is not None:
        out["torch_initial_seed"] = int(torch.initial_seed())
        out["cuda_available"] = bool(torch.cuda.is_available())
        mps = getattr(torch.backends, "mps", None)
        out["mps_available"] = bool(mps is not None and mps.is_available())
        cudnn = getattr(torch.backends, "cudnn", None)
        if cudnn is not None:
            out["cudnn_deterministic"] = bool(getattr(cudnn, "deterministic", False))
            out["cudnn_benchmark"] = bool(getattr(cudnn, "benchmark", True))
        try:
            out["deterministic_algorithms"] = bool(
                torch.are_deterministic_algorithms_enabled())
        except Exception:                          # pragma: no cover
            out["deterministic_algorithms"] = None
    return out
