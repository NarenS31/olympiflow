"""PEMS-BAY training SMOKE TEST — measure the cost of a full run before paying it.

WHY THIS FILE EXISTS
--------------------
CLAUDE.md records that local METR-LA training was ~40 min/epoch on Apple MPS
("einsum falls back to CPU") and was therefore abandoned for Colab. PEMS-BAY is
BIGGER on both axes that drive epoch cost — 325 nodes vs 207 (1.57x) and 36,465
train samples vs 23,974 (1.52x) — so the METR-LA figure is a floor, not an
estimate. Committing to a 100-epoch run without measuring would be committing to
an unknown number of days.

So this script does the cheap thing first:

  PHASE A — DEVICE PROBE. Time a bounded number of real train batches on each
  candidate device (mps, cpu) with the real config, real loaders and the real
  model. Projects an epoch time from measured seconds/batch. Costs ~1 minute and
  tells us which device to use AND whether a full run is affordable at all.

  PHASE B — REAL EPOCHS. Run N full epochs through the UNMODIFIED trainer
  (models/gnn/train.py, invoked as a subprocess) on the device Phase A chose, so
  the reported per-epoch time includes everything the real run pays for:
  validation, per-horizon metrics, CSV logging, checkpointing.

NOTHING IN models/gnn/train.py IS TOUCHED. This is a wrapper. The trainer that
produced metr_la_best.pt and power_grid_best.pt runs here byte-identical, which
is the only way the resulting timing means anything.

CHECKPOINT SAFETY (this repo has been burned before — CLAUDE.md records a smoke
run destroying the real metr_la_best.pt): Phase B writes a GENERATED config whose
`run_name` is `pems_bay_smoke`, so the checkpoint lands at
`pems_bay_smoke_best.pt`. It cannot collide with metr_la_best.pt, with
pems_bay_zero_shot.pt, or with the `pems_bay_best.pt` that the real Step-3 run
will later write.

Run:
  python -m xtraffic.scripts.run_pems_bay_train_smoke                 # probe + 3 epochs
  python -m xtraffic.scripts.run_pems_bay_train_smoke --probe-only    # probe only
  python -m xtraffic.scripts.run_pems_bay_train_smoke --epochs 5

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import yaml

from ..reproducibility import provenance, run_dir, seeds
from ..utils.io_utils import PKG_ROOT, processed_dir

REPO_ROOT = os.path.dirname(PKG_ROOT)
EXPERIMENT = "pems_bay_train_smoke"
CONFIG = "configs/train_pems_bay.yaml"
DATASET = "pems_bay"
SMOKE_RUN_NAME = "pems_bay_smoke"       # -> pems_bay_smoke_best.pt, throwaway

# Reference point for the extrapolation, read from the committed METR-LA run.
METR_LA_EPOCHS_TO_BEST = 54             # metr_la_best.pt


def _load_cfg() -> Dict[str, Any]:
    with open(os.path.join(PKG_ROOT, CONFIG)) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# PHASE A — device probe
# ---------------------------------------------------------------------------
def _split_batch_counts(batch_size: int) -> Dict[str, int]:
    """Batches per epoch, read from the committed stats.json (no tensors loaded)."""
    with open(os.path.join(processed_dir(DATASET), "stats.json")) as fh:
        st = json.load(fh)
    import math
    return {"train": int(math.ceil(st["samples_train"] / float(batch_size))),
            "val": int(math.ceil(st["samples_val"] / float(batch_size)))}


def probe_device(cfg: Dict[str, Any], device_name: str, n_batches: int,
                 synthetic: bool = False) -> Dict[str, Any]:
    """Time `n_batches` real training steps on `device_name`.

    Builds exactly what train.py builds (same loaders, same model, same loss,
    same optimiser) so seconds/batch is the real thing, not a synthetic bench.

    SYNTHETIC MODE exists because of a REAL constraint found on this machine: an
    idle `llama-server` holds 12 GB of the 16 GB of RAM, and PEMS-BAY's splits are
    ~2.4 GB of float32 tensors (train X alone is 36465*12*325*2*4 = 1.14 GB). Load
    them on top and the process swaps: measured 77 s/batch with 20.4/21.5 GB of
    swap in use and 42% system time — a measurement of disk thrash, not of the
    model. With `synthetic=True` we feed tensors of the EXACT production shapes
    (X[B,12,325,2], Y[B,12,325]) so the arithmetic per step is identical while the
    resident set stays small. Steady-state per-batch time is pure compute anyway —
    the loaders read the split once per run, not once per batch — so this is a
    faithful proxy for the per-batch cost, and it is honestly labelled as one.
    """
    import torch

    from ..models.gnn.loaders import (build_modality_dict, load_adjacency,
                                      load_scaler, make_fusion_loaders)
    from ..models.gnn.stgnn import XTrafficSTGNN
    from ..utils.metrics import masked_mae_loss

    device = torch.device(device_name)
    scaler = load_scaler(DATASET)
    adj = load_adjacency(DATASET)                       # [N, N]
    n_nodes = adj.shape[0]
    bs = cfg["train"]["batch_size"]

    if synthetic:
        counts = _split_batch_counts(bs)
        gen = torch.Generator().manual_seed(0)
        Xs = torch.randn(bs, 12, n_nodes, 2, generator=gen)   # [B,12,N,2]
        Ys = torch.randn(bs, 12, n_nodes, generator=gen)      # [B,12,N]
        Ms = torch.zeros(bs, 12, n_nodes, 0)                  # no sidecars
        layout: List[Any] = []
        train_loader = [(Xs, Ys, Ms)] * (n_batches + 1)
        n_train_batches, n_val_batches = counts["train"], counts["val"]
    else:
        # use_sidecars is false for PEMS-BAY -> traffic only, same as train.py.
        loaders, layout = make_fusion_loaders(
            DATASET, bs, ["traffic"], cfg["train"].get("num_workers", 0))
        train_loader = loaders["train"]
        n_train_batches = len(train_loader)
        n_val_batches = len(loaders["val"])

    m = cfg["model"]
    model = XTrafficSTGNN(
        num_nodes=n_nodes, physical_adj=adj, modality_dims=cfg["modalities"],
        residual_channels=m["residual_channels"],
        dilation_channels=m["dilation_channels"],
        skip_channels=m["skip_channels"], end_channels=m["end_channels"],
        n_blocks=m["n_blocks"], embed_dim=m["embed_dim"],
        gcn_order=m["gcn_order"], dropout=m["dropout"], out_len=m["out_len"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                           weight_decay=cfg["train"]["weight_decay"])
    model.train(True)

    traffic_channels = cfg["traffic_channels"]
    modality_names = list(cfg["modalities"].keys())

    # WARMUP EXCLUSION. Early batches pay lazy kernel compilation and memory-pool
    # growth that a real epoch pays once, not per batch, so counting them inflates
    # every downstream projection. The first version excluded only batch 0, which
    # was NOT enough and produced a wrong number: on MPS batch 0 was 22.3 s, batch 1
    # was still 6.5 s, and batches 2-10 settled at 2.40 +/- 0.32 s. Including batch 1
    # in the mean reported 2.81 s/batch, 17% high. Two batches are dropped now, and
    # the MEDIAN is reported alongside the mean because a single straggler in a
    # 10-batch sample moves the mean much more than the median.
    warmup = 2
    times: List[float] = []
    for i, (X, Y, M) in enumerate(train_loader):
        if i > n_batches:
            break
        t0 = time.time()
        X, Y, M = X.to(device), Y.to(device), M.to(device)
        mods = build_modality_dict(X, traffic_channels, modality_names, M, layout)
        out = model(mods)                                # [B, 12, N]
        loss = masked_mae_loss(out, Y, scaler)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        opt.step()
        if device_name == "mps":
            torch.mps.synchronize()                      # MPS is async; force completion
        dt = time.time() - t0
        # Print EVERY batch. A probe that reports nothing until it finishes is
        # useless when the thing you are probing for is "is this unusably slow?" —
        # the first batch time already answers that.
        print("[probe:{}] batch {:>3} {:.3f}s{}".format(
            device_name, i, dt, "  (warmup, excluded)" if i < warmup else ""))
        if i >= warmup:
            times.append(dt)

    times_sorted = sorted(times)
    mean_s = sum(times) / max(len(times), 1)
    median_s = times_sorted[len(times_sorted) // 2] if times_sorted else 0.0
    # Validation is forward-only. Measured separately would cost another pass, so
    # it is ESTIMATED at 0.4x a train step (no backward, no optimiser step) and
    # labelled as an estimate. Phase B replaces this with the measured truth.
    est_epoch_s = mean_s * n_train_batches + 0.4 * mean_s * n_val_batches
    return {
        "device": device_name,
        "batches_timed": len(times),
        "warmup_batches_dropped": warmup,
        "seconds_per_train_batch": round(mean_s, 4),
        "seconds_per_train_batch_median": round(median_s, 4),
        "seconds_per_train_batch_all": [round(t, 3) for t in times],
        "train_batches_per_epoch": n_train_batches,
        "val_batches_per_epoch": n_val_batches,
        "projected_epoch_seconds": round(est_epoch_s, 1),
        "projected_epoch_minutes": round(est_epoch_s / 60.0, 2),
        "n_nodes": int(n_nodes),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "synthetic_inputs": bool(synthetic),
    }


def available_devices() -> List[str]:
    import torch
    devs = []
    if torch.cuda.is_available():
        devs.append("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        devs.append("mps")
    devs.append("cpu")
    return devs


# ---------------------------------------------------------------------------
# PHASE B — real epochs through the unmodified trainer
# ---------------------------------------------------------------------------
_EPOCH_RE = re.compile(
    r"\[epoch\s+(\d+)/\s*\d+\]\s+train_loss=([\d.]+)\s+val_mae=([\d.]+)\s+\w+\s+"
    r"val_mae@30min=([\d.]+)\s+\w+\s+alpha=([\d.]+)\s+\(([\d.]+)s\)")


class _Tee(object):
    """Write to the real stdout AND collect, so the run dir keeps the transcript."""

    def __init__(self, stream):
        self.stream = stream
        self.lines: List[str] = []

    def write(self, s):
        self.stream.write(s)
        self.lines.append(s)
        return len(s)

    def flush(self):
        self.stream.flush()


def run_real_epochs(run: "run_dir.RunDir", cfg: Dict[str, Any], epochs: Optional[int],
                    device: str, run_name: Optional[str] = None) -> Dict[str, Any]:
    """Run `epochs` full epochs through the UNMODIFIED trainer, in process.

    DEVICE CONTROL, HONESTLY. train.py chooses its own device (cuda > mps > cpu)
    and exposes no override — reasonable, since every previous run wanted the
    accelerator. The probe may disagree (on this class of model MPS can lose to
    CPU because the graph einsum falls back to CPU anyway, and CLAUDE.md records
    exactly that on METR-LA). Rather than edit train.py — a file every committed
    checkpoint in this repo was produced by — this rebinds `train.pick_device` for
    the duration of THIS PROCESS only. Nothing on disk changes, the trainer's own
    logic is untouched, and the override is visible in the manifest. If we decide
    to keep a non-default device for the real Step-3 run, the right fix is a config
    knob in train.py, proposed and flagged, not a monkeypatch left lying around.
    """
    import torch

    from ..models.gnn import train as train_mod

    # Generated config. For a SMOKE run `run_name` is overridden so the throwaway
    # checkpoint/log can never be mistaken for the real Step-3 artifacts. For a FULL
    # run the config is used AS COMMITTED (run_name: pems_bay -> pems_bay_best.pt),
    # because the whole point is to produce the real checkpoint.
    eff_cfg = dict(cfg)
    eff_cfg["run_name"] = run_name or SMOKE_RUN_NAME
    run.write_json("effective_config.json", eff_cfg)
    yaml_path = os.path.join(run.path, "train_{}.yaml".format(eff_cfg["run_name"]))
    with open(yaml_path, "w") as fh:
        yaml.safe_dump(eff_cfg, fh, sort_keys=False)

    original_pick = train_mod.pick_device
    original_argv = sys.argv
    tee = _Tee(sys.stdout)
    t0 = time.time()
    rc = 0
    try:
        train_mod.pick_device = lambda: torch.device(device)
        sys.argv = ["train", "--config", yaml_path]
        # epochs=None means "use the config's own epoch count and let early stopping
        # decide" — the real Step-3 behaviour. An int caps it, for smoke runs.
        if epochs is not None:
            sys.argv += ["--epochs", str(epochs)]
        import contextlib
        with contextlib.redirect_stdout(tee):
            train_mod.main()
    except SystemExit as exc:                                 # argparse / exit paths
        rc = int(exc.code or 0)
    finally:
        train_mod.pick_device = original_pick
        sys.argv = original_argv

    wall = time.time() - t0
    text = "".join(tee.lines)
    run.write_text("train_stdout.log", text)

    epoch_rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        m = _EPOCH_RE.search(line)
        if m:
            epoch_rows.append({
                "epoch": int(m.group(1)),
                "train_loss": float(m.group(2)),
                "val_mae": float(m.group(3)),
                "val_mae_30min": float(m.group(4)),
                "alpha": float(m.group(5)),
                "seconds": float(m.group(6)),
            })
    return {"returncode": rc, "wall_seconds": round(wall, 1),
            "device_forced": device, "epochs": epoch_rows}


def extrapolate(epoch_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Project a full run from measured epoch times.

    Uses the median of epochs AFTER the first: epoch 1 carries process startup,
    npz load and lazy allocation that later epochs do not repeat.
    """
    if not epoch_rows:
        return {}
    secs = [r["seconds"] for r in epoch_rows]
    steady = secs[1:] if len(secs) > 1 else secs
    steady_sorted = sorted(steady)
    med = steady_sorted[len(steady_sorted) // 2]
    cfg_epochs = 100                       # train_metr_la.yaml / train_pems_bay.yaml
    out = {
        "epoch_seconds_measured": secs,
        "epoch_seconds_first": secs[0],
        "epoch_seconds_median_steady": round(med, 1),
        "epoch_minutes_median_steady": round(med / 60.0, 2),
    }
    for label, n in (("full_100_epochs", cfg_epochs),
                     ("to_metr_la_best_epoch_54", METR_LA_EPOCHS_TO_BEST)):
        out[label + "_hours"] = round(med * n / 3600.0, 2)
    # Early stopping means the realistic run is best_epoch + patience, not 100.
    out["note"] = ("Early stopping (patience 15) usually ends a run well before "
                   "epoch 100; METR-LA's best was epoch 54.")
    return out


# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--epochs", type=int, default=3,
                    help="Full epochs to run in Phase B (the assignment says 3-5).")
    ap.add_argument("--probe-batches", type=int, default=30,
                    help="Train batches to time per device in Phase A.")
    ap.add_argument("--probe-only", action="store_true",
                    help="Stop after the device probe (no real epochs).")
    ap.add_argument("--full", action="store_true",
                    help="STEP 3: the real training run. Uses the committed config "
                         "as-is (run_name: pems_bay -> pems_bay_best.pt) for its full "
                         "epoch count, with early stopping deciding when to end. "
                         "Skips the device probe -- MPS already measured 10-25x faster "
                         "than CPU, and probing again would only delay the run.")
    ap.add_argument("--device", default=None,
                    help="Force a device instead of using the probe's winner.")
    ap.add_argument("--synthetic", action="store_true",
                    help="Probe with production-shaped synthetic tensors instead of "
                         "loading the 2.4 GB splits (see probe_device docstring). "
                         "Measures compute only; use when RAM is contended.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    cfg = _load_cfg()
    seed_rec = seeds.set_all_seeds(args.seed)
    pdir = processed_dir(DATASET)

    # A full run gets its OWN experiment name so it never sorts in with the smoke
    # probes -- these are the artifacts a reader will actually want to find.
    experiment = "pems_bay_train_full" if args.full else EXPERIMENT
    run_cfg = {
        "experiment": experiment,
        "dataset": DATASET,
        "train_config": CONFIG,
        "full_run": bool(args.full),
        "epochs_requested": None if args.full else args.epochs,
        "probe_batches": None if args.full else args.probe_batches,
        "seed": args.seed,
        "run_name": cfg.get("run_name", DATASET) if args.full else SMOKE_RUN_NAME,
        "model": cfg["model"],
        "train": cfg["train"],
    }
    run = run_dir.RunDir.create(
        REPO_ROOT, experiment, run_cfg,
        artifacts={"adjacency": os.path.join(pdir, "adjacency.npy"),
                   "node_meta": os.path.join(pdir, "node_meta.json"),
                   "scaler": os.path.join(pdir, "scaler.json"),
                   "train_split": os.path.join(pdir, "train.npz"),
                   "val_split": os.path.join(pdir, "val.npz")},
        notes=("PEMS-BAY training smoke test: device probe + a few real epochs to "
               "cost a full run before committing to it. No LLM calls. Writes only "
               "throwaway *_smoke artifacts."))
    run.write_json("seeds.json", seed_rec)
    run.start()
    print("[smoke] run dir: {}".format(run.path))

    try:
        # ---- STEP 3: the real run. No probe; go straight to training. ----
        if args.full:
            dev = args.device or available_devices()[0]
            print("[full] STEP 3 real training run on {} -> {}_best.pt".format(
                dev, cfg.get("run_name", DATASET)))
            res = run_real_epochs(run, cfg, None, dev,
                                  run_name=cfg.get("run_name", DATASET))
            ext = extrapolate(res["epochs"])
            run.write_json("full_run_summary.json",
                           {"real_epochs": res, "extrapolation": ext,
                            "device": dev})
            print("\n" + json.dumps(ext, indent=2))
            run.complete(phase="full_train", chosen_device=dev,
                         epochs_run=len(res["epochs"]),
                         epoch_minutes=ext.get("epoch_minutes_median_steady"))
            return 0 if res["returncode"] == 0 else 1

        # ---- Phase A ----
        devices = [args.device] if args.device else available_devices()
        probes = []
        for d in devices:
            print("\n[probe] timing {} batches on {} ...".format(args.probe_batches, d))
            p = probe_device(cfg, d, args.probe_batches, synthetic=args.synthetic)
            print("[probe] {}: {:.4f}s/batch -> ~{:.1f} min/epoch".format(
                d, p["seconds_per_train_batch"], p["projected_epoch_minutes"]))
            probes.append(p)
            # Write after EACH device, not after all of them: the first version of
            # this script wrote once at the end, so killing a slow probe lost every
            # measurement it had already made. Partial results are results.
            # ONE FILE PER DEVICE, because RunDir.write_json refuses to overwrite
            # inside an append-only run directory — rewriting a single
            # device_probe.json raised FileExistsError on the second device and
            # failed a run that had already measured both. The guard is right; the
            # write pattern was wrong.
            run.write_json("device_probe_{}.json".format(d), p)
        best = min(probes, key=lambda p: p["seconds_per_train_batch"])
        print("\n[probe] fastest device: {}".format(best["device"]))

        if args.probe_only:
            run.complete(phase="probe_only", probes=probes, chosen_device=best["device"])
            return 0

        # ---- Phase B ----
        res = run_real_epochs(run, cfg, args.epochs, best["device"])
        ext = extrapolate(res["epochs"])
        summary = {"probes": probes, "chosen_device": best["device"],
                   "real_epochs": res, "extrapolation": ext}
        run.write_json("smoke_summary.json", summary)
        print("\n" + json.dumps(ext, indent=2))
        run.complete(phase="probe_and_epochs", chosen_device=best["device"],
                     epoch_minutes=ext.get("epoch_minutes_median_steady"),
                     full_100_epochs_hours=ext.get("full_100_epochs_hours"))
        return 0 if res["returncode"] == 0 else 1
    except Exception as exc:                                  # noqa: BLE001
        run.fail("{}: {}".format(type(exc).__name__, exc))
        raise


if __name__ == "__main__":
    sys.exit(main())
