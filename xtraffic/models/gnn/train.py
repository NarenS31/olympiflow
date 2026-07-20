"""Phase 2 trainer for the XTraffic ST-GNN.

Run:  python -m xtraffic.models.gnn.train --config configs/train_metr_la.yaml
      python -m xtraffic.models.gnn.train --config ... --smoke   # 1 epoch, tiny subset

What it does, in order:
  1. seed everything (reproducibility, CLAUDE.md hard constraint)
  2. load processed tensors + adjacency + scaler (loaders.py)
  3. build XTrafficSTGNN from the YAML
  4. Adam + gradient clipping, masked-MAE loss (masking explained in utils/metrics)
  5. per-epoch: train, validate, log MAE/RMSE/MAPE at 15/30/60 min to CSV,
     log the learned alpha (MOD 1) and modality gates (MOD 3)
  6. early-stopping on val MAE; keep the best checkpoint
  7. save loss-curve PNG to evaluation/results/

Everything configurable via YAML; nothing hard-coded here.
Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from typing import Dict

import numpy as np
import torch
import yaml

from ...utils.io_utils import PKG_ROOT, units_for_dataset
from ...utils.metrics import masked_mae_loss, masked_metrics, per_horizon_metrics
from .loaders import (build_modality_dict, load_adjacency, load_scaler,
                      make_fusion_loaders)
from .stgnn import XTrafficSTGNN


# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Pin every RNG so a rerun reproduces the same numbers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict:
    # Allow a path relative to the package root for convenience.
    if not os.path.isabs(path):
        cand = os.path.join(PKG_ROOT, path)
        path = cand if os.path.exists(cand) else path
    with open(path) as f:
        return yaml.safe_load(f)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon MPS speeds up the laptop case; falls back to CPU otherwise.
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
def run_epoch(model, loader, scaler, cfg, device, optimizer=None, layout=None):
    """One pass over `loader`. Trains if optimizer is given, else evaluates.

    Returns (mean_loss, aggregated per-horizon metrics dict). We accumulate the
    full-horizon predictions to compute masked metrics once at the end (cheaper
    and less noisy than averaging per-batch metrics)."""
    is_train = optimizer is not None
    model.train(is_train)
    traffic_channels = cfg["traffic_channels"]
    modality_names = list(cfg["modalities"].keys())

    total_loss, n_batches = 0.0, 0
    preds, trues = [], []
    for X, Y, M in loader:                                  # M:[B,12,N,Csum] sidecars
        X, Y, M = X.to(device), Y.to(device), M.to(device)  # X:[B,12,N,2] Y:[B,12,N]
        mods = build_modality_dict(X, traffic_channels, modality_names, M, layout)
        with torch.set_grad_enabled(is_train):
            out = model(mods)                              # [B, 12, N] z-scored
            loss = masked_mae_loss(out, Y, scaler)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg["train"]["grad_clip"])
                optimizer.step()
        total_loss += float(loss.detach())
        n_batches += 1
        preds.append(out.detach().cpu())
        trues.append(Y.detach().cpu())

    preds = torch.cat(preds, 0)                            # [S, 12, N]
    trues = torch.cat(trues, 0)
    horizon = per_horizon_metrics(preds, trues, scaler,
                                  tuple(cfg["horizons_steps"]))
    overall = masked_metrics(preds, trues, scaler)         # avg over all 12 steps
    return total_loss / max(n_batches, 1), overall, horizon


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch on a tiny subset — the Phase 2 shape-sanity gate.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override config epoch count (for quick gate checks).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = pick_device()
    dataset = cfg["dataset"]
    # Phase 19: metrics are unit-agnostic but the labels were hard-coded "mph".
    # `units` drives every printed/plotted label; defaults to "mph" so the traffic
    # runs print exactly what they always did. `prec` widens the decimals for
    # per-unit voltage, where MAE ~0.005 would otherwise round to nothing.
    units = units_for_dataset(dataset)
    prec = 3 if units == "mph" else 5
    print(f"[train] dataset={dataset} device={device} units={units} smoke={args.smoke}")

    # --- data ---
    scaler = load_scaler(dataset)
    adj = load_adjacency(dataset)                          # [N, N]
    n_nodes = adj.shape[0]
    # Fusion is OPT-IN so rerunning the Phase-2 config stays traffic-only even
    # after sidecars are built (CLAUDE.md: never silently change earlier work).
    use_sidecars = bool(cfg.get("use_sidecars", False))
    # sidecar_modalities lets an ablation load only SOME feeds (leave-one-out): the
    # model still declares every encoder (from cfg.modalities) but a dropped feed is
    # passed None -> its MOD-3 gate zeroes it. Default = all declared modalities.
    if use_sidecars:
        sidecar_mods = cfg.get("sidecar_modalities", list(cfg["modalities"].keys()))
    else:
        sidecar_mods = ["traffic"]
    loaders, layout = make_fusion_loaders(
        dataset, cfg["train"]["batch_size"], sidecar_mods,
        cfg["train"].get("num_workers", 0))

    # --- model ---
    m = cfg["model"]
    model = XTrafficSTGNN(
        num_nodes=n_nodes, physical_adj=adj, modality_dims=cfg["modalities"],
        residual_channels=m["residual_channels"],
        dilation_channels=m["dilation_channels"],
        skip_channels=m["skip_channels"], end_channels=m["end_channels"],
        n_blocks=m["n_blocks"], embed_dim=m["embed_dim"],
        gcn_order=m["gcn_order"], dropout=m["dropout"], out_len=m["out_len"],
        # Phase 7 ablation switches (default True = full model).
        use_semantic=m.get("use_semantic", True),
        use_multiscale=m.get("use_multiscale", True),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] XTrafficSTGNN: {n_params:,} parameters, {n_nodes} nodes")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                                 weight_decay=cfg["train"]["weight_decay"])

    # --- output paths ---
    ckpt_dir = os.path.join(PKG_ROOT, cfg["checkpoint_dir"])
    res_dir = os.path.join(PKG_ROOT, cfg["results_dir"])
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    # run_name lets fusion vs traffic-only checkpoints coexist for the comparison
    # table (defaults to the dataset name = the Phase-2 path, unchanged).
    run_name = cfg.get("run_name", dataset)
    # A --smoke run trains 1 epoch on 64 samples — a shape sanity check, NOT a real
    # model. It must NEVER overwrite the real best checkpoint that downstream phases
    # load, so smoke writes to a clearly-throwaway path.
    ckpt_suffix = "_smoke" if args.smoke else "_best"
    ckpt_path = os.path.join(ckpt_dir, f"{run_name}{ckpt_suffix}.pt")
    log_path = os.path.join(res_dir, f"train_{run_name}.csv")

    epochs = 1 if args.smoke else (args.epochs or cfg["train"]["epochs"])

    # Smoke test: replace loaders with tiny slices so shapes get exercised fast.
    if args.smoke:
        from torch.utils.data import DataLoader, Subset
        for k in loaders:
            sub = Subset(loaders[k].dataset, range(min(64, len(loaders[k].dataset))))
            loaders[k] = DataLoader(sub, batch_size=cfg["train"]["batch_size"],
                                    shuffle=(k == "train"))

    # --- CSV log header ---
    fieldnames = ["epoch", "train_loss", "val_loss", "val_mae", "val_rmse",
                  "val_mape", "alpha"]
    for h in cfg["horizons_steps"]:
        fieldnames += [f"val_mae_{h*5}min", f"val_rmse_{h*5}min", f"val_mape_{h*5}min"]
    for name in cfg["modalities"]:
        fieldnames.append(f"gate_{name}")
    log_f = open(log_path, "w", newline="")
    writer = csv.DictWriter(log_f, fieldnames=fieldnames)
    writer.writeheader()

    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    # Smallest val-MAE gain that counts as improvement (see the fix note below).
    # Default 1e-4 = the historical hard-coded value, so traffic runs are unchanged.
    min_delta = float(cfg["train"].get("early_stopping_min_delta", 1e-4))
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, _, _ = run_epoch(model, loaders["train"], scaler, cfg, device,
                                  optimizer, layout=layout)
        va_loss, va_overall, va_h = run_epoch(model, loaders["val"], scaler, cfg,
                                               device, optimizer=None, layout=layout)
        dt = time.time() - t0
        alpha = model.current_alpha()
        gates = model.fusion.gate_values()

        row = {"epoch": epoch, "train_loss": round(tr_loss, 4),
               "val_loss": round(va_loss, 4), "val_mae": round(va_overall["mae"], 4),
               "val_rmse": round(va_overall["rmse"], 4),
               "val_mape": round(va_overall["mape"], 4), "alpha": round(alpha, 4)}
        for h in cfg["horizons_steps"]:
            lab = f"{h*5}min"
            row[f"val_mae_{lab}"] = round(va_h[lab]["mae"], 4)
            row[f"val_rmse_{lab}"] = round(va_h[lab]["rmse"], 4)
            row[f"val_mape_{lab}"] = round(va_h[lab]["mape"], 4)
        for name, g in gates.items():
            row[f"gate_{name}"] = round(g, 4)
        writer.writerow(row)
        log_f.flush()
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        print(f"[epoch {epoch:3d}/{epochs}] "
              f"train_loss={tr_loss:.{prec}f}  val_mae={va_overall['mae']:.{prec}f} {units}  "
              f"val_mae@30min={va_h['30min']['mae']:.{prec}f} {units}  "
              f"alpha={alpha:.3f}  ({dt:.1f}s)")

        # --- early stopping + best checkpoint on val MAE ---
        # BUG FIX (Phase 19): this used to be a hard-coded `best_val - 1e-4`, the
        # minimum val-MAE improvement that counts as progress. That constant
        # silently assumes the mph scale. On METR-LA (MAE ~3.1) 1e-4 is 0.003% of
        # the metric and is exactly the intended "ignore numerical noise" guard.
        # On the power grid (per-unit voltage, MAE ~0.0017) the SAME constant is
        # 5.9% of the metric, so real progress was being scored as no-progress:
        # the first run reached 0.00163 at epoch 21 — better than the recorded
        # best 0.00170 — yet early-stopped and saved the WORSE checkpoint.
        # Now a config knob. Default 1e-4 keeps every traffic run byte-identical.
        if va_overall["mae"] < best_val - min_delta:
            best_val = va_overall["mae"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(), "config": cfg,
                        "scaler": scaler, "n_nodes": n_nodes, "epoch": epoch,
                        "val_mae": best_val}, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["train"]["early_stopping_patience"]:
                print(f"[train] early stopping at epoch {epoch} "
                      f"(best val MAE {best_val:.{prec}f} {units} @ epoch {best_epoch})")
                break

    log_f.close()

    # --- loss-curve PNG ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(history["train_loss"], label="train")
        plt.plot(history["val_loss"], label="val")
        plt.xlabel("epoch"); plt.ylabel(f"masked MAE ({units})")
        plt.title(f"XTraffic ST-GNN training — {dataset}")
        plt.legend(); plt.tight_layout()
        png = os.path.join(res_dir, f"loss_curve_{run_name}.png")
        plt.savefig(png, dpi=120)
        print(f"[train] saved loss curve -> {png}")
    except Exception as e:
        print(f"[train] (loss-curve plot skipped: {e})")

    print(f"[train] done. best val MAE {best_val:.{prec}f} {units} @ epoch {best_epoch}. "
          f"checkpoint -> {ckpt_path}")

    # Write a tiny JSON summary next to the CSV (CLAUDE.md: JSON + CSV).
    with open(os.path.join(res_dir, f"train_{run_name}_summary.json"), "w") as f:
        json.dump({"dataset": dataset, "run_name": run_name, "best_val_mae": best_val,
                   "units": units, "best_epoch": best_epoch, "n_params": n_params,
                   "device": str(device)}, f, indent=2)


if __name__ == "__main__":
    main()
