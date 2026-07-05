"""Phase 6 — cross-city generalization experiments (Contribution #1's transfer result).

Run:  python -m xtraffic.evaluation.cross_city --source metr_la --target chicago
      python -m xtraffic.evaluation.cross_city --source metr_la --target pems_bay

Three transfer settings, one table (rows) x MAE/RMSE/MAPE at 15/30/60 min (cols):
  ZERO-SHOT    : the LA-trained model run on the target city AS-IS (no target training)
  FINE-TUNED   : that same model, briefly fine-tuned on 10% of the target's train split
  FROM-SCRATCH : a model trained only on the target (in-domain upper-bound reference)

WHAT ACTUALLY TRANSFERS (the honest, important detail — also in DIFFERENCES.md):
Our ST-GNN has two kinds of parameters:
  * NODE-AGNOSTIC : the temporal convs, graph-conv mixers, modality encoders/gates,
    and readout. These are Conv2d layers over channels/time (kernel width 1 on the
    node axis), so they do NOT depend on how many nodes a city has — they transfer.
  * NODE-SPECIFIC : the semantic embeddings (MOD 1), the adaptive-adjacency node
    vectors, and the physical-adjacency buffer. These are [N, d] / [N, N] and are
    inherently tied to LA's 207 sensors. They CANNOT be copied onto Chicago's graph.

So "transfer" means: keep the node-agnostic weights, swap in the TARGET city's
physical adjacency, and RE-INITIALISE the node-specific embeddings at the target's
node count. For ZERO-SHOT those embeddings start near-uniform (as at init), so the
model leans on the physical graph + transferred temporal/spatial filters. FINE-TUNE
then adapts the fresh embeddings on a little target data. This is the standard
limitation of adaptive-graph models and is exactly the generalization story the
paper discusses.

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..models.gnn.loaders import (build_modality_dict, load_adjacency,
                                   load_scaler, make_fusion_loaders,
                                   modality_layout)
from ..models.gnn.stgnn import XTrafficSTGNN
from ..utils.io_utils import PKG_ROOT, processed_dir
from ..utils.metrics import masked_mae_loss, per_horizon_metrics

# Parameter-name prefixes that are inherently tied to a specific node set and so
# must NOT be copied across cities (they are re-initialised at the target N).
NODE_SPECIFIC = ("sem_embed", "nodevec1", "nodevec2", "physical_adj")


def _is_node_specific(param_name: str) -> bool:
    return any(param_name.startswith(p) or param_name == p for p in NODE_SPECIFIC)


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _dataset_exists(dataset: str) -> bool:
    return os.path.exists(os.path.join(processed_dir(dataset), "test.npz"))


def build_model_for(dataset: str, modality_dims: Dict[str, int],
                    model_cfg: Dict, device: torch.device) -> XTrafficSTGNN:
    """Fresh model sized to `dataset`'s node count + physical adjacency."""
    adj = load_adjacency(dataset)                         # [N,N] target city graph
    m = model_cfg
    model = XTrafficSTGNN(
        num_nodes=adj.shape[0], physical_adj=adj, modality_dims=modality_dims,
        residual_channels=m["residual_channels"], dilation_channels=m["dilation_channels"],
        skip_channels=m["skip_channels"], end_channels=m["end_channels"],
        n_blocks=m["n_blocks"], embed_dim=m["embed_dim"], gcn_order=m["gcn_order"],
        dropout=m["dropout"], out_len=m["out_len"]).to(device)
    return model


def transfer_weights(source_state: Dict, target_model: XTrafficSTGNN) -> Tuple[int, int]:
    """Copy every NODE-AGNOSTIC source weight into target_model; leave node-specific
    params (embeddings, adjacency buffer) at their fresh init. Returns (copied, skipped)."""
    tgt = target_model.state_dict()
    copied = skipped = 0
    for name, val in source_state.items():
        if name not in tgt:
            continue
        if _is_node_specific(name) or tgt[name].shape != val.shape:
            skipped += 1                                 # node-specific or shape mismatch
            continue
        tgt[name] = val.clone()
        copied += 1
    target_model.load_state_dict(tgt)
    return copied, skipped


@torch.no_grad()
def evaluate(model, dataset: str, batch_size: int, modality_names: List[str],
             device: torch.device, horizons: Tuple[int, ...]) -> Dict:
    """Masked per-horizon metrics on `dataset`'s TEST split, in real mph (target scaler)."""
    scaler = load_scaler(dataset)
    loaders, layout = make_fusion_loaders(dataset, batch_size, modality_names)
    model.eval()
    preds, trues = [], []
    for X, Y, M in loaders["test"]:
        X, Y, M = X.to(device), Y.to(device), M.to(device)
        mods = build_modality_dict(X, [0, 1], modality_names, M, layout)
        preds.append(model(mods).cpu()); trues.append(Y.cpu())
    preds = torch.cat(preds, 0); trues = torch.cat(trues, 0)
    return per_horizon_metrics(preds, trues, scaler, horizons)


def fit(model, dataset: str, batch_size: int, modality_names: List[str],
        device: torch.device, epochs: int, lr: float, grad_clip: float,
        train_fraction: float = 1.0) -> None:
    """Small training loop (used for fine-tune and from-scratch). train_fraction<1
    uses only the EARLIEST fraction of the train split (chronological, no leakage)."""
    scaler = load_scaler(dataset)
    loaders, layout = make_fusion_loaders(dataset, batch_size, modality_names)
    train_loader = loaders["train"]
    if train_fraction < 1.0:
        from torch.utils.data import DataLoader, Subset
        n = int(len(train_loader.dataset) * train_fraction)
        sub = Subset(train_loader.dataset, range(n))     # earliest n windows
        train_loader = DataLoader(sub, batch_size=batch_size, shuffle=True)
        print(f"[fit] fine-tune on earliest {n} of {len(loaders['train'].dataset)} train windows")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(1, epochs + 1):
        t0 = time.time(); tot = 0.0; nb = 0
        for X, Y, M in train_loader:
            X, Y, M = X.to(device), Y.to(device), M.to(device)
            mods = build_modality_dict(X, [0, 1], modality_names, M, layout)
            loss = masked_mae_loss(model(mods), Y, scaler)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step(); tot += float(loss.detach()); nb += 1
        print(f"[fit] epoch {ep}/{epochs} train_mae={tot/max(nb,1):.3f} ({time.time()-t0:.0f}s)")


def save_transferred(model, dataset: str, setting: str, cfg: Dict,
                     scaler_ds: str) -> str:
    """Persist a transferred/fine-tuned/from-scratch model in the same checkpoint
    format ExplanationBuilder expects, so the Phase-6 Chicago faithfulness study
    can load it directly (python -m ...run_faithfulness_study --checkpoint ...)."""
    ckpt_dir = os.path.join(PKG_ROOT, "models", "gnn", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{dataset}_{setting}.pt")
    torch.save({"model_state": model.state_dict(), "config": cfg,
                "scaler": load_scaler(scaler_ds), "n_nodes": model.num_nodes,
                "setting": setting}, path)
    return path


def _row(setting: str, metrics: Dict, horizons: Tuple[int, ...]) -> Dict:
    row = {"setting": setting}
    for h in horizons:
        lab = f"{h*5}min"
        row[f"mae_{lab}"] = round(metrics[lab]["mae"], 3)
        row[f"rmse_{lab}"] = round(metrics[lab]["rmse"], 3)
        row[f"mape_{lab}"] = round(metrics[lab]["mape"], 3)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="metr_la", help="city the model was trained on")
    ap.add_argument("--target", default="chicago", help="city to transfer to")
    ap.add_argument("--source-ckpt", default=None,
                    help="checkpoint path (default: models/gnn/checkpoints/<source>_best.pt)")
    ap.add_argument("--finetune-epochs", type=int, default=5)
    ap.add_argument("--scratch-epochs", type=int, default=30)
    ap.add_argument("--finetune-fraction", type=float, default=0.10)
    args = ap.parse_args()

    device = pick_device()
    horizons = (3, 6, 12)

    for ds in (args.source, args.target):
        if not _dataset_exists(ds):
            raise SystemExit(f"[cross_city] {ds} not processed — run its Phase-1 pipeline first")

    ckpt_path = args.source_ckpt or os.path.join(
        PKG_ROOT, "models", "gnn", "checkpoints", f"{args.source}_best.pt")
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"[cross_city] source checkpoint missing: {ckpt_path}\n"
                         f"Train the source model first (train_{args.source}*.yaml).")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    set_seed(cfg.get("seed", 42))
    modality_dims = cfg["modalities"]
    modality_names = list(modality_dims.keys())
    model_cfg = cfg["model"]
    bs = cfg["train"]["batch_size"]
    lr = cfg["train"]["lr"]; clip = cfg["train"]["grad_clip"]

    rows: List[Dict] = []

    # --- ZERO-SHOT: transfer node-agnostic weights, fresh target embeddings, no training.
    print(f"\n=== ZERO-SHOT  {args.source} -> {args.target} ===")
    zs = build_model_for(args.target, modality_dims, model_cfg, device)
    copied, skipped = transfer_weights(ckpt["model_state"], zs)
    print(f"[transfer] copied {copied} node-agnostic tensors, "
          f"re-initialised {skipped} node-specific tensors")
    rows.append(_row("zero_shot", evaluate(zs, args.target, bs, modality_names, device, horizons), horizons))
    save_transferred(zs, args.target, "zero_shot", cfg, args.target)

    # --- FINE-TUNED: same transferred model, brief fine-tune on a little target data.
    print(f"\n=== FINE-TUNED ({int(args.finetune_fraction*100)}% target) ===")
    ft = build_model_for(args.target, modality_dims, model_cfg, device)
    transfer_weights(ckpt["model_state"], ft)
    fit(ft, args.target, bs, modality_names, device, args.finetune_epochs, lr, clip,
        train_fraction=args.finetune_fraction)
    rows.append(_row("fine_tuned", evaluate(ft, args.target, bs, modality_names, device, horizons), horizons))
    ft_path = save_transferred(ft, args.target, "fine_tuned", cfg, args.target)
    print(f"[cross_city] fine-tuned checkpoint (use for the {args.target} "
          f"faithfulness study) -> {ft_path}")

    # --- FROM-SCRATCH: in-domain upper-bound reference (fresh model, full target train).
    print(f"\n=== FROM-SCRATCH (target only) ===")
    sc = build_model_for(args.target, modality_dims, model_cfg, device)
    fit(sc, args.target, bs, modality_names, device, args.scratch_epochs, lr, clip)
    rows.append(_row("from_scratch", evaluate(sc, args.target, bs, modality_names, device, horizons), horizons))
    save_transferred(sc, args.target, "from_scratch", cfg, args.target)

    # --- Persist CSV + JSON (CLAUDE.md: both) and print the table. ---
    out_dir = os.path.join(PKG_ROOT, "evaluation", "results", "cross_city")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{args.source}_to_{args.target}"
    fields = list(rows[0].keys())
    with open(os.path.join(out_dir, f"{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
        json.dump({"source": args.source, "target": args.target, "rows": rows}, f, indent=2)

    print(f"\n===== CROSS-CITY TRANSFER: {args.source} -> {args.target} =====")
    print("  " + "  ".join(f"{c:>12}" for c in fields))
    for r in rows:
        print("  " + "  ".join(f"{str(r[c]):>12}" for c in fields))
    print(f"\n[cross_city] saved -> {out_dir}/{tag}.csv")


if __name__ == "__main__":
    main()
