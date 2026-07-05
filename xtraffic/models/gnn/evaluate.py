"""Evaluate a trained checkpoint on any processed dataset by name.

Run:  python -m xtraffic.models.gnn.evaluate --checkpoint models/gnn/checkpoints/metr_la_best.pt --dataset metr_la
The checkpoint carries its own training config + scaler, so the model is rebuilt
exactly as trained. Evaluating on a DIFFERENT --dataset than trained on is the
cross-city setup Phase 6 leans on (adjacency/scaler come from the eval dataset).
Writes one JSON results row to evaluation/results/. Python 3.9.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from ...utils.io_utils import PKG_ROOT
from ...utils.metrics import masked_metrics, per_horizon_metrics
from .loaders import (build_modality_dict, load_adjacency, load_scaler,
                      make_fusion_loaders)
from .stgnn import XTrafficSTGNN
from .train import pick_device


def evaluate(checkpoint: str, dataset: str) -> dict:
    if not os.path.isabs(checkpoint):
        checkpoint = os.path.join(PKG_ROOT, checkpoint)
    device = pick_device()
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ck["config"]
    m = cfg["model"]

    # Adjacency + scaler come from the EVAL dataset (may differ from training city).
    adj = load_adjacency(dataset)
    scaler = load_scaler(dataset)
    n_nodes = adj.shape[0]

    model = XTrafficSTGNN(
        num_nodes=n_nodes, physical_adj=adj, modality_dims=cfg["modalities"],
        residual_channels=m["residual_channels"],
        dilation_channels=m["dilation_channels"],
        skip_channels=m["skip_channels"], end_channels=m["end_channels"],
        n_blocks=m["n_blocks"], embed_dim=m["embed_dim"],
        gcn_order=m["gcn_order"], dropout=m["dropout"], out_len=m["out_len"],
        # Must match training so an ablated checkpoint's conv shapes line up.
        use_semantic=m.get("use_semantic", True),
        use_multiscale=m.get("use_multiscale", True),
    ).to(device)

    # strict=False so cross-city eval (different N -> different node embeddings)
    # still loads the shared conv weights; node-count-dependent params reinit.
    incompat = model.load_state_dict(ck["model_state"], strict=(n_nodes == ck["n_nodes"]))
    if n_nodes != ck["n_nodes"]:
        print(f"[eval] cross-size load: trained N={ck['n_nodes']} eval N={n_nodes}. "
              f"Node-specific params reinitialised.")
    model.eval()

    traffic_channels = cfg["traffic_channels"]
    modality_names = list(cfg["modalities"].keys())
    # Fusion-aware: a checkpoint trained with use_sidecars must be evaluated WITH its
    # sidecars, else we'd zero the very modalities it learned on. Traffic-only
    # checkpoints (no use_sidecars) evaluate exactly as before.
    use_sidecars = bool(cfg.get("use_sidecars", False))
    # Honor sidecar_modalities so a leave-one-out ablation is EVALUATED with the same
    # feeds it was trained on (else e.g. a no-weather model would see weather at test).
    if use_sidecars:
        sidecar_mods = cfg.get("sidecar_modalities", modality_names)
    else:
        sidecar_mods = ["traffic"]
    loaders, layout = make_fusion_loaders(dataset, cfg["train"]["batch_size"], sidecar_mods)

    preds, trues = [], []
    with torch.no_grad():
        for X, Y, M in loaders["test"]:
            X, Y, M = X.to(device), Y.to(device), M.to(device)
            mods = build_modality_dict(X, traffic_channels, modality_names, M, layout)
            preds.append(model(mods).cpu())
            trues.append(Y.cpu())
    preds, trues = torch.cat(preds, 0), torch.cat(trues, 0)

    overall = masked_metrics(preds, trues, scaler)
    horizon = per_horizon_metrics(preds, trues, scaler, tuple(cfg["horizons_steps"]))
    result = {"model": "XTrafficSTGNN", "checkpoint": os.path.basename(checkpoint),
              "trained_on": cfg["dataset"], "evaluated_on": dataset,
              "run_name": cfg.get("run_name", cfg["dataset"]),
              "use_sidecars": use_sidecars, "overall": overall, "per_horizon": horizon}

    res_dir = os.path.join(PKG_ROOT, cfg["results_dir"])
    os.makedirs(res_dir, exist_ok=True)
    tag = f"eval_{cfg.get('run_name', cfg['dataset'])}_on_{dataset}"
    with open(os.path.join(res_dir, f"{tag}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[eval] {tag}: MAE={overall['mae']:.3f}  RMSE={overall['rmse']:.3f}  "
          f"MAPE={overall['mape']:.2f}%")
    for h, lab in [(3, "15min"), (6, "30min"), (12, "60min")]:
        if lab in horizon:
            print(f"       {lab}: MAE={horizon[lab]['mae']:.3f}  "
                  f"RMSE={horizon[lab]['rmse']:.3f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    evaluate(args.checkpoint, args.dataset)


if __name__ == "__main__":
    main()
