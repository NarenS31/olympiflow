"""STAGE 0b — the learned semantic graph, read straight out of the checkpoint.

    python -m xtraffic.scripts.analyze_learned_graph

No GNN solves, no explainer, no LLM. `A_sem` is a stored parameter, so this is a
static read plus arithmetic: it runs in seconds and is exactly reproducible.

WHAT THE OBJECT IS
    XTrafficSTGNN blends two graphs into its first support (stgnn.py:261-268):

        A_final = sigmoid(alpha) * A_phys + (1 - sigmoid(alpha)) * A_sem
        A_sem   = softmax(relu(E @ E.T), dim=1),   E = sem_embed [N, embed_dim]

    The (1 - alpha) component is A_sem, and E is initialised at
    `torch.randn(N, embed_dim) * 0.01` (stgnn.py:239) -- RANDOM, **not** seeded
    from the kernel adjacency. So the learned graph is not a delta from the given
    one and the difference between them is not "the object": A_sem is its own
    graph, learned from scratch, and comparing it to the adjacency is the same
    kind of comparison Stage 1 makes for the explainer.

DIRECTION CONVENTION
    `GraphConv._nconv` is einsum("bcnt,nm->bcmt"), so A[n, m] weights source n
    into target m. A_sem is therefore indexed [source, target], and W_learned --
    which this module keeps in the Stage-1 [target, source] convention -- is its
    TRANSPOSE. Getting this backwards would produce plausible, wrong numbers, so
    it is asserted in the tests.

Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..evaluation import influence_graph as ig
from ..reproducibility import provenance, run_dir, seeds
from ..utils.io_utils import PKG_ROOT
from .analyze_influence_graph import (K_HOPS, MASS_FRAC, TOP_K, fig_influence_map)

REPO_ROOT = os.path.dirname(PKG_ROOT)
EXPERIMENT = "learned_semantic_graph"
CKPT_DIR = "models/gnn/checkpoints"


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def semantic_graph(state: Dict[str, "object"]) -> Tuple[np.ndarray, np.ndarray, float]:
    """Rebuild A_sem exactly as stgnn._semantic_support does.

    Returns (A_sem [source,target], E [N,d], sigmoid(alpha)).
    """
    import torch

    E = state["sem_embed"].detach().cpu()                       # [N, d]
    alpha = float(torch.sigmoid(state["alpha_logit"].detach().cpu()).item())
    S = torch.softmax(torch.relu(E @ E.t()), dim=1)             # [N, N]
    return S.numpy().astype(np.float64), E.numpy().astype(np.float64), alpha


def adaptive_graph(state: Dict[str, "object"]) -> np.ndarray:
    """The second, purely learned support — reported as a secondary comparison."""
    import torch

    v1 = state["nodevec1"].detach().cpu()
    v2 = state["nodevec2"].detach().cpu()
    return torch.softmax(torch.relu(v1 @ v2.t()), dim=1).numpy().astype(np.float64)


def describe_parameterisation(S: np.ndarray, E: np.ndarray, geo: ig.Geometry
                              ) -> Dict[str, object]:
    """Everything item 1 asks for, MEASURED rather than asserted."""
    pre = E @ E.T                                               # [N,N] similarity
    relu_pre = np.maximum(pre, 0.0)
    off = geo.off
    # softmax(0) contributions: relu zeroes every negative similarity, so all
    # those pairs enter the softmax at exactly exp(0)=1 -- one shared floor value
    # rather than a small weight. That shapes the whole matrix.
    floor_share = float((relu_pre[off] == 0).mean())
    row_sums = S.sum(axis=1)

    # Effective sparsity: softmax makes every entry > 0, so structural sparsity is
    # zero and the honest measure is concentration. Perplexity = exp(entropy) is
    # the effective number of targets each source actually spreads mass over.
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(S * np.log(np.clip(S, 1e-300, None))).sum(axis=1)
    perp = np.exp(ent)
    top8 = np.sort(S, axis=1)[:, -TOP_K:].sum(axis=1)

    return {
        "parameterisation": (
            "A_sem = softmax(relu(E @ E.T), dim=1) with E = sem_embed, "
            "a learned [N, d] parameter"),
        "embed_shape": list(E.shape),
        "embed_dim": int(E.shape[1]),
        "rank_bound_of_pre_softmax": int(min(E.shape)),
        "measured_rank_of_pre_softmax": int(np.linalg.matrix_rank(pre)),
        "initialisation": (
            "torch.randn(N, embed_dim) * 0.01 (stgnn.py:239) -- random, NOT "
            "seeded from the kernel adjacency"),
        "seeded_from_kernel_adjacency": False,
        "pre_softmax_symmetric": bool(np.allclose(pre, pre.T)),
        "post_softmax_symmetric": bool(np.allclose(S, S.T)),
        "directed": not bool(np.allclose(S, S.T)),
        "asymmetry_source": (
            "E @ E.T is symmetric by construction; the row-wise softmax breaks "
            "the symmetry, so all directionality comes from normalisation, not "
            "from a learned asymmetric similarity"),
        "normalisation": "row-softmax over dim=1, i.e. per SOURCE outgoing mass",
        "row_sums_min": round(float(row_sums.min()), 6),
        "row_sums_max": round(float(row_sums.max()), 6),
        "note_on_normalisation_axis": (
            "rows are sources under the einsum('bcnt,nm->bcmt') convention, so "
            "each source's outgoing weights sum to 1 rather than each target's "
            "incoming weights. physical_adj is row-normalised the same way "
            "(stgnn.py:230), so the two supports are consistent."),
        "structural_sparsity": {
            "strictly_positive_entries": int((S > 0).sum()),
            "total_entries": int(S.size),
            "share_positive": round(float((S > 0).mean()), 6),
            "comment": "softmax output: nothing is exactly zero, so structural "
                       "sparsity is 0 by construction",
        },
        "effective_sparsity": {
            "relu_floor_share_of_offdiag_pairs": round(floor_share, 4),
            "relu_floor_comment": (
                "share of off-diagonal pairs whose pre-softmax similarity is <= 0 "
                "and therefore enters the softmax at the identical value exp(0); "
                "these pairs are not distinguished from one another at all"),
            "perplexity_mean": round(float(perp.mean()), 2),
            "perplexity_median": round(float(np.median(perp)), 2),
            "perplexity_min": round(float(perp.min()), 2),
            "perplexity_max": round(float(perp.max()), 2),
            "perplexity_uniform_reference": int(S.shape[0]),
            "top8_row_mass_mean": round(float(top8.mean()), 4),
            "top8_row_mass_median": round(float(np.median(top8)), 4),
            "uniform_top8_mass_reference": round(TOP_K / S.shape[0], 4),
        },
    }


def density_report(state: Dict[str, "object"], geo: ig.Geometry) -> Dict[str, object]:
    """Is the graph the model ACTUALLY diffuses over dense — structurally, and
    effectively?

    Structural density (share of strictly positive entries) is the weaker claim:
    a softmax makes it 100% trivially. The claim that matters is EFFECTIVE
    density — whether the mass is genuinely spread — measured as row perplexity
    exp(entropy), the effective number of nodes a row distributes over.

    Reported for the MIXED support A_final that the blocks receive, not just for
    A_sem, because A_final is what sets the receptive field.
    """
    import torch

    E = state["sem_embed"].detach().cpu()
    a = torch.sigmoid(state["alpha_logit"].detach().cpu())
    A_sem = torch.softmax(torch.relu(E @ E.t()), dim=1)
    phys = torch.from_numpy(_row_normalised_physical(geo)).float()
    A_final = (a * phys + (1 - a) * A_sem).numpy().astype(np.float64)
    ada = adaptive_graph(state)

    out: Dict[str, object] = {}
    for name, M in (("physical_adj_row_normalised", phys.numpy().astype(np.float64)),
                    ("A_final_mixed_support", A_final),
                    ("A_sem_learned_semantic", A_sem.numpy().astype(np.float64)),
                    ("adaptive_support", ada)):
        pos = M > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -(M * np.log(np.where(M > 0, M, 1.0))).sum(axis=1)
        perp = np.exp(ent)
        out[name] = {
            "strictly_positive_entries": int(pos.sum()),
            "total_entries": int(M.size),
            "structural_density": round(float(pos.mean()), 5),
            "row_perplexity_mean": round(float(perp.mean()), 2),
            "row_perplexity_median": round(float(np.median(perp)), 2),
            "row_perplexity_as_share_of_uniform": round(
                float(perp.mean() / M.shape[0]), 4),
        }
    out["effectively_dense"] = bool(
        out["A_final_mixed_support"]["structural_density"] > 0.99
        and out["A_final_mixed_support"]["row_perplexity_as_share_of_uniform"] > 0.25)
    out["receptive_field_conclusion"] = (
        "A_final is the support every STBlock diffuses over. It is structurally "
        "dense ({:.0%} of entries positive) and effectively dense (each row "
        "spreads over {:.0f} of {} nodes, {:.0%} of uniform), so the model's "
        "spatial receptive field is the FULL GRAPH IN ONE HOP. gcn_order x "
        "n_blocks bounds diffusion over the PHYSICAL component only. Any stratum "
        "defined by hop distance in the kernel adjacency is therefore a "
        "geometric description, not an architectural limit, and influence "
        "falling outside it is not surprising.".format(
            out["A_final_mixed_support"]["structural_density"],
            out["A_final_mixed_support"]["row_perplexity_mean"], geo.n,
            out["A_final_mixed_support"]["row_perplexity_as_share_of_uniform"]))
    return out


def _row_normalised_physical(geo: ig.Geometry) -> np.ndarray:
    """Reproduce stgnn.py:228-231 — self-loops added, then row-normalised."""
    A = geo.A.astype(np.float64).copy()
    A = A + np.eye(A.shape[0])
    return A / np.clip(A.sum(axis=1, keepdims=True), 1e-6, None)


def road_alignment(S: np.ndarray, geo: ig.Geometry) -> Dict[str, object]:
    """How much of the off-diagonal mass lands on adjacency edges, vs chance."""
    off, adj = geo.off, geo.adj
    share = float(S[adj].sum() / S[off].sum())
    dens = float(adj.sum() / off.sum())
    return {"mass_on_adjacency_edges": round(share, 4),
            "adjacency_share_of_pairs": round(dens, 5),
            "lift_over_chance": round(share / dens, 3)}


# ---------------------------------------------------------------------------
# Stage-2 pipeline applied to a static W
# ---------------------------------------------------------------------------
def analyse_static_W(W: np.ndarray, geo: ig.Geometry, label: str,
                     seed: int = 42) -> Dict[str, object]:
    """Same treatment Stage 1 gives the explainer's W, on a matrix that does not
    vary by window. Split-half is not applicable and is replaced upstream by the
    cross-checkpoint Jaccard."""
    tg = list(range(geo.n))
    e_top = ig.edges_top_k(W, tg, k=TOP_K)
    e_mass = ig.edges_cumulative_mass(W, tg, frac=MASS_FRAC)
    b_uni = ig.baseline_uniform(geo, tg, k=TOP_K, seed=seed)
    b_deg = ig.baseline_degree_matched(geo, tg, k=TOP_K, seed=seed)
    b_near, near_info = ig.baseline_nearest_road(geo, tg, k=TOP_K)

    by_hop = ig.stratify_by_hop(e_top, geo, K_HOPS)
    bk = ig.beyond_k_set(e_top, geo, K_HOPS)
    sm = ig.self_mass(W, tg)
    # sources needed to cover MASS_FRAC, per target -- the flatness measure
    need = []
    for t in tg:
        v = W[t].copy()
        v[t] = 0.0
        tot = v.sum()
        if tot <= 0:
            continue
        c = np.cumsum(np.sort(v)[::-1])
        need.append(int(np.searchsorted(c, MASS_FRAC * tot) + 1))

    return {
        "label": label,
        "edges": {
            "top_k": ig.compare_to_adjacency(e_top, geo, tg),
            "cumulative_mass": ig.compare_to_adjacency(e_mass, geo, tg),
        },
        "sources_for_mass_frac": {
            "mass_frac": MASS_FRAC,
            "mean": round(float(np.mean(need)), 1) if need else None,
            "median": int(np.median(need)) if need else None,
            "min": int(np.min(need)) if need else None,
            "max": int(np.max(need)) if need else None,
            "of_available": geo.n - 1,
        },
        "self_mass": {
            "mean": round(float(np.nanmean(sm)), 5),
            "median": round(float(np.nanmedian(sm)), 5),
            "chance_1_over_N": round(1.0 / geo.n, 5),
        },
        "baselines": {
            "uniform_random": ig.compare_to_adjacency(b_uni, geo, tg),
            "degree_matched_random": ig.compare_to_adjacency(b_deg, geo, tg),
            "nearest_by_road_distance": dict(
                ig.compare_to_adjacency(b_near, geo, tg), **near_info),
        },
        "overlap_with_nearest_road": round(ig.jaccard(e_top, b_near), 4),
        "hop_strata": [{"bucket": b, "n_effective_edges": len(es),
                        "share_of_effective": round(len(es) / len(e_top), 4)}
                       for b, es in by_hop.items()],
        "beyond_k": {
            "k_hops": K_HOPS, "n_edges": len(bk),
            "share_of_effective": round(len(bk) / len(e_top), 4) if e_top else None,
            "n_unreachable": int(sum(1 for (s, t) in bk
                                     if geo.hops[s, t] == ig.Geometry.UNREACHABLE)),
        },
        "_edges_top_k": e_top,
    }


# ---------------------------------------------------------------------------
def metr_la_checkpoints() -> List[str]:
    out = []
    for p in sorted(glob.glob(os.path.join(PKG_ROOT, CKPT_DIR, "*.pt"))):
        import torch
        ck = torch.load(p, map_location="cpu", weights_only=False)
        if ck.get("n_nodes") == 207:
            out.append(p)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    seed_rec = seeds.set_all_seeds(args.seed)
    geo = ig.load_geometry("metr_la")
    ckpts = metr_la_checkpoints()
    print("METR-LA checkpoints (N=207): {}".format(
        [os.path.basename(c) for c in ckpts]))

    cfg = {"experiment": EXPERIMENT, "checkpoints": [os.path.basename(c) for c in ckpts],
           "top_k": TOP_K, "mass_frac": MASS_FRAC, "k_hops": K_HOPS, "seed": args.seed}
    run = run_dir.RunDir.create(
        REPO_ROOT, EXPERIMENT, cfg,
        artifacts={os.path.basename(c): c for c in ckpts},
        notes=("Learned semantic graph A_sem read straight from the checkpoints. "
               "No GNN solves, no explainer, no LLM."))
    run.write_json("seeds.json", seed_rec)
    run.start()

    per_ckpt: Dict[str, Dict] = {}
    W_by_ckpt: Dict[str, np.ndarray] = {}
    edges_by_ckpt: Dict[str, set] = {}

    for path in ckpts:
        name = os.path.basename(path)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        st = ck["model_state"]
        S, E, alpha = semantic_graph(st)                  # S is [source, target]
        W = S.T.copy()                                    # -> [target, source]
        W_by_ckpt[name] = W
        res = analyse_static_W(W, geo, name, args.seed)
        edges_by_ckpt[name] = res.pop("_edges_top_k")
        res["checkpoint"] = {
            "file": name, "sha256": provenance.sha256_file(path),
            "epoch": ck.get("epoch"), "val_mae": ck.get("val_mae"),
            "run_name": (ck.get("config") or {}).get("run_name"),
            "use_sidecars": (ck.get("config") or {}).get("use_sidecars"),
        }
        res["alpha"] = {
            "sigmoid_alpha": round(alpha, 4),
            "weight_on_given_adjacency": round(alpha, 4),
            "weight_on_learned_semantic": round(1 - alpha, 4),
        }
        res["parameterisation"] = describe_parameterisation(S, E, geo)
        res["density"] = density_report(st, geo)
        res["road_alignment_semantic"] = road_alignment(S, geo)
        res["road_alignment_adaptive"] = road_alignment(adaptive_graph(st), geo)
        res["beyond_k_stratum"] = {
            "label": ig.beyond_k_label(geo, K_HOPS),
            "base_rate_over_all_pairs": round(
                ig.beyond_k_base_rate(geo, list(range(geo.n)), K_HOPS), 4),
            "note": ("share of ordered pairs already in the stratum; an edge set "
                     "with no spatial preference would show this rate"),
        }
        per_ckpt[name] = res
        print("  {:38s} alpha={:.4f}  precision={:.4f}  lift={:.2f}x".format(
            name, alpha, res["edges"]["top_k"]["precision"],
            res["road_alignment_semantic"]["lift_over_chance"]))

    # ------------------------------------------------- cross-checkpoint stability
    names = list(per_ckpt.keys())
    stab = {"pairwise_topk_jaccard": [], "note": (
        "Replaces split-half: A_sem is a stored parameter and does not vary by "
        "window, so the meaningful stability question is whether different "
        "training states agree. Note these checkpoints are NOT all the same run "
        "-- metr_la_fusion_best.pt is a separate fusion training run, so a "
        "comparison involving it measures run-to-run variation, not epoch drift.")}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ja = ig.jaccard(edges_by_ckpt[a], edges_by_ckpt[b])
            # per-target rank agreement as well as set agreement
            sp = []
            for t in range(geo.n):
                x, y = W_by_ckpt[a][t], W_by_ckpt[b][t]
                rx, ry = np.argsort(np.argsort(-x)), np.argsort(np.argsort(-y))
                sp.append(float(np.corrcoef(rx, ry)[0, 1]))
            stab["pairwise_topk_jaccard"].append({
                "a": a, "b": b, "topk_jaccard": round(ja, 4),
                "spearman_mean": round(float(np.mean(sp)), 4),
                "spearman_median": round(float(np.median(sp)), 4)})
            print("  J({} , {}) = {:.4f}  spearman {:.4f}".format(
                a, b, ja, float(np.mean(sp))))
    rnd = ig.jaccard(ig.baseline_uniform(geo, list(range(geo.n)), TOP_K, args.seed + 1),
                     ig.baseline_uniform(geo, list(range(geo.n)), TOP_K, args.seed + 2))
    stab["two_random_draws_reference"] = round(rnd, 4)

    alpha_traj = [{"checkpoint": n,
                   "epoch": per_ckpt[n]["checkpoint"]["epoch"],
                   "val_mae": per_ckpt[n]["checkpoint"]["val_mae"],
                   "run_name": per_ckpt[n]["checkpoint"]["run_name"],
                   "sigmoid_alpha": per_ckpt[n]["alpha"]["sigmoid_alpha"]}
                  for n in names]

    res_all = {
        "experiment": EXPERIMENT,
        "geometry": {"n_nodes": geo.n, "n_adjacency_edges": int(geo.adj.sum()),
                     "adjacency_density": round(float(geo.adj.sum() / geo.off.sum()), 5),
                     "cutoff_m": round(geo.cutoff_m, 1), "sigma_m": round(geo.sigma_m, 1)},
        "settings": {"top_k": TOP_K, "mass_frac": MASS_FRAC, "k_hops": K_HOPS},
        "per_checkpoint": per_ckpt,
        "cross_checkpoint_stability": stab,
        "alpha_trajectory": alpha_traj,
        "checkpoints_excluded": {
            "reason": ("different N, so sem_embed is re-initialised at the target "
                       "size and its semantic graph is random init, not learned "
                       "on that city"),
            "files": ["pems_bay_zero_shot.pt (N=325)", "chicago_zero_shot.pt (N=1020)",
                      "chicago_fine_tuned.pt (N=1020)", "power_grid_best.pt (N=14)",
                      "power_grid_smoke.pt (N=14)"]},
    }
    run.write_json("learned_graph.json", res_all)

    # ------------------------------------------------------------------ outputs
    out_dir = os.path.join(PKG_ROOT, run_dir.PROCESSED, run.run_id)
    os.makedirs(out_dir, exist_ok=True)

    primary = "metr_la_best.pt"
    rows = ig.edge_rows(edges_by_ckpt[primary], W_by_ckpt[primary], geo,
                        "learned_semantic", K_HOPS)
    edf = pd.DataFrame(rows)
    edf.to_csv(os.path.join(out_dir, "learned_edges.csv"), index=False)
    edf[~edf.in_adjacency].to_csv(
        os.path.join(out_dir, "learned_off_adjacency_edges.csv"), index=False)
    pd.DataFrame([dict(checkpoint=n, **h)
                  for n in names for h in per_ckpt[n]["hop_strata"]]).to_csv(
        os.path.join(out_dir, "learned_hop_strata.csv"), index=False)
    pd.DataFrame(alpha_traj).to_csv(os.path.join(out_dir, "alpha_trajectory.csv"),
                                    index=False)
    base_rows = []
    for n in names:
        r = per_ckpt[n]
        base_rows.append(dict(checkpoint=n, method="learned_top_k", **r["edges"]["top_k"]))
        for bn in ("uniform_random", "degree_matched_random", "nearest_by_road_distance"):
            base_rows.append(dict(checkpoint=n, method=bn,
                                  **{k: v for k, v in r["baselines"][bn].items()
                                     if not isinstance(v, str)}))
    pd.DataFrame(base_rows).to_csv(os.path.join(out_dir, "learned_baselines.csv"),
                                   index=False)

    fig_influence_map({n.replace(".pt", ""): edges_by_ckpt[n] for n in names},
                      geo, os.path.join(out_dir, "fig_learned_graph_map.pdf"))

    with open(os.path.join(out_dir, "learned_graph.json"), "w", encoding="utf-8") as fh:
        json.dump(res_all, fh, indent=2, default=str)

    from .report_learned_graph import render
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(render(res_all, geo))

    run.complete(n_checkpoints=len(names),
                 primary_precision=per_ckpt[primary]["edges"]["top_k"]["precision"])
    print("\nwrote -> {}".format(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
