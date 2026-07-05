"""Phase 7 — the ablation harness. One command, every configuration.

Run:
    python -m xtraffic.evaluation.ablations              # full run (Colab-scale)
    python -m xtraffic.evaluation.ablations --smoke      # 1-epoch structural check
    python -m xtraffic.evaluation.ablations --model-only # skip the LLM pipeline part
    python -m xtraffic.evaluation.ablations --pipeline-only

Two blocks, driven entirely by configs/ablations.yaml:

MODEL ablations (prediction quality on METR-LA): full model, then knock out each
paper contribution in turn (-semantic MOD1, -multiscale MOD2, -fusion MOD3, and a
leave-one-out for each modality). Every variant is trained from scratch under N
seeds; we report mean +/- std of MAE/RMSE/MAPE at 15/30/60 min.

PIPELINE ablations (faithfulness): reruns the Phase-5 study under conditions A/B/C
and, for the LLM-model ablation, against each configured local model. Reads back
each run's faithfulness_summary JSON and tabulates F1 + hallucination.

Outputs (evaluation/results/ablations/): master CSV + a LaTeX booktabs table per
block, ready to paste into the paper. Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
from typing import Dict, List, Optional

import yaml

from ..models.gnn.evaluate import evaluate
from ..utils.io_utils import PKG_ROOT

HORIZONS = [("15min", 3), ("30min", 6), ("60min", 12)]
_TMP_CFG_DIR = os.path.join(PKG_ROOT, "configs", "_ablations")   # temp variant configs


# ---------------------------------------------------------------------------
def _load_yaml(path: str) -> Dict:
    if not os.path.isabs(path):
        path = os.path.join(PKG_ROOT, path)
    with open(path) as f:
        return yaml.safe_load(f)


def _mean_std(xs: List[float]):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / n
    return mu, math.sqrt(var)


def _apply_overrides(base: Dict, variant: Dict, seed: int, run_name: str) -> Dict:
    """Deep-copy the base config and apply this variant's overrides."""
    cfg = copy.deepcopy(base)
    cfg["seed"] = seed
    cfg["run_name"] = run_name
    if "model" in variant:
        cfg.setdefault("model", {}).update(variant["model"])
    if "use_sidecars" in variant:
        cfg["use_sidecars"] = variant["use_sidecars"]
    if "sidecar_modalities" in variant:
        cfg["sidecar_modalities"] = variant["sidecar_modalities"]
    return cfg


def _write_tmp_cfg(cfg: Dict, run_name: str) -> str:
    os.makedirs(_TMP_CFG_DIR, exist_ok=True)
    path = os.path.join(_TMP_CFG_DIR, f"{run_name}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    # Return a path relative to PKG_ROOT (train.load_config resolves it).
    return os.path.relpath(path, PKG_ROOT)


# ---------------------------------------------------------------------------
def run_model_ablations(abl: Dict, base_cfg: Dict, seeds: List[int],
                        epochs: int, smoke: bool, dataset: str) -> List[Dict]:
    """Train every model-ablation variant x seed, evaluate, aggregate mean+/-std."""
    rows: List[Dict] = []
    for variant in abl["model_ablations"]:
        name = variant["name"]
        per_seed: List[Dict] = []          # each = per_horizon dict from evaluate()
        for seed in seeds:
            run_name = f"abl_{name}_s{seed}"
            cfg = _apply_overrides(base_cfg, variant, seed, run_name)
            cfg_path = _write_tmp_cfg(cfg, run_name)
            cmd = [sys.executable, "-m", "xtraffic.models.gnn.train",
                   "--config", cfg_path]
            cmd += ["--smoke"] if smoke else ["--epochs", str(epochs)]
            print(f"\n[ablation:model] {name} seed={seed} -> training")
            subprocess.run(cmd, check=True, cwd=os.path.dirname(PKG_ROOT))
            suffix = "_smoke" if smoke else "_best"
            # Path is relative to PKG_ROOT (== .../xtraffic); evaluate() resolves it.
            ckpt = os.path.join("models", "gnn", "checkpoints", f"{run_name}{suffix}.pt")
            res = evaluate(ckpt, dataset)
            per_seed.append(res["per_horizon"])

        # Aggregate mean+/-std across seeds for each horizon + metric.
        row = {"variant": name, "n_seeds": len(per_seed)}
        for lab, _step in HORIZONS:
            for metric in ("mae", "rmse", "mape"):
                vals = [s[lab][metric] for s in per_seed]
                mu, sd = _mean_std(vals)
                row[f"{lab}_{metric}_mean"] = round(mu, 3)
                row[f"{lab}_{metric}_std"] = round(sd, 3)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
def run_pipeline_ablations(abl: Dict, smoke: bool) -> List[Dict]:
    """Rerun the faithfulness study per LLM model; read back its summary JSON."""
    p = abl["pipeline_ablations"]
    fdir = os.path.join(PKG_ROOT, "evaluation", "results", "faithfulness")
    rows: List[Dict] = []
    for model in p["llm_models"]:
        tag = model.replace(":", "_").replace(".", "")
        cmd = [sys.executable, "-m", "xtraffic.evaluation.run_faithfulness_study",
               "--checkpoint", p["checkpoint"], "--city", p["city"],
               "--conditions", ",".join(p["conditions"]),
               "--model", model, "--out-tag", tag]
        cmd += ["--limit", "6"] if smoke else ["--per-stratum", str(p["per_stratum"])]
        print(f"\n[ablation:pipeline] LLM={model} -> faithfulness study")
        try:
            subprocess.run(cmd, check=True, cwd=os.path.dirname(PKG_ROOT))
        except subprocess.CalledProcessError as e:
            # A missing Ollama model shouldn't kill the whole harness — record + skip.
            print(f"[ablation:pipeline] {model} FAILED ({e}); skipping. "
                  f"(Is it pulled? `ollama pull {model}`)")
            continue
        summ_path = os.path.join(fdir, f"faithfulness_summary_{tag}.json")
        if not os.path.exists(summ_path):
            print(f"[ablation:pipeline] no summary for {model}; skipping")
            continue
        with open(summ_path) as f:
            summ = json.load(f)
        for agg in summ["overall"]:               # one per condition
            rows.append({
                "llm_model": model,
                "condition": agg["condition"],
                "n": summ["n_scenarios"],
                "cause_f1": round(agg.get("faithfulness_f1_mean", float("nan")), 3),
                "precision": round(agg.get("cause_precision_mean", float("nan")), 3),
                "recall": round(agg.get("cause_recall_mean", float("nan")), 3),
                "hallucination": round(agg.get("hallucination_rate_mean", float("nan")), 3),
            })
    return rows


# ---------------------------------------------------------------------------
def _write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _latex_model_table(rows: List[Dict]) -> str:
    """Booktabs table: variant x (MAE at 15/30/60), mean+/-std."""
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Model & MAE@15 & MAE@30 & MAE@60 \\", r"\midrule"]
    for r in rows:
        name = r["variant"].replace("_", r"\_")
        cells = []
        for lab, _ in HORIZONS:
            cells.append(f"{r[f'{lab}_mae_mean']:.2f}$\\pm${r[f'{lab}_mae_std']:.2f}")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _latex_pipeline_table(rows: List[Dict]) -> str:
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"LLM & Cond & F1 & Halluc. & Prec. \\", r"\midrule"]
    for r in rows:
        model = r["llm_model"].replace("_", r"\_")
        lines.append(f"{model} & {r['condition']} & {r['cause_f1']:.3f} & "
                     f"{r['hallucination']:.3f} & {r['precision']:.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _print_model_table(rows: List[Dict]) -> None:
    print("\n===== MODEL ABLATIONS (METR-LA test, MAE mph, mean+/-std) =====")
    print(f"  {'variant':>14} {'MAE@15':>14} {'MAE@30':>14} {'MAE@60':>14}")
    for r in rows:
        cells = [f"{r[f'{lab}_mae_mean']:.2f}+/-{r[f'{lab}_mae_std']:.2f}"
                 for lab, _ in HORIZONS]
        print(f"  {r['variant']:>14} " + " ".join(f"{c:>14}" for c in cells))


def _print_pipeline_table(rows: List[Dict]) -> None:
    print("\n===== PIPELINE ABLATIONS (faithfulness) =====")
    print(f"  {'LLM':>14} {'cond':>5} {'F1':>7} {'halluc':>8} {'prec':>7}")
    for r in rows:
        print(f"  {r['llm_model']:>14} {r['condition']:>5} {r['cause_f1']:>7} "
              f"{r['hallucination']:>8} {r['precision']:>7}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ablations.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="1-epoch training + 6-scenario faithfulness — harness sanity.")
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs")
    ap.add_argument("--model-only", action="store_true")
    ap.add_argument("--pipeline-only", action="store_true")
    args = ap.parse_args()

    abl = _load_yaml(args.config)
    base_cfg = _load_yaml(abl["base_config"])
    seeds = abl["seeds"] if not args.smoke else abl["seeds"][:1]
    epochs = args.epochs or abl["epochs"]
    dataset = abl["dataset"]

    out_dir = os.path.join(PKG_ROOT, abl["results_subdir"])
    os.makedirs(out_dir, exist_ok=True)

    if not args.pipeline_only:
        model_rows = run_model_ablations(abl, base_cfg, seeds, epochs, args.smoke, dataset)
        _write_csv(os.path.join(out_dir, "model_ablations.csv"), model_rows)
        with open(os.path.join(out_dir, "model_ablations.tex"), "w") as f:
            f.write(_latex_model_table(model_rows))
        _print_model_table(model_rows)

    if not args.model_only:
        pipe_rows = run_pipeline_ablations(abl, args.smoke)
        _write_csv(os.path.join(out_dir, "pipeline_ablations.csv"), pipe_rows)
        with open(os.path.join(out_dir, "pipeline_ablations.tex"), "w") as f:
            f.write(_latex_pipeline_table(pipe_rows))
        _print_pipeline_table(pipe_rows)

    print(f"\n[ablations] done -> {out_dir}")


if __name__ == "__main__":
    main()
