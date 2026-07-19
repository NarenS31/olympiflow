#!/usr/bin/env bash
# =============================================================================
# XTraffic — sequential Ollama queue runner
# =============================================================================
# WHY THIS EXISTS
#   Phases 11-16 each need the ONE local Ollama server. Running them in parallel
#   would make them fight over the same model and thrash. This runs them strictly
#   one after another, and — critically — SKIPS any phase whose results already
#   exist on disk, so you can rerun this script safely any number of times.
#
# HOW TO RUN (from the repo root, NOT from inside xtraffic/):
#   nohup bash xtraffic/run_queue.sh > xtraffic/evaluation/results/queue_run.log 2>&1 &
#
# Every phase is:  check_*  ->  DONE (skip)  or  PENDING (run)
# The check functions probe the actual RESULT FILE, not a marker file, so a
# half-finished run is correctly seen as incomplete and gets resumed.
# All six underlying scripts are themselves RESUMABLE (they cache per-decision),
# so killing this script mid-phase loses at most the current LLM call.
# -----------------------------------------------------------------------------

set -u  # error on undefined vars. NOT -e: one failing phase must not kill the queue.

# Resolve the repo root from this script's location so the `python3 -m xtraffic.*`
# module paths always work no matter where you launch from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT" || exit 1

RESULTS="xtraffic/evaluation/results"
LOGDIR="$RESULTS/queue_logs"
mkdir -p "$LOGDIR"

PY=python3

# --- bookkeeping for the end-of-run summary ---------------------------------
declare -a SUMMARY_NAME SUMMARY_STATUS SUMMARY_DURATION SUMMARY_NOTE

record() {  # record <name> <status> <duration> <note>
  SUMMARY_NAME+=("$1"); SUMMARY_STATUS+=("$2")
  SUMMARY_DURATION+=("$3"); SUMMARY_NOTE+=("$4")
}

hr() { printf '%.0s=' {1..78}; echo; }

now()      { date '+%Y-%m-%d %H:%M:%S'; }
epoch()    { date '+%s'; }
human()    {  # seconds -> "1h 23m 45s"
  local s=$1
  printf '%dh %02dm %02ds' $((s/3600)) $((s%3600/60)) $((s%60))
}

# -----------------------------------------------------------------------------
# run_phase <name> <check_fn> <logfile> <command...>
#   Runs <command> only if <check_fn> returns non-zero (i.e. "not done").
# -----------------------------------------------------------------------------
run_phase() {
  local name="$1"; shift
  local check_fn="$1"; shift
  local logfile="$1"; shift

  hr
  echo "PHASE: $name"
  echo "CHECK: $check_fn"

  local note
  note="$($check_fn 2>&1)"
  local is_done=$?

  if [ $is_done -eq 0 ]; then
    echo "STATUS: ALREADY COMPLETE -> skipping"
    echo "DETAIL: $note"
    record "$name" "SKIPPED" "-" "$note"
    return 0
  fi

  echo "STATUS: PENDING -> running"
  echo "DETAIL: $note"
  echo "CMD:    $*"
  echo "LOG:    $logfile"

  # DRY_RUN=1 shows what WOULD run without touching Ollama — use it to sanity
  # check the skip logic before committing to an overnight run.
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "STATUS: DRY RUN -> not executing"
    record "$name" "WOULD-RUN" "-" "$note"
    return 0
  fi

  local t0 t1 rc
  t0=$(epoch)
  echo "START:  $(now)"

  "$@" >> "$logfile" 2>&1
  rc=$?

  t1=$(epoch)
  echo "END:    $(now)"
  echo "ELAPSED: $(human $((t1 - t0)))"
  echo "EXIT:   $rc"

  # Re-probe: did the run actually produce complete results?
  local after
  after="$($check_fn 2>&1)"
  local now_done=$?

  if [ $rc -ne 0 ]; then
    record "$name" "FAILED(rc=$rc)" "$(human $((t1 - t0)))" "$after"
  elif [ $now_done -eq 0 ]; then
    record "$name" "COMPLETED" "$(human $((t1 - t0)))" "$after"
  else
    record "$name" "PARTIAL" "$(human $((t1 - t0)))" "$after"
  fi
  return 0
}

# =============================================================================
# COMPLETION CHECKS
#   Each returns 0 = already done (skip), 1 = needs running.
#   Each echoes a one-line human-readable reason (shown in the summary).
# =============================================================================

# --- Phase 15b: C_RICH + D_CONTRA faithfulness conditions --------------------
# NOTE: guarded on the *_15b tagged artifacts, NOT the untagged A/B/C ones.
# The untagged files are the committed Phase-5 headline numbers and must not be
# touched by this queue (see the command below, which passes --out-tag 15b).
check_15b() {
  local f="$RESULTS/faithfulness/faithfulness_summary_15b.json"
  [ -f "$f" ] || { echo "missing $f"; return 1; }
  $PY - "$f" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
conds = set(s.get("conditions", []))
n = max((a.get("cause_precision_n") or a.get("n") or 0)
        for a in (s.get("aggregate") or s.get("aggregates") or {}).values()) \
    if isinstance(s.get("aggregate") or s.get("aggregates"), dict) else 0
need = {"C_RICH", "D_CONTRA"}
ok = need.issubset(conds)
print("conditions={} n~{}".format(sorted(conds), n))
sys.exit(0 if ok else 1)
EOF
}

# --- Phase 16: active grounding loop, full 93-scenario run -------------------
check_16() {
  local f="$RESULTS/active_grounding/active_grounding_summary.json"
  [ -f "$f" ] || { echo "missing $f"; return 1; }
  $PY - "$f" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
n = s.get("n_scenarios", 0)
rounds = len(s.get("per_round", []))
print("n_scenarios={} rounds_logged={} frac_reached={}"
      .format(n, rounds, s.get("frac_reached_overall")))
# The stratified population is ~93; anything >= 90 with a full round curve is the real run.
sys.exit(0 if (n >= 90 and rounds >= 2) else 1)
EOF
}

# --- Phase 11: cross-city x cross-model faithfulness ------------------------
# The master table writes one row per (city, model) cell with status PENDING
# until that cell's study actually lands. Done == zero PENDING rows.
check_11() {
  local f="$RESULTS/cross_city_faithfulness/cross_city_faithfulness_master.csv"
  [ -f "$f" ] || { echo "missing $f"; return 1; }
  local pending total
  pending=$(grep -c "PENDING" "$f")
  total=$(($(wc -l < "$f") - 1))
  echo "cells=$total pending=$pending"
  [ "$pending" -eq 0 ]
}

# --- Phase 12: SHAP baseline comparison -------------------------------------
# The committed summary is the n=3 SMOKE run; the paper needs the full 30.
check_12() {
  local f="$RESULTS/shap_comparison/shap_comparison_summary.json"
  [ -f "$f" ] || { echo "missing $f"; return 1; }
  $PY - "$f" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
n = s.get("n_scenarios", 0)
print("n_scenarios={} (smoke was 3, paper needs the full sweep)".format(n))
sys.exit(0 if n >= 25 else 1)
EOF
}

# --- Phase 13: condition-A failure-mode taxonomy ----------------------------
# NEEDS OLLAMA: it regenerates ~7 condition-A advisories to recover the cited
# causes Phase 5 never logged. Cached under failure_modes/failure_advisories/,
# so a rerun is nearly free — but it is already complete.
check_13() {
  local f="$RESULTS/failure_modes/failure_modes.json"
  [ -f "$f" ] || { echo "missing $f"; return 1; }
  $PY - "$f" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
cats = s.get("categories") or s.get("counts") or {}
tot = s.get("n_failures", s.get("n_failures_total", 0))
print("n_failures={} categories={}".format(tot, len(cats) if hasattr(cats, "__len__") else "?"))
sys.exit(0 if (tot or cats) else 1)
EOF
}

# --- Phase 14: sparse-regime faithfulness sweep -----------------------------
# --faith-only skips the (expensive, GPU-ish) retraining half and runs just the
# A/B faithfulness scoring per sensor density. Done == the faith columns are
# populated for every density row.
check_14() {
  local f="$RESULTS/sparse_regime/sparse_regime_per_density.csv"
  [ -f "$f" ] || { echo "missing $f"; return 1; }
  $PY - "$f" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("empty csv"); sys.exit(1)
def filled(r, k):
    v = (r.get(k) or "").strip()
    return v not in ("", "nan", "None", "PENDING")
ok = all(filled(r, "faith_f1_A") and filled(r, "halluc_A") and filled(r, "halluc_B") for r in rows)
print("densities={} faith_cols_filled={}".format(
    [r.get("density_pct") for r in rows], ok))
sys.exit(0 if ok else 1)
EOF
}

# =============================================================================
# THE QUEUE
# =============================================================================
QUEUE_T0=$(epoch)
hr
echo "XTraffic Ollama queue — started $(now)"
echo "repo: $REPO_ROOT"
echo "ollama: $(ollama --version 2>/dev/null || echo 'NOT FOUND')"
hr

# --- 1. Phase 15b ------------------------------------------------------------
# IMPORTANT DEVIATION FROM THE REQUESTED COMMAND (flagged per CLAUDE.md):
# a bare `run_faithfulness_study` defaults to --conditions A,B,C --out-tag ""
# which would OVERWRITE the committed Phase-5 headline artifacts
# (faithfulness_summary.json / _per_scenario.csv / _distributions.pdf).
# Phase 15b is the C_RICH + D_CONTRA study, so we pass its real flags.
run_phase "Phase 15b — C_RICH + D_CONTRA faithfulness" \
  check_15b "$LOGDIR/phase15b.log" \
  $PY -m xtraffic.evaluation.run_faithfulness_study \
      --conditions A,C_RICH,D_CONTRA --out-tag 15b

# --- 2. Phase 16 -------------------------------------------------------------
run_phase "Phase 16 — active grounding loop (93 scenarios)" \
  check_16 "$LOGDIR/phase16.log" \
  $PY -m xtraffic.models.advisor.active_grounding

# --- 3. Phase 11 -------------------------------------------------------------
# The long one: 3 cities x 2 models. Builds/reuses the zero-shot transfer
# checkpoints, runs the explainer per city, then the A/B(/C) LLM study.
run_phase "Phase 11 — cross-city x cross-model faithfulness" \
  check_11 "$LOGDIR/phase11.log" \
  $PY -m xtraffic.evaluation.cross_city_faithfulness

# --- 4. Phase 12 -------------------------------------------------------------
run_phase "Phase 12 — SHAP baseline comparison (30 scenarios)" \
  check_12 "$LOGDIR/phase12.log" \
  $PY -m xtraffic.evaluation.shap_comparison

# --- 5. Phase 13 -------------------------------------------------------------
run_phase "Phase 13 — condition-A failure-mode taxonomy" \
  check_13 "$LOGDIR/phase13.log" \
  $PY -m xtraffic.evaluation.failure_modes

# --- 6. Phase 14 -------------------------------------------------------------
run_phase "Phase 14 — sparse-regime faithfulness sweep" \
  check_14 "$LOGDIR/phase14.log" \
  $PY -m xtraffic.evaluation.sparse_regime --faith-only

# =============================================================================
# SUMMARY
# =============================================================================
QUEUE_T1=$(epoch)
echo
hr
echo "QUEUE SUMMARY — finished $(now)"
echo "total wall clock: $(human $((QUEUE_T1 - QUEUE_T0)))"
hr
printf '%-46s %-16s %-14s\n' "PHASE" "STATUS" "ELAPSED"
printf '%-46s %-16s %-14s\n' "----------------------------------------------" "----------------" "--------------"
for i in "${!SUMMARY_NAME[@]}"; do
  printf '%-46s %-16s %-14s\n' \
    "${SUMMARY_NAME[$i]}" "${SUMMARY_STATUS[$i]}" "${SUMMARY_DURATION[$i]}"
  printf '    %s\n' "${SUMMARY_NOTE[$i]}"
done
hr

# Non-zero exit if anything failed, so you can tell at a glance.
fails=0
for s in "${SUMMARY_STATUS[@]}"; do
  case "$s" in FAILED*|PARTIAL) fails=$((fails+1));; esac
done
echo "phases needing attention: $fails"
echo "per-phase logs: $LOGDIR/"
exit 0
