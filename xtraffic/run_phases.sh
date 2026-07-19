#!/usr/bin/env bash
# =============================================================================
# XTraffic — sequential runner for Phase 11 then Phase 12
# =============================================================================
# Phase 12 starts ONLY after Phase 11 has fully exited. Both share the one local
# Ollama server, so they must never overlap.
#
# Launch it under screen (tmux is not installed on this machine):
#   screen -dmS xtraffic bash /Users/narensara/Desktop/OlympiFlow/xtraffic/run_phases.sh
#
# Everything below uses ABSOLUTE paths, so it runs correctly from any directory.
#
# Both underlying scripts are RESUMABLE — they cache per-decision to JSONL and
# skip work already on disk. Killing this mid-run loses at most the current LLM
# call, and relaunching picks up where it stopped.
# -----------------------------------------------------------------------------

set -u  # error on undefined variables

REPO="/Users/narensara/Desktop/OlympiFlow"
PY="/usr/bin/env python3"
LOGDIR="$REPO/xtraffic/evaluation/results/phase_logs"

mkdir -p "$LOGDIR"

# Run from the repo root — required for `python3 -m xtraffic.*` module imports
# to resolve. This is why an absolute REPO path matters.
cd "$REPO" || { echo "FATAL: cannot cd to $REPO"; exit 1; }

PHASE11_LOG="$LOGDIR/phase11_cross_city.log"
PHASE12_LOG="$LOGDIR/phase12_shap.log"
MAIN_LOG="$LOGDIR/run_phases_main.log"

# Mirror this script's own progress output to a file as well as the screen,
# so you can check status without reattaching to the screen session.
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "============================================================"
echo "XTraffic sequential run"
echo "started:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "repo:     $REPO"
echo "logs:     $LOGDIR"
echo "============================================================"

# -----------------------------------------------------------------------------
# PHASE 11 — cross-city x cross-model faithfulness
# -----------------------------------------------------------------------------
echo
echo "------------------------------------------------------------"
echo "PHASE 11 — cross_city_faithfulness"
echo "start:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "log:     $PHASE11_LOG"
echo "------------------------------------------------------------"

P11_T0=$(date '+%s')
$PY -m xtraffic.evaluation.cross_city_faithfulness >> "$PHASE11_LOG" 2>&1
P11_RC=$?
P11_T1=$(date '+%s')
P11_ELAPSED=$((P11_T1 - P11_T0))

echo "PHASE 11 finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "PHASE 11 exit code: $P11_RC"
printf 'PHASE 11 elapsed: %dh %02dm %02ds\n' \
  $((P11_ELAPSED/3600)) $((P11_ELAPSED%3600/60)) $((P11_ELAPSED%60))

if [ $P11_RC -ne 0 ]; then
  echo "WARNING: Phase 11 exited non-zero. Continuing to Phase 12 anyway —"
  echo "         Phase 12 is independent of Phase 11's results, and Phase 11"
  echo "         resumes from its cache on a later relaunch."
fi

# -----------------------------------------------------------------------------
# PHASE 12 — SHAP baseline comparison
# -----------------------------------------------------------------------------
# This line is only reached after the Phase 11 process above has fully exited,
# which is the sequencing guarantee we need for the shared Ollama server.
echo
echo "------------------------------------------------------------"
echo "PHASE 12 — shap_comparison"
echo "start:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "log:     $PHASE12_LOG"
echo "------------------------------------------------------------"

P12_T0=$(date '+%s')
$PY -m xtraffic.evaluation.shap_comparison >> "$PHASE12_LOG" 2>&1
P12_RC=$?
P12_T1=$(date '+%s')
P12_ELAPSED=$((P12_T1 - P12_T0))

echo "PHASE 12 finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "PHASE 12 exit code: $P12_RC"
printf 'PHASE 12 elapsed: %dh %02dm %02ds\n' \
  $((P12_ELAPSED/3600)) $((P12_ELAPSED%3600/60)) $((P12_ELAPSED%60))

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
echo
echo "============================================================"
echo "BOTH PHASES DONE — $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
printf 'Phase 11 (cross-city):  exit %-3s  %dh %02dm %02ds\n' \
  "$P11_RC" $((P11_ELAPSED/3600)) $((P11_ELAPSED%3600/60)) $((P11_ELAPSED%60))
printf 'Phase 12 (SHAP):        exit %-3s  %dh %02dm %02ds\n' \
  "$P12_RC" $((P12_ELAPSED/3600)) $((P12_ELAPSED%3600/60)) $((P12_ELAPSED%60))
echo
echo "Check the results landed:"
echo "  cat $REPO/xtraffic/evaluation/results/cross_city_faithfulness/cross_city_faithfulness_master.csv"
echo "  (no PENDING rows == Phase 11 complete)"
echo "  grep n_scenarios $REPO/xtraffic/evaluation/results/shap_comparison/shap_comparison_summary.json"
echo "  (should be ~28-30, not 3)"
echo "============================================================"

exit 0
