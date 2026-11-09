#!/usr/bin/env bash
# Stop the strict episodes and delete their pool pods before the RunPod balance runs out (user's rule, 2026-11-08:
# below $5, save state for --resume and delete the pods; continue after a top-up). Runs on the Mac:
#   nohup bash scripts/low_balance_stop.sh [threshold] > logs/low_balance_stop.log 2>&1 < /dev/null &
# Episodes resume without losing a reply (openwish run --resume); the pods hold no state (each pass sends the edits).
# Only the pods listed below are touched. Exits when none of them is left.
set -u
cd "$(dirname "$0")/.."
TH=${1:-5}
# node session run-dir(relative to harness/) pods episode-log
RUNS=(
  "wsl q9bx1 runs/phase0/qwen9b-hinted-p50-strict2-s1 ilppyd3ecb6qjj,jvoma6feywsyqm q9b_strict2_s1"
  "laptop q9bx2 runs_laptop/phase0/qwen9b-hinted-p50-strict2-s2 2pc75n05yrqfyi,mutruiktz8i5e1,5plj5dix1grckr q9b_strict2_s2"
)
say() { echo "$(date '+%m-%d %H:%M') $*"; }
all_pods() { for r in "${RUNS[@]}"; do set -- $r; echo "${4//,/ }"; done; }
listed() { runpodctl pod list 2>/dev/null | jq -er '.[].id // empty' || true; }
listed_ok() { runpodctl pod list 2>/dev/null | jq -e 'type == "array"' >/dev/null; }  # an empty list is a valid answer

say "watching balance, threshold \$$TH"
while true; do
  live=$(listed)
  ours=""; for p in $(all_pods); do grep -qx "$p" <<< "$live" && ours="$ours $p"; done
  if [ -z "$ours" ] && listed_ok; then say "none of the listed pods is left; exiting"; exit 0; fi
  bal=$(runpodctl me 2>/dev/null | jq -r '.clientBalance // empty')
  if [ -n "$bal" ] && awk -v b="$bal" -v t="$TH" 'BEGIN{exit !(b < t)}'; then break; fi
  sleep 300
done

say "ALERT balance \$$bal < \$$TH: stopping episodes and deleting pods"
# 1. post-episode pipelines first, so none of them starts scoring or a hold-out on a pod about to go
for p in $(pgrep -f "[p]ost_episode.sh"); do kill "$p"; done
# 2. episodes (they resume from state.json); then flush the node's sync so the run dirs are complete here
for r in "${RUNS[@]}"; do
  set -- $r
  if rx -n "$1" "tmux kill-session -t $2 2>/dev/null; echo stopped" 2>/dev/null | grep -q stopped; then
    say "$2 on $1 stopped"
  else
    say "ALERT could not reach $1 to stop $2 (its next pass will fail on the deleted pods; resume it later)"
  fi
done
# 3. pods
for p in $ours; do runpodctl pod delete "$p" >/dev/null 2>&1; sleep 5; done
left=""; live=$(listed); for p in $ours; do grep -qx "$p" <<< "$live" && left="$left $p"; done
[ -n "$left" ] && say "ALERT pods still listed after delete:$left"
say "deleted:$ours"
# 4. how to resume
{
  echo "# Resume after the low-balance stop ($(date '+%Y-%m-%d %H:%M'), balance \$$bal)"
  echo "Get new pods (scripts/acquire_pods.sh), write each run's scratch_remote_*.json with them (scripts/swap_pool.sh),"
  echo "then restart each episode with the same command (run_episode.sh resumes an existing run dir) and its post_episode.sh."
  for r in "${RUNS[@]}"; do
    set -- $r
    echo "- $2 on $1: harness/$3, log logs/$5.log, last: $(grep -o 'forward [0-9]*/50' "logs/$5.log" 2>/dev/null | tail -n 1)"
  done
} > logs/RESUME_low_balance.md
say "DONE low-balance stop; see logs/RESUME_low_balance.md"
