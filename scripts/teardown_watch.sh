#!/usr/bin/env bash
# Mac-side watcher: delete a target-GPU pod once its episode is scored and finished on WSL.
#   teardown_watch.sh <tmux session on WSL> <run dir relative to harness/> <pod id> [--check]
# Prints one line per state change (for a Monitor). If the tmux session ends without eval/score.json, it prints
# ALERT and exits WITHOUT deleting (so the run can be resumed on the same pod).
set -uo pipefail
cd "$(dirname "$0")/.."
sess=$1; run=$2; pod=$3; mode=${4:-}
alive() { rx "tmux has-session -t $sess 2>/dev/null && echo yes || echo no" 2>/dev/null | tail -n 1; }
scored() { [ -f "harness/$run/eval/score.json" ] && echo yes || echo no; }
if [ "$mode" = "--check" ]; then
  echo "session alive: $(alive); scored: $(scored); pod: $(runpodctl pod get "$pod" 2>/dev/null | jq -r '.desiredStatus // "missing"')"
  exit 0
fi
last=""
while true; do
  a=$(alive); s=$(scored)
  state="alive=$a scored=$s"
  [ "$state" != "$last" ] && echo "$(date +%H:%M) $sess $state" && last=$state
  if [ "$a" = "no" ] && [ "$s" = "yes" ]; then
    runpodctl pod delete "$pod" >/dev/null 2>&1
    sleep 5
    if runpodctl pod list 2>/dev/null | jq -e --arg p "$pod" 'any(.[]; .id == $p)' >/dev/null; then
      echo "ALERT $sess: delete of pod $pod did not take; retrying"; continue
    fi
    echo "$(date +%H:%M) $sess DONE: score $(jq -r '.score' "harness/$run/eval/score.json"); pod $pod deleted"
    exit 0
  fi
  if [ "$a" = "no" ] && [ "$s" = "no" ]; then
    echo "ALERT $sess ended without a score; pod $pod kept for --resume"
    exit 1
  fi
  sleep 300
done
