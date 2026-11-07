#!/usr/bin/env bash
# Run ON WSL: hold-out evaluations of the base on the local GPU, in its own session, with a memory watchdog.
#   tmux new -d -s localq "bash scripts/local_queue.sh <logfile>"   (a tmux pane is its own session and process group)
# The watchdog kills this queue's whole process group (never an episode) if MemAvailable drops below MIN_MB.
set -uo pipefail
cd "$(dirname "$0")/../harness"
export PATH="$HOME/.local/bin:$PATH" QWEN_GPU_UTIL=0.75 HF_HUB_DISABLE_XET=1
MIN_MB=${MIN_MB:-3000}
log=${1:-../logs/local_queue.log}
pg=$(ps -o pgid= -p $$ | tr -d ' ')
( while sleep 10; do
    a=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
    if [ "$a" -lt "$MIN_MB" ]; then echo "$(date +%H:%M:%S) WATCHDOG: MemAvailable ${a} MB < ${MIN_MB}; killing local queue (pgid $pg)" >> "$log"; kill -TERM -"$pg"; exit; fi
  done ) &
step() { echo "$(date +%H:%M:%S) == $*" >> "$log"; }
# hold-out benchmarks of the unedited base (the reference for "no regression"; published: MMLU-Pro 40.3, GPQA 22.7, IFEval 21.0)
H="assets/venvs/qwen/bin/python -I python/adapters/holdout_eval.py evaluate --model assets/exp8/qwen35-text --data assets/exp8/holdout"
step "hold-out smoke: IFEval, 5 problems"
$H --set ifeval --limit 5 --out runs/holdout/smoke_ifeval >> "$log" 2>&1 || { step "smoke failed; stopping"; exit 1; }
for s in ifeval mmlupro1k gpqa; do
  step "hold-out $s, unedited base (authors' samples)"
  [ -f runs/holdout/base_$s/summary.json ] || $H --set $s --out runs/holdout/base_$s >> "$log" 2>&1 || { step "$s failed; stopping"; exit 1; }
done
step "done"
kill %1 2>/dev/null
