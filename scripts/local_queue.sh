#!/usr/bin/env bash
# Run ON WSL: a queue of local GPU evaluations, in its own session, with a memory watchdog.
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
# each step runs only if the previous one succeeded (an empty edit folder would silently evaluate the base model)
step "KidIkaros scripted-solver checkpoint -> edit folder"
rm -rf assets/exp8/published/kidikaros assets/exp8/published/kidikaros.partial
assets/venvs/sandbox/bin/python -I ../scripts/kidikaros_edits.py >> "$log" 2>&1 || { step "conversion failed; stopping"; exit 1; }
step "MATH-500, KidIkaros scripted floor"
python3 scripts/exp8_conformance.py kidikaros >> "$log" 2>&1 || { step "evaluation failed"; exit 1; }
step "done"
kill %1 2>/dev/null
