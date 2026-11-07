#!/usr/bin/env bash
# Run ON WSL inside tmux: one local GPU job under the memory watchdog of local_queue.sh.
#   tmux new -d -s localq "bash scripts/local_job.sh <log> <command...>"      (command runs in harness/)
# The watchdog kills this job's process group (never an episode) if MemAvailable drops below MIN_MB.
set -uo pipefail
cd "$(dirname "$0")/../harness"
export PATH="$HOME/.local/bin:$PATH" QWEN_GPU_UTIL=${QWEN_GPU_UTIL:-0.75} HF_HUB_DISABLE_XET=1
MIN_MB=${MIN_MB:-3000}
log=$1; shift
pg=$(ps -o pgid= -p $$ | tr -d ' ')
( while sleep 10; do
    a=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
    if [ "$a" -lt "$MIN_MB" ]; then echo "$(date +%H:%M:%S) WATCHDOG: MemAvailable ${a} MB < ${MIN_MB}; killing local job (pgid $pg)" >> "$log"; kill -TERM -"$pg"; exit; fi
  done ) &
echo "$(date +%H:%M:%S) == $*" >> "$log"
"$@" >> "$log" 2>&1
echo "$(date +%H:%M:%S) == exit $?" >> "$log"
kill %1 2>/dev/null
