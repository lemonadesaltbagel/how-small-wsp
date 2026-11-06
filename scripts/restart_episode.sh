#!/usr/bin/env bash
# Run ON WSL (via rx): stop an episode between forward passes and resume it with the current binary.
#   restart_episode.sh <tmux session> <task> <run dir rel. to harness/> <remote.json|local> <log> [openwish flags...]
# Waits (up to 3 h) until the run's openwish process has no qwen_pass child (no forward pass in flight), then
# sends Ctrl-C, waits for the process to exit, and starts run_episode.sh again (which resumes the run dir).
set -uo pipefail
cd "$(dirname "$0")/.."
sess=$1; task=$2; run=$3; remote=$4; log=$5; shift 5
pid=$(pgrep -f "openwish run .*--out $run( |$)" | head -n 1)
[ -n "$pid" ] || { echo "no running process for $run"; exit 1; }
busy() { for c in $(pgrep -P "$pid"); do ps -o args= -p "$c" 2>/dev/null | grep -q qwen_pass && return 0; done; return 1; }
for i in $(seq 1 360); do busy || break; sleep 30; done
busy && { echo "still in a forward pass after 3 h; not stopping"; exit 1; }
echo "$(date +%H:%M:%S) stopping $sess (pid $pid)"
tmux send-keys -t "$sess" C-c
for i in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
if kill -0 "$pid" 2>/dev/null; then kill "$pid"; sleep 5; fi
tmux kill-session -t "$sess" 2>/dev/null
cmd="bash scripts/run_episode.sh $task $run $remote $* 2>&1 | tee -a $log"
tmux new -d -s "$sess" "$cmd"
echo "$(date +%H:%M:%S) resumed $sess: $cmd"
