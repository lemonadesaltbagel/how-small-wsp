#!/usr/bin/env bash
# One-off (2026-11-13): the desktop was restarted (Windows driver) while two 4-sample MATH-500 scorers ran from wsl;
# both were stopped before the reboot without writing RESCORE_EXIT, so their Mac-side chains (chain_rescore4_dsh2.sh,
# chain_cf48.sh) still wait and still own their pods. This re-starts both scorers in wsl tmux with the same pool files
# and logs (qwen_pass ends any old worker on the pod first). Mac side.
set -u
cd "$(dirname "$0")/.."
start() {  # start <tmux name> <run dir> <final net dir> <pool file> <log>
  rx -n wsl "mkdir -p logs && tmux has-session -t $1 2>/dev/null && echo 'already running' || tmux new -d -s $1 \"cd harness && OPENWISH_RUN=$2 OPENWISH_FINAL_NET=$3 OPENWISH_QWEN_REMOTE=\\\$(cd .. && pwd)/$4 assets/venvs/sandbox/bin/python -I python/adapters/qwen_pass.py --evaluate --sets math500 --samples 4 > ../$5 2>&1; echo RESCORE_EXIT=\\\$? >> ../$5\""
}
r=runs_laptop/phase0
start rescore-dsh2-4s $r/deepseekflash-hinted-p50-strict2-s2-rescore4 $r/deepseekflash-hinted-p50-strict2-s2/final/net scratch_remote_rescore_dsh2_4s.json logs/rescore_dsh2_4s.log
start rescore-cf48 $r/deepseekflash-unhinted-p50-strict2-s2-cf48 $r/deepseekflash-unhinted-p50-strict2-s2-cf48/final/net scratch_remote_cf48.json logs/rescore_cf48.log
sleep 20
rx -n wsl 'ps -eo etime,args | grep "[q]wen_pass" | cut -c1-120'
