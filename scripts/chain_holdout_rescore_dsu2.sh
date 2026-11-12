#!/usr/bin/env bash
# One-off (2026-11-12): DS unhinted s2 scored 53.0 % (a repair). Replaces the tail of post_episode.sh dsux2 so the
# target pod is kept after the hold-out: wait for the hold-out (as post_episode.sh does), then rescore MATH-500 with
# 4 samples on the same pod (from wsl, as for DS hinted s1), then delete the pod. Mac side.
set -u
cd "$(dirname "$0")/.."
run=runs_laptop/phase0/deepseekflash-unhinted-p50-strict2-s2; pod=h5920phi2ty7jp; remote=scratch_remote_dsu_strict2_s2.json
sess=dsux2; log=logs/holdout_$sess.log; rlog=logs/rescore_dsu2.log
say() { echo "$(date +%H:%M) $sess $*"; }
until grep -q "HOLDOUT_EXIT=" "$log" 2>/dev/null; do sleep 300; done
out=$(grep '^{' "$log" | tail -n 1)
echo "$out" > "harness/$run/eval/holdout_summary.json"
say "hold-out done ($(grep HOLDOUT_EXIT "$log")): $(echo "$out" | jq -c '{ok, mmlupro1k: .mmlupro1k.accuracy, gpqa: .gpqa.accuracy, ifeval: .ifeval.accuracy}' 2>/dev/null || echo "$out" | cut -c1-200)"
sleep 30
rx -n wsl "mkdir -p logs && tmux new -d -s rescore-dsu2 \"cd harness && OPENWISH_RUN=$run-rescore4 OPENWISH_FINAL_NET=$run/final/net OPENWISH_QWEN_REMOTE=\\\$(cd .. && pwd)/$remote assets/venvs/sandbox/bin/python -I python/adapters/qwen_pass.py --evaluate --sets math500 --samples 4 > ../$rlog 2>&1; echo RESCORE_EXIT=\\\$? >> ../$rlog\"" > /dev/null 2>&1
say "rescore started (wsl tmux rescore-dsu2, $rlog)"
until grep -q "RESCORE_EXIT=" "$rlog" 2>/dev/null; do sleep 300; done
say "rescore done ($(grep RESCORE_EXIT "$rlog")): $(grep '^{' "$rlog" | tail -n 1 | jq -c '{score, se: .math500.accuracy_se, cut: .math500.share_cut_at_max_tokens}' 2>/dev/null)"
runpodctl pod delete "$pod" >/dev/null 2>&1; sleep 5
if runpodctl pod list 2>/dev/null | jq -e --arg p "$pod" 'any(.[]; .id == $p)' >/dev/null; then say "ALERT pod $pod still listed after delete"; exit 1; fi
say "DONE: pod $pod deleted"
