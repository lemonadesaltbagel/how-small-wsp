#!/usr/bin/env bash
# One-off (2026-11-13): score the DS unhinted s2 counterfactual #48 (the submitted edit with lam = 0; built by
# scripts/make_cf48.py) on MATH-500 with 4 samples, as the 4-sample rescore of the submission (54.7 %). Waits until
# the re-run truncation chain has its pod (acquisitions read the last line of logs/pool_hosts.jsonl), rents one pod,
# runs the scorer from wsl, deletes the pod. Mac side.
set -u
cd "$(dirname "$0")/.."
run=runs_laptop/phase0/deepseekflash-unhinted-p50-strict2-s2-cf48; remote=scratch_remote_cf48.json; rlog=logs/rescore_cf48.log
say() { echo "$(date +%H:%M) cf48 $*"; }
[ -s "harness/$run/edits/ledger.jsonl" ] && [ -s "harness/$run/cf48.json" ] || { say "ALERT counterfactual dir incomplete"; exit 1; }
# the truncation chain's re-run starts after the marker line; wait for its pod (or its ALERT)
until awk '/trunc re-run/ {on=1} on && /trunc (pod .* ready|ALERT)/ {f=1} END {exit !f}' logs/chain_trunc.log 2>/dev/null; do sleep 60; done
n0=$(wc -l < logs/pool_hosts.jsonl)
for attempt in 1 2 3 4 5 6 7 8; do
  CLOUD=COMMUNITY DISK_GB=40 GPU_LIST="NVIDIA GeForce RTX 5090:1.79;NVIDIA GeForce RTX 4090:1.01;NVIDIA RTX A5000:0.77;NVIDIA GeForce RTX 3090:0.94" \
    bash scripts/acquire_pods.sh 1 30 >> logs/acquire_cf48.log 2>&1 < /dev/null
  n1=$(wc -l < logs/pool_hosts.jsonl)
  [ "$n1" -gt "$n0" ] && break
  say "acquire attempt $attempt gave no ready pod"; sleep 60
done
[ "$n1" -gt "$n0" ] || { say "ALERT no pod acquired (logs/acquire_cf48.log)"; exit 1; }
h=$(tail -n 1 logs/pool_hosts.jsonl); pod=$(echo "$h" | jq -r .pod)
echo "{\"hosts\":[$h]}" > "$remote"
say "pod $pod ready ($(echo "$h" | jq -r .gpu))"
sleep 30   # the Mutagen sync carries the pool file to wsl
rx -n wsl "mkdir -p logs && tmux new -d -s rescore-cf48 \"cd harness && OPENWISH_RUN=$run OPENWISH_FINAL_NET=$run/final/net OPENWISH_QWEN_REMOTE=\\\$(cd .. && pwd)/$remote assets/venvs/sandbox/bin/python -I python/adapters/qwen_pass.py --evaluate --sets math500 --samples 4 > ../$rlog 2>&1; echo RESCORE_EXIT=\\\$? >> ../$rlog\"" > /dev/null 2>&1
say "re-score started (wsl tmux rescore-cf48, $rlog)"
until grep -q "RESCORE_EXIT=" "$rlog" 2>/dev/null; do sleep 300; done
say "re-score done ($(grep RESCORE_EXIT "$rlog")): $(grep '^{' "$rlog" | tail -n 1 | jq -c '{score, n_edited: (.edited_tensors|length), se: .math500.accuracy_se, cut: .math500.share_cut_at_max_tokens}' 2>/dev/null)"
runpodctl pod delete "$pod" >/dev/null 2>&1; sleep 5
if runpodctl pod list 2>/dev/null | jq -e --arg p "$pod" 'any(.[]; .id == $p)' >/dev/null; then say "ALERT pod $pod still listed after delete"; exit 1; fi
say "DONE: pod $pod deleted"
