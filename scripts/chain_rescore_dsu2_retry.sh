#!/usr/bin/env bash
# One-off (2026-11-12): the first DS unhinted s2 4-sample rescore ran the base model, because the rescore run dir had
# no edits/ledger.jsonl and so no edited tensors were staged (kept as base-math500-rescore4-20261112). This script
# rents one pod, reruns the rescore with the ledger staged, then deletes the pod. Mac side.
set -u
cd "$(dirname "$0")/.."
run=runs_laptop/phase0/deepseekflash-unhinted-p50-strict2-s2; remote=scratch_remote_rescore_dsu2.json; rlog=logs/rescore_dsu2.log
say() { echo "$(date +%H:%M) rescore-dsu2 $*"; }
[ -s "harness/$run-rescore4/edits/ledger.jsonl" ] || { say "ALERT no ledger in the rescore dir"; exit 1; }
n0=$(wc -l < logs/pool_hosts.jsonl)
# acquire_pods.sh counts created pods, not set-up ones (a pod on a known bad host is deleted and still counted): retry
for attempt in 1 2 3 4 5 6 7 8; do
  CLOUD=COMMUNITY DISK_GB=40 GPU_LIST="NVIDIA GeForce RTX 5090:1.79;NVIDIA GeForce RTX 4090:1.01;NVIDIA RTX A5000:0.77;NVIDIA GeForce RTX 3090:0.94" \
    bash scripts/acquire_pods.sh 1 30 >> logs/acquire_rescore_dsu2.log 2>&1 < /dev/null
  n1=$(wc -l < logs/pool_hosts.jsonl)
  [ "$n1" -gt "$n0" ] && break
  say "acquire attempt $attempt gave no ready pod"; sleep 60
done
[ "$n1" -gt "$n0" ] || { say "ALERT no pod acquired (logs/acquire_rescore_dsu2.log)"; exit 1; }
h=$(tail -n 1 logs/pool_hosts.jsonl); pod=$(echo "$h" | jq -r .pod)
echo "{\"hosts\":[$h]}" > "$remote"
say "pod $pod ready ($(echo "$h" | jq -r .gpu))"
rx -n wsl "mkdir -p logs && tmux new -d -s rescore-dsu2 \"cd harness && OPENWISH_RUN=$run-rescore4 OPENWISH_FINAL_NET=$run/final/net OPENWISH_QWEN_REMOTE=\\\$(cd .. && pwd)/$remote assets/venvs/sandbox/bin/python -I python/adapters/qwen_pass.py --evaluate --sets math500 --samples 4 > ../$rlog 2>&1; echo RESCORE_EXIT=\\\$? >> ../$rlog\"" > /dev/null 2>&1
say "rescore started (wsl tmux rescore-dsu2, $rlog)"
until grep -q "RESCORE_EXIT=" "$rlog" 2>/dev/null; do sleep 300; done
say "rescore done ($(grep RESCORE_EXIT "$rlog")): $(grep '^{' "$rlog" | tail -n 1 | jq -c '{score, n_edited: (.edited_tensors|length), se: .math500.accuracy_se, cut: .math500.share_cut_at_max_tokens}' 2>/dev/null)"
runpodctl pod delete "$pod" >/dev/null 2>&1; sleep 5
if runpodctl pod list 2>/dev/null | jq -e --arg p "$pod" 'any(.[]; .id == $p)' >/dev/null; then say "ALERT pod $pod still listed after delete"; exit 1; fi
say "DONE: pod $pod deleted"
