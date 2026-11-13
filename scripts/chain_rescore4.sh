#!/usr/bin/env bash
# 4-sample MATH-500 rescore of one counted repair (2026-11-13: success runs are reported on 4 samples).
#   bash scripts/chain_rescore4.sh <tag> <rescore run dir> <final net dir>      (paths relative to harness/)
# The rescore dir must hold edits/ledger.jsonl (else the base is scored). WAIT_FOR="<log>:<regex>" waits for that line
# first (acquisitions read the last line of logs/pool_hosts.jsonl, so chains must not acquire at the same time).
# Rents one pod (community, bad hosts skipped; CLOUD / GPU_LIST overridable), scores from wsl, deletes the pod. Mac side.
set -u
cd "$(dirname "$0")/.."
tag=$1; run=$2; net=$3; remote=scratch_remote_rescore4_$tag.json; rlog=logs/rescore4_$tag.log
say() { echo "$(date +%H:%M) rescore4-$tag $*"; }
[ -s "harness/$run/edits/ledger.jsonl" ] || { say "ALERT no ledger in the rescore dir"; exit 1; }
[ -d "harness/$net" ] || { say "ALERT no final net $net"; exit 1; }
if [ -n "${WAIT_FOR:-}" ]; then
  until grep -q -E "${WAIT_FOR#*:}" "${WAIT_FOR%%:*}" 2>/dev/null; do sleep 60; done
fi
n0=$(wc -l < logs/pool_hosts.jsonl)
for attempt in 1 2 3 4 5 6 7 8; do
  CLOUD="${CLOUD:-COMMUNITY}" DISK_GB=40 GPU_LIST="${GPU_LIST:-NVIDIA GeForce RTX 4090:1.01;NVIDIA RTX A5000:0.77;NVIDIA GeForce RTX 3090:0.94;NVIDIA GeForce RTX 5090:1.79}" \
    bash scripts/acquire_pods.sh 1 30 >> logs/acquire_rescore4_$tag.log 2>&1 < /dev/null
  n1=$(wc -l < logs/pool_hosts.jsonl)
  [ "$n1" -gt "$n0" ] && break
  say "acquire attempt $attempt gave no ready pod"; sleep 60
done
[ "$n1" -gt "$n0" ] || { say "ALERT no pod acquired (logs/acquire_rescore4_$tag.log)"; exit 1; }
h=$(tail -n 1 logs/pool_hosts.jsonl); pod=$(echo "$h" | jq -r .pod)
echo "{\"hosts\":[$h]}" > "$remote"
say "pod $pod ready ($(echo "$h" | jq -r .gpu))"
sleep 30   # the Mutagen sync carries the pool file and the rescore dir to wsl
rx -n wsl "mkdir -p logs && tmux new -d -s rescore4-$tag \"cd harness && OPENWISH_RUN=$run OPENWISH_FINAL_NET=$net OPENWISH_QWEN_REMOTE=\\\$(cd .. && pwd)/$remote assets/venvs/sandbox/bin/python -I python/adapters/qwen_pass.py --evaluate --sets math500 --samples 4 > ../$rlog 2>&1; echo RESCORE_EXIT=\\\$? >> ../$rlog\"" > /dev/null 2>&1
say "rescore started (wsl tmux rescore4-$tag, $rlog)"
until grep -q "RESCORE_EXIT=" "$rlog" 2>/dev/null; do sleep 300; done
say "rescore done ($(grep RESCORE_EXIT "$rlog")): $(grep '^{' "$rlog" | tail -n 1 | jq -c '{score, n_edited: (.edited_tensors|length), se: .math500.accuracy_se, cut: .math500.share_cut_at_max_tokens}' 2>/dev/null)"
runpodctl pod delete "$pod" >/dev/null 2>&1; sleep 5
if runpodctl pod list 2>/dev/null | jq -e --arg p "$pod" 'any(.[]; .id == $p)' >/dev/null; then say "ALERT pod $pod still listed after delete"; exit 1; fi
say "DONE: pod $pod deleted"
