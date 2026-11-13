#!/usr/bin/env bash
# One-off (2026-11-13): truncation controls T1 matched / T2 fixed at DS hinted s2's thinking lengths
# (notes/run_records/truncation_control/bundle_dsh2.json.xz, scripts/truncation_prep.py "DS hinted s2"). Waits until the
# re-run cf48 chain has its pod (acquisitions read the last line of logs/pool_hosts.jsonl), rents one pod, runs
# scripts/truncation_control.py, fetches the result, deletes the pod. Mac side.
set -u
cd "$(dirname "$0")/.."
K=~/.runpod/ssh/runpodctl-ssh-key; out=notes/run_records/truncation_control
say() { echo "$(date +%H:%M) trunc-dsh2 $*"; }
[ -s "$out/bundle_dsh2.json.xz" ] || { say "ALERT no bundle"; exit 1; }
until awk '/cf48 re-run/ {on=1} on && /cf48 (pod .* ready|ALERT)/ {f=1} END {exit !f}' logs/chain_cf48.log 2>/dev/null; do sleep 60; done
n0=$(wc -l < logs/pool_hosts.jsonl)
for attempt in 1 2 3 4 5 6 7 8; do
  CLOUD=COMMUNITY DISK_GB=40 GPU_LIST="${GPU_LIST:-NVIDIA GeForce RTX 4090:1.01;NVIDIA GeForce RTX 3090:0.94;NVIDIA RTX A5000:0.77;NVIDIA GeForce RTX 5090:1.79}" \
    bash scripts/acquire_pods.sh 1 30 >> logs/acquire_trunc_dsh2.log 2>&1 < /dev/null
  n1=$(wc -l < logs/pool_hosts.jsonl)
  [ "$n1" -gt "$n0" ] && break
  say "acquire attempt $attempt gave no ready pod"; sleep 60
done
[ "$n1" -gt "$n0" ] || { say "ALERT no pod acquired (logs/acquire_trunc_dsh2.log)"; exit 1; }
h=$(tail -n 1 logs/pool_hosts.jsonl); pod=$(echo "$h" | jq -r .pod); ip=$(echo "$h" | jq -r .host); port=$(echo "$h" | jq -r .port)
say "pod $pod ready ($(echo "$h" | jq -r .gpu))"
scp -q -P "$port" -i $K scripts/truncation_control.py "$out/bundle_dsh2.json.xz" "$ip:/root/" < /dev/null \
  && scp -q -P "$port" -i $K harness/python/adapters/qwen_worker.py "$ip:/root/openwish/qwen_worker.py" < /dev/null \
  || { say "ALERT upload failed; pod $pod kept"; exit 1; }
ssh -p "$port" -i $K "$ip" 'cd /root && nohup setsid bash -c "/root/vllm-env/bin/python -I truncation_control.py bundle_dsh2.json.xz trunc_dsh2_out.json > trunc_dsh2.log 2>&1; echo CTRL_EXIT=\$? >> trunc_dsh2.log" > /dev/null 2>&1 < /dev/null & echo started' < /dev/null
say "controls started"
t0=$(date +%s)
until ssh -o ConnectTimeout=30 -p "$port" -i $K "$ip" 'grep -q CTRL_EXIT= /root/trunc_dsh2.log' < /dev/null 2>/dev/null; do
  [ $(( $(date +%s) - t0 )) -gt 14400 ] && { say "ALERT controls not done after 4 h; pod $pod kept"; exit 1; }
  sleep 300
done
scp -q -P "$port" -i $K "$ip:/root/trunc_dsh2_out.json" "$ip:/root/trunc_dsh2.log" "$out/" < /dev/null
say "controls done: $(grep -c '^{' $out/trunc_dsh2.log) conditions ($(grep CTRL_EXIT $out/trunc_dsh2.log))"
if [ -s "$out/trunc_dsh2_out.json" ]; then
  runpodctl pod delete "$pod" >/dev/null 2>&1; sleep 5
  if runpodctl pod list 2>/dev/null | jq -e --arg p "$pod" 'any(.[]; .id == $p)' >/dev/null; then say "ALERT pod $pod still listed after delete"; exit 1; fi
  say "DONE: pod $pod deleted"
else
  say "ALERT no trunc_dsh2_out.json fetched; pod $pod kept"; exit 1
fi
