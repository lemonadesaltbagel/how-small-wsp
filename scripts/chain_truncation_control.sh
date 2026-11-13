#!/usr/bin/env bash
# One-off (2026-11-13): rent one target pod, run the truncation controls (scripts/truncation_control.py, T1/T2 for DS
# unhinted s2, Astra p50c, Opus strict s1-s3) and the 4-sample thinking-off control on DS unhinted s2's rescore items,
# fetch the results, delete the pod. Mac side.
set -u
cd "$(dirname "$0")/.."
K=~/.runpod/ssh/runpodctl-ssh-key; out=notes/run_records/truncation_control
items4=harness/runs_laptop/phase0/deepseekflash-unhinted-p50-strict2-s2-rescore4/eval/qwen_math500/items.json
say() { echo "$(date +%H:%M) trunc $*"; }
# REUSE=<pod id>: use that pod (a line of logs/pool_hosts.jsonl) instead of renting one
if [ -n "${REUSE:-}" ]; then
  h=$(grep -F "\"$REUSE\"" logs/pool_hosts.jsonl | tail -n 1); [ -n "$h" ] || { say "ALERT $REUSE not in logs/pool_hosts.jsonl"; exit 1; }
else
  n0=$(wc -l < logs/pool_hosts.jsonl)
  for attempt in 1 2 3 4 5 6 7 8; do
    CLOUD=COMMUNITY DISK_GB=40 GPU_LIST="NVIDIA GeForce RTX 5090:1.79;NVIDIA GeForce RTX 4090:1.01;NVIDIA RTX A5000:0.77;NVIDIA GeForce RTX 3090:0.94" \
      bash scripts/acquire_pods.sh 1 30 >> logs/acquire_trunc.log 2>&1 < /dev/null
    n1=$(wc -l < logs/pool_hosts.jsonl)
    [ "$n1" -gt "$n0" ] && break
    say "acquire attempt $attempt gave no ready pod"; sleep 60
  done
  [ "$n1" -gt "$n0" ] || { say "ALERT no pod acquired (logs/acquire_trunc.log)"; exit 1; }
  h=$(tail -n 1 logs/pool_hosts.jsonl)
fi
pod=$(echo "$h" | jq -r .pod); ip=$(echo "$h" | jq -r .host); port=$(echo "$h" | jq -r .port)
say "pod $pod ready ($(echo "$h" | jq -r .gpu))"
scp -q -P "$port" -i $K scripts/truncation_control.py scripts/nothink_control.py "$out/bundle.json.xz" "$ip:/root/" < /dev/null \
  && scp -q -P "$port" -i $K harness/python/adapters/qwen_worker.py "$ip:/root/openwish/qwen_worker.py" < /dev/null \
  && scp -q -P "$port" -i $K "$items4" "$ip:/root/items4.json" < /dev/null || { say "ALERT upload failed; pod $pod kept"; exit 1; }
ssh -p "$port" -i $K "$ip" 'cd /root && nohup setsid bash -c "/root/vllm-env/bin/python -I truncation_control.py bundle.json.xz trunc_out.json > trunc.log 2>&1; /root/vllm-env/bin/python -I nothink_control.py items4.json nothink4_out.json > nothink4.log 2>&1; echo CTRL_EXIT=\$? >> trunc.log" > /dev/null 2>&1 < /dev/null & echo started' < /dev/null
say "controls started"
t0=$(date +%s)
until ssh -o ConnectTimeout=30 -p "$port" -i $K "$ip" 'grep -q CTRL_EXIT= /root/trunc.log' < /dev/null 2>/dev/null; do
  [ $(( $(date +%s) - t0 )) -gt 28800 ] && { say "ALERT controls not done after 8 h; pod $pod kept"; exit 1; }
  sleep 300
done
scp -q -P "$port" -i $K "$ip:/root/trunc_out.json" "$ip:/root/trunc.log" "$ip:/root/nothink4_out.json" "$ip:/root/nothink4.log" "$out/" < /dev/null
say "controls done: $(grep -c '^{' $out/trunc.log) truncation conditions; $(grep nothink_control $out/nothink4.log | cut -c1-300)"
if [ -s "$out/trunc_out.json" ]; then
  runpodctl pod delete "$pod" >/dev/null 2>&1; sleep 5
  if runpodctl pod list 2>/dev/null | jq -e --arg p "$pod" 'any(.[]; .id == $p)' >/dev/null; then say "ALERT pod $pod still listed after delete"; exit 1; fi
  say "DONE: pod $pod deleted"
else
  say "ALERT no trunc_out.json fetched; pod $pod kept"; exit 1
fi
