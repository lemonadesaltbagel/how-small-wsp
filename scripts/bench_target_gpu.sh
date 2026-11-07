#!/usr/bin/env bash
# Benchmark one target-GPU type on a real pass: create a secure pod, set it up, replay a saved generate request
# (default: Opus s1 pass 1 = 512 base-model answers + activations; 64.3 min on an RTX 5090), time it, delete the pod.
#   bench_target_gpu.sh "<gpu id>" <label> [request.json]      (Mac side; one result line at the end)
set -uo pipefail
cd "$(dirname "$0")/.."
gpu=$1; label=$2; req=${3:-harness/runs/phase0/opus55-hinted-p50-s1/requests/forward_1.json}
KEY=~/.runpod/ssh/runpodctl-ssh-key
say() { echo "$(date +%H:%M) bench $label $*"; }
id=$(scripts/editor_pod.sh create "wsl-ladder-bench-$label" "$gpu" 60 2>&1 | grep -E '^[a-z0-9]{10,}$')
[ -n "$id" ] || { say "no stock"; exit 1; }
echo "$(date +%Y-%m-%dT%H:%M) wsl-ladder-bench-$label $id $gpu" >> logs/pods_bench.txt
say "pod $id"
read -r ip port < <(scripts/editor_pod.sh wait "$id")
p() { ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ServerAliveInterval=30 -p "$port" -i "$KEY" "root@$ip" "$@" < /dev/null; }
scp -q -o StrictHostKeyChecking=accept-new -P "$port" -i "$KEY" scripts/pod_target_setup.sh harness/python/adapters/qwen_worker.py "$req" "root@$ip:/root/" \
  && p 'bash /root/pod_target_setup.sh 2>&1 | grep -E "^[0-9a-f]{64}.*model.safetensors|rror"; cp /root/qwen_worker.py /root/openwish/; nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null' | sed "s/^/  /"
p "mkdir -p /root/bench/edits && cd /root/openwish && s=\$(date +%s); env -u GH_RO_TOKEN /root/vllm-env/bin/python qwen_worker.py generate --model /root/openwish/model --edits /root/bench/edits --req /root/$(basename "$req") --out /root/bench/out --practice /root/openwish/practice --work /root/bench/work > /root/bench/log 2>&1; echo exit=\$? sec=\$((\$(date +%s)-s))" | tee "logs/bench_$label.txt"
p "grep '^{' /root/bench/log | tail -n 1 | cut -c1-1500; grep -iE 'error|Traceback' /root/bench/log | tail -n 3" >> "logs/bench_$label.txt" 2>&1   # the worker prints its summary as one JSON line
cost=$(runpodctl pod get "$id" | jq -r .costPerHr)
runpodctl pod delete "$id" >/dev/null 2>&1; sleep 5
runpodctl pod list | jq -e --arg p "$id" 'any(.[]; .id == $p)' >/dev/null && say "ALERT pod $id still listed" || say "pod deleted"
say "RESULT $(head -n 1 "logs/bench_$label.txt") \$/h=$cost"
