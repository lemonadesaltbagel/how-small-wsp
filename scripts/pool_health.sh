#!/usr/bin/env bash
# Mac side: torch CUDA check on every live pod listed in logs/pool_hosts.jsonl; prints "BAD <pod> <host:port> <error>"
# for each pod whose GPU torch cannot use (community hosts lose GPUs; old drivers), nothing for healthy pods.
set -uo pipefail
cd "$(dirname "$0")/.."
live=$(runpodctl pod list 2>/dev/null | jq -r '.[].id') || exit 0
jq -r '"\(.pod) \(.host) \(.port)"' logs/pool_hosts.jsonl | sort -u | while read -r pod h p; do
  grep -qx "$pod" <<< "$live" || continue
  r=$(ssh -o ConnectTimeout=10 -o BatchMode=yes -i ~/.runpod/ssh/runpodctl-ssh-key -p "$p" "$h" \
      '/root/vllm-env/bin/python -c "import torch; torch.zeros(1).cuda(); print(\"gpu-ok\")" 2>&1 | tail -n 1' < /dev/null 2>&1)
  case "$r" in *gpu-ok*) ;; *) echo "BAD $pod ${h#root@}:$p $(cut -c1-120 <<< "$r")";; esac
done
