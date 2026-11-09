#!/usr/bin/env bash
# Mac side: get up to <count> cheap target GPUs (secure cloud; CLOUD=COMMUNITY for community cloud, approved for target
# pools 2026-11-09), retrying every 3 min until <minutes> pass. A pod whose setup fails is deleted. Each pod is
# set up (pod_target_setup.sh + hold-out deps), checked against the model hash, authorized for every node's key
# (desktop, laptop, mac: a pod can serve any of them), and appended as a host line to logs/pool_hosts.jsonl:
#   {"pod": id, "gpu": ..., "host": "root@ip", "port": p, "key": "~/.ssh/id_ed25519_pods", "root": "/root/openwish",
#    "python": "/root/vllm-env/bin/python", "gpu_util": 0.85, "weight": <memory bandwidth TB/s>, "price": $/h}
#   acquire_pods.sh <count> [minutes=120]
set -uo pipefail
cd "$(dirname "$0")/.."
want=$1; minutes=${2:-120}
KEY=~/.runpod/ssh/runpodctl-ssh-key
GPUS=("NVIDIA RTX A5000:0.77" "NVIDIA GeForce RTX 3090:0.94")
# GPU_LIST="id:TB/s;id:TB/s" overrides the list (e.g. add community 4090s when A5000/3090 stock is low)
[ -n "${GPU_LIST:-}" ] && IFS=';' read -ra GPUS <<< "$GPU_LIST"
MODEL_SHA=68194c7beef8da37beaddb33b02a3b076f0930d81b55808be59998ba10cfc9fb
keys="$(cat scripts/wsl_pod_key.pub)
$(ssh cn-laptop 'cat ~/.ssh/id_ed25519_pods.pub' < /dev/null)
$(ssh cn-mac 'cat ~/.ssh/id_ed25519_pods.pub' < /dev/null)"
say() { echo "$(date +%H:%M) acquire $*"; }
setup() {  # setup <id> <gpu> <weight>
  local id=$1 gpu=$2 w=$3 ip port sha
  read -r ip port < <(scripts/editor_pod.sh wait "$id") || { say "pod $id: no ssh; deleting"; runpodctl pod delete "$id" >/dev/null; return 1; }
  ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -p "$port" -i "$KEY" "root@$ip" "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys" <<< "$keys" \
    && scp -q -o StrictHostKeyChecking=accept-new -P "$port" -i "$KEY" scripts/pod_target_setup.sh "root@$ip:/root/" || { say "pod $id: ssh setup failed; deleting"; runpodctl pod delete "$id" >/dev/null; return 1; }
  sha=$(ssh -p "$port" -i "$KEY" "root@$ip" 'bash /root/pod_target_setup.sh 2>&1 | grep model.safetensors | cut -c1-64; export PATH=/root/.local/bin:$PATH; VIRTUAL_ENV=/root/vllm-env uv pip install -q langdetect nltk immutabledict absl-py pyarrow >/dev/null 2>&1' < /dev/null)
  [ "$sha" = "$MODEL_SHA" ] || { say "pod $id: model hash mismatch ($sha); deleting"; runpodctl pod delete "$id" >/dev/null; return 1; }
  printf '{"pod": "%s", "gpu": "%s", "host": "root@%s", "port": %s, "key": "~/.ssh/id_ed25519_pods", "root": "/root/openwish", "python": "/root/vllm-env/bin/python", "gpu_util": 0.85, "weight": %s, "price": %s}\n' \
    "$id" "$gpu" "$ip" "$port" "$w" "$(runpodctl pod get "$id" | jq -r .costPerHr)" >> logs/pool_hosts.jsonl
  say "ready $id $gpu $ip:$port"
}
got=0; end=$(( $(date +%s) + minutes * 60 )); n=$(wc -l < logs/pool_hosts.jsonl 2>/dev/null || echo 0)
while [ "$got" -lt "$want" ] && [ "$(date +%s)" -lt "$end" ]; do
  for g in "${GPUS[@]}"; do
    gpu=${g%%:*}; w=${g##*:}
    while [ "$got" -lt "$want" ]; do
      n=$((n + 1)); name="wsl-ladder-pool-$n-$$"  # unique per process: two loops once claimed the same pod by name
      out=$(scripts/editor_pod.sh create "$name" "$gpu" "${DISK_GB:-100}" 2>&1)
      id=$(echo "$out" | grep -E '^[a-z0-9]{10,}$')
      if [ -z "$id" ]; then
        # a create can succeed without printing a bare id (11-08 stray pod): look it up by name before moving on
        id=$(runpodctl pod list | jq -r --arg n "$name" '.[] | select(.name == $n) | .id' | head -n 1)
        [ -z "$id" ] && break
      fi
      echo "$(date +%Y-%m-%dT%H:%M) $name $id $gpu ${CLOUD:-SECURE}" >> logs/pods_strict.txt
      got=$((got + 1)); say "created $id $gpu ($got/$want)"
      setup "$id" "$gpu" "$w" &
    done
  done
  [ "$got" -lt "$want" ] && sleep 180
done
wait
say "done: $got created"
