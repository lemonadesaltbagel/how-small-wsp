#!/usr/bin/env bash
# Serve an open-weight editor model on a RunPod pod with vLLM (OpenAI-compatible API) and reach it from the
# harness machine (WSL) through an SSH tunnel. The harness itself never runs on the pod (bubblewrap cannot
# create namespaces inside RunPod containers; tested 2026-11-06).
#
#   editor_pod.sh create <name> <gpu-id> [disk-gb]      create a secure-cloud pod from the pytorch-gh template
#   editor_pod.sh wait <pod-id>                         print "ip port" once SSH is up
#   editor_pod.sh serve <pod-id> <hf-model> [max-len] [extra vllm args...]   install vLLM and start the server (background)
#   editor_pod.sh health <pod-id>                       0 when the server answers /v1/models
#   editor_pod.sh tunnel <pod-id> <local-port>          forward localhost:<local-port> -> pod:8000 (background, from WSL)
#   editor_pod.sh delete <pod-id>
set -euo pipefail
KEY=~/.runpod/ssh/runpodctl-ssh-key
TEMPLATE=ccucqzq7cf

addr() { runpodctl pod get "$1" 2>/dev/null | jq -r '"\(.ssh.ip // "") \(.ssh.port // "")"'; }
pssh() {  # pssh <pod-id> <command>
  read -r ip port < <(addr "$1"); shift
  ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30 -p "$port" -i "$KEY" "root@$ip" "$@" < /dev/null
}

case "$1" in
  create)
    runpodctl pod create --name "$2" --gpu-id "$3" --cloud-type SECURE --template-id "$TEMPLATE" \
      --container-disk-in-gb "${4:-150}" --ports "22/tcp" | jq -r '.id' ;;
  wait)
    for _ in $(seq 1 90); do
      read -r ip port < <(addr "$2")
      if [ -n "$ip" ] && [ -n "$port" ]; then echo "$ip $port"; exit 0; fi
      sleep 10
    done
    echo "timeout waiting for ssh" >&2; exit 1 ;;
  serve)
    pod=$2; model=$3; maxlen=${4:-262144}; shift 4 || shift $#
    extra="$*"
    pssh "$pod" "mkdir -p ~/.ssh && grep -q wsl-wsp-ladder ~/.ssh/authorized_keys 2>/dev/null || echo '$(cat "$(dirname "$0")/wsl_pod_key.pub")' >> ~/.ssh/authorized_keys"
    pssh "$pod" "set -e; cd /root; if [ ! -x /root/vllm-env/bin/vllm ]; then command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; export PATH=/root/.local/bin:\$PATH; uv venv -q -p 3.12 /root/vllm-env; VIRTUAL_ENV=/root/vllm-env uv pip install -q vllm; fi; /root/vllm-env/bin/python -c 'import vllm; print(\"vllm\", vllm.__version__)'
      export PATH=/root/vllm-env/bin:\$PATH; vllm download-kernels >/dev/null 2>&1 || true
      nohup setsid env -u GH_RO_TOKEN vllm serve $model --served-model-name editor --max-model-len $maxlen --enable-prefix-caching \
        --reasoning-parser qwen3 --gpu-memory-utilization 0.92 --port 8000 $extra > /root/vllm.log 2>&1 < /dev/null &
      echo started" ;;
  health)
    pssh "$2" "curl -sf localhost:8000/v1/models >/dev/null" ;;
  tunnel)  # run from the Mac: opens the tunnel ON WSL (where the harness runs) with WSL's own pod key
    read -r ip port < <(addr "$2")
    rx "tmux kill-session -t tun-$3 2>/dev/null; tmux new -d -s tun-$3 'ssh -N -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -L $3:localhost:8000 -p $port -i ~/.ssh/id_ed25519_pods root@$ip'; sleep 3; curl -sf localhost:$3/v1/models | head -c 300; echo" ;;
  delete)
    runpodctl pod delete "$2" ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
