#!/usr/bin/env bash
# Mac-side pipeline after an episode: wait for its MATH-500 score, run the hold-out benchmarks of the final edit on the
# same target pod, then delete the pod. Replaces teardown_watch.sh for runs that need hold-out scores.
#   post_episode.sh <tmux session> <run dir rel. to harness/> <pod id> <remote.json rel. to repo>
# One line per state change on stdout (for a Monitor). If the session ends without a score: ALERT, pod kept.
set -uo pipefail
cd "$(dirname "$0")/.."
sess=$1; run=$2; pod=$3; remote=$4
KEY=~/.runpod/ssh/runpodctl-ssh-key
say() { echo "$(date +%H:%M) $sess $*"; }
alive() { rx "tmux has-session -t $sess 2>/dev/null && echo yes || echo no" 2>/dev/null | tail -n 1; }
scored() { [ -f "harness/$run/eval/score.json" ] && echo yes || echo no; }
podssh() { read -r ip port < <(runpodctl pod get "$pod" 2>/dev/null | jq -r '"\(.ssh.ip) \(.ssh.port)"'); ssh -o ConnectTimeout=20 -p "$port" -i "$KEY" "root@$ip" "$@" < /dev/null; }

last=""
while true; do
  a=$(alive); s=$(scored); st="alive=$a scored=$s"
  [ "$st" != "$last" ] && say "$st" && last=$st
  [ "$a" = no ] && [ "$s" = yes ] && break
  if [ "$a" = no ] && [ "$s" = no ]; then say "ALERT ended without a score; pod $pod kept for --resume"; exit 1; fi
  sleep 300
done
say "MATH-500 score $(jq -r '.score' "harness/$run/eval/score.json")"

# hold-out data and dependencies on the pod (small files; the model and worker are there already)
podssh "export PATH=/root/.local/bin:\$PATH; VIRTUAL_ENV=/root/vllm-env uv pip install -q langdetect nltk immutabledict absl-py pyarrow && mkdir -p /root/openwish/holdout && echo deps-ok" | tail -n 1
read -r ip port < <(runpodctl pod get "$pod" | jq -r '"\(.ssh.ip) \(.ssh.port)"')
rx "for f in manifest.json mmlupro1k.jsonl gpqa.jsonl ifeval.jsonl; do rsync -az -e 'ssh -i ~/.ssh/id_ed25519_pods -p $port -o StrictHostKeyChecking=accept-new' harness/assets/exp8/holdout/\$f root@$ip:/root/openwish/holdout/\$f; done; rsync -az -e 'ssh -i ~/.ssh/id_ed25519_pods -p $port' harness/assets/exp8/holdout/nltk_data/ root@$ip:/root/openwish/holdout/nltk_data/ && echo data-ok" | tail -n 1
log="logs/holdout_$sess.log"
rx "tmux new -d -s ho-$sess 'cd harness && OPENWISH_RUN=$run OPENWISH_FINAL_NET=$run/final/net OPENWISH_QWEN_REMOTE=\$(cd .. && pwd)/$remote assets/venvs/sandbox/bin/python -I python/adapters/qwen_pass.py --evaluate --sets mmlupro1k,gpqa,ifeval > ../$log 2>&1; echo HOLDOUT_EXIT=\$? >> ../$log'"
say "hold-out started (tmux ho-$sess, $log)"
until grep -q "HOLDOUT_EXIT=" "$log" 2>/dev/null; do sleep 300; done
out=$(grep '^{' "$log" | tail -n 1)
echo "$out" > "harness/$run/eval/holdout_summary.json"
say "hold-out done ($(grep HOLDOUT_EXIT "$log")): $(echo "$out" | jq -c '{ok, mmlupro1k: .mmlupro1k.accuracy, gpqa: .gpqa.accuracy, ifeval: .ifeval.accuracy}' 2>/dev/null || echo "$out" | cut -c1-200)"

runpodctl pod delete "$pod" >/dev/null 2>&1; sleep 5
if runpodctl pod list 2>/dev/null | jq -e --arg p "$pod" 'any(.[]; .id == $p)' >/dev/null; then say "ALERT pod $pod still listed after delete"; exit 1; fi
say "DONE: pod $pod deleted"
