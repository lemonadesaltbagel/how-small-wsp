#!/usr/bin/env bash
# One-off (2026-11-11): after the DS hinted s1 4-sample rescore ends, run the thinking-off control on the same pod and
# fetch its result. Mac side.
set -u
cd "$(dirname "$0")/.."
K=~/.runpod/ssh/runpodctl-ssh-key; H=root@174.94.157.109; P=47449
say() { echo "$(date +%H:%M) chain $*"; }
until grep -q "RESCORE_EXIT=" logs/rescore_dsh1.log 2>/dev/null; do sleep 120; done
say "rescore ended: $(grep RESCORE_EXIT logs/rescore_dsh1.log)"
sleep 30
ssh -p $P -i $K $H 'pgrep -f "^/root/vllm-env/bin/python /root/openwish/qwen_worker.py" && echo worker-still-running; cd /root && nohup setsid /root/vllm-env/bin/python -I nothink_control.py items.json nothink_out.json > nothink.log 2>&1 < /dev/null & echo started' < /dev/null
say "control started"
until ssh -p $P -i $K $H 'test -f /root/nothink_out.json || ! pgrep -f "[n]othink_control.py" >/dev/null' < /dev/null; do sleep 120; done
mkdir -p notes/run_records/nothink_control
scp -q -P $P -i $K $H:/root/nothink_out.json $H:/root/nothink.log notes/run_records/nothink_control/ < /dev/null
say "control done: $(grep nothink_control notes/run_records/nothink_control/nothink.log | cut -c1-400)"
say "DONE"
