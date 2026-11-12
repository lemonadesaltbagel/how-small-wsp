#!/usr/bin/env bash
# One-off (2026-11-13): DS hinted s2's only target pod (v0sybot950jx57, community 3090) lost its GPU ("Failed to
# initialize NVML") during forward pass 48, whose worker hangs. Mac side, in order:
# 1. wait until the truncation chain has its pod (acquisitions read the last line of logs/pool_hosts.jsonl), rent one
# 2. swap the episode's pool to it (swap_pool.sh; the next pass reads the new host)
# 3. end the hung worker on the old pod: pass 48 fails (a failed pass is not charged) and the editor retries
# 4. re-attach post_episode / auto-resume / low-balance stop with the new pod id (the running ones list the old one only)
# 5. delete the old pod once the laptop holds no ssh to it; record its host in logs/bad_hosts.txt
# 6. run the DS unhinted s3 launcher (start_dsu3.sh), which waited for the same reason
set -u
cd "$(dirname "$0")/.."
K=~/.runpod/ssh/runpodctl-ssh-key
old=v0sybot950jx57; oip=64.119.209.250; oport=16702
run=runs_laptop/phase0/deepseekflash-hinted-p50-strict2-s2; remote=scratch_remote_ds_strict2_s2.json
sess=dsx2h2; name=ds_strict2_s2; task=tasks/exp8sl/hinted-p50-strict2-claudegate.toml
say() { echo "$(date +%H:%M) dsh2-rescue $*"; }
stop_pid() {  # stop_pid <pid> <pattern its command must match>
  ps -o command= -p "$1" 2>/dev/null | grep -q -- "$2" && kill "$1" && say "stopped $1 ($2)"; }

grep -q "^$oip " logs/bad_hosts.txt 2>/dev/null || echo "$oip # lost its GPU (NVML) mid-episode, 2026-11-13" >> logs/bad_hosts.txt
until grep -q -E 'pod .* ready|ALERT' logs/chain_trunc.log 2>/dev/null; do sleep 60; done
n0=$(wc -l < logs/pool_hosts.jsonl)
for attempt in 1 2 3 4 5 6 7 8; do
  CLOUD=COMMUNITY DISK_GB=40 GPU_LIST="NVIDIA GeForce RTX 5090:1.79;NVIDIA GeForce RTX 4090:1.01;NVIDIA RTX A5000:0.77;NVIDIA GeForce RTX 3090:0.94" \
    bash scripts/acquire_pods.sh 1 30 >> logs/acquire_dsh2.log 2>&1 < /dev/null
  n1=$(wc -l < logs/pool_hosts.jsonl)
  [ "$n1" -gt "$n0" ] && break
  say "acquire attempt $attempt gave no ready pod"; sleep 60
done
[ "$n1" -gt "$n0" ] || { say "ALERT no pod acquired (logs/acquire_dsh2.log); s3 launcher not started"; exit 1; }
h=$(tail -n 1 logs/pool_hosts.jsonl); pod=$(echo "$h" | jq -r .pod)
say "pod $pod ready ($(echo "$h" | jq -r .gpu))"
bash scripts/swap_pool.sh laptop "$remote" - "$h" || { say "ALERT swap failed; pod $pod kept"; exit 1; }
sleep 45   # the Mutagen sync carries the pool file to the laptop
nh=$(echo "$h" | jq -r .host)   # the laptop has no jq
rx -n laptop "grep -cF '$nh' $remote; grep -cF '$oip' $remote" 2>/dev/null | tail -n 2 | tr '\n' ' ' | grep -q '^1 0 ' \
  || { say "ALERT laptop does not see the new pool yet; old worker left running"; exit 1; }

stop_pid 28990 "post_episode.sh $sess"
stop_pid 26726 "auto_resume_limit.sh laptop $sess"
NODE=laptop nohup bash scripts/post_episode.sh $sess $run $pod $remote >> logs/post_$name.log 2>&1 < /dev/null &
nohup bash scripts/auto_resume_limit.sh laptop $sess $run $remote $pod $name $task DS "" >> logs/auto_resume_$sess.log 2>&1 < /dev/null &
sed -e "/dsux2 /d" -e "s/$old/$pod/" scripts/low_balance_stop.sh > scripts/low_balance_stop_dsh2.sh
nohup bash scripts/low_balance_stop_dsh2.sh > logs/low_balance_stop_dsh2.log 2>&1 < /dev/null &
say "helpers re-attached with pod $pod"

ssh -o ConnectTimeout=20 -p $oport -i $K root@$oip \
  'for p in $(pgrep -f "^/root/vllm-env/bin/python /root/openwish/qwen_worker.py") $(pgrep -f "^VLLM::EngineCore"); do kill -9 $p; done; echo killed' < /dev/null
quiet=0; t0=$(date +%s)
while [ $quiet -lt 2 ]; do
  n=$(rx -n laptop "pgrep -fc -- '[-]p $oport .*root@$oip' || true" 2>/dev/null | tail -n 1)
  if [ "${n:-1}" = 0 ]; then quiet=$((quiet + 1)); else quiet=0; fi
  [ $(( $(date +%s) - t0 )) -gt 1800 ] && { say "ALERT laptop still holds ssh to $old after 30 min; old pod kept"; break; }
  sleep 60
done
if [ $quiet -ge 2 ]; then
  runpodctl pod delete "$old" >/dev/null 2>&1; sleep 5
  runpodctl pod list 2>/dev/null | jq -e --arg p "$old" 'any(.[]; .id == $p)' >/dev/null && say "ALERT pod $old still listed" || say "old pod $old deleted"
fi
say "starting the s3 launcher"
bash scripts/start_dsu3.sh >> logs/start_dsu3.log 2>&1
