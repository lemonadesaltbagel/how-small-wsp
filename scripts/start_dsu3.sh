#!/usr/bin/env bash
# One-off (2026-11-13): DS unhinted s1 failed (6.8 %), so seed 3 of the cell. Waits until the truncation-control chain
# has its pod (both read the last line of logs/pool_hosts.jsonl), rents one target pod, starts the episode on the
# laptop (tmux dsux3) and attaches post_episode.sh, auto-resume and a low-balance stop for this run. Mac side.
set -u
cd "$(dirname "$0")/.."
run=runs_laptop/phase0/deepseekflash-unhinted-p50-strict2-s3; remote=scratch_remote_dsu_strict2_s3.json
task=tasks/exp8sl/unhinted-p50-strict2-claudegate.toml; name=dsu_strict2_s3; sess=dsux3
say() { echo "$(date +%H:%M) dsux3 $*"; }
until grep -q -E 'pod .* ready|ALERT' logs/chain_trunc.log 2>/dev/null; do sleep 60; done
[ -e "harness/$run" ] && { say "ALERT harness/$run exists already"; exit 1; }
n0=$(wc -l < logs/pool_hosts.jsonl)
for attempt in 1 2 3 4 5 6 7 8; do
  CLOUD=COMMUNITY DISK_GB=40 GPU_LIST="NVIDIA GeForce RTX 5090:1.79;NVIDIA GeForce RTX 4090:1.01;NVIDIA RTX A5000:0.77;NVIDIA GeForce RTX 3090:0.94" \
    bash scripts/acquire_pods.sh 1 30 >> logs/acquire_dsu3.log 2>&1 < /dev/null
  n1=$(wc -l < logs/pool_hosts.jsonl)
  [ "$n1" -gt "$n0" ] && break
  say "acquire attempt $attempt gave no ready pod"; sleep 60
done
[ "$n1" -gt "$n0" ] || { say "ALERT no pod acquired (logs/acquire_dsu3.log)"; exit 1; }
h=$(tail -n 1 logs/pool_hosts.jsonl); pod=$(echo "$h" | jq -r .pod); echo "{\"hosts\":[$h]}" > "$remote"
say "pod $pod ready ($(echo "$h" | jq -r .gpu))"
avail=$(rx -n laptop "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo" 2>/dev/null | tail -n 1)
[ "${avail:-0}" -ge 9000 ] || { say "ALERT laptop MemAvailable ${avail:-?} MB < 9000; pod $pod kept, episode not started"; exit 1; }
rx -n laptop "mkdir -p logs && tmux new -d -s $sess 'bash scripts/launch_ds_episode.sh $task $run $remote $name'" > /dev/null 2>&1
sleep 60
rx -n laptop "tmux has-session -t $sess 2>/dev/null && echo yes" 2>/dev/null | grep -qx yes || { say "ALERT $sess not running after start; pod $pod kept"; exit 1; }
say "episode started (laptop tmux $sess, logs/$name.log)"
NODE=laptop nohup bash scripts/post_episode.sh $sess $run $pod $remote > logs/post_$name.log 2>&1 < /dev/null &
nohup bash scripts/auto_resume_limit.sh laptop $sess $run $remote $pod $name $task DS "" > logs/auto_resume_$sess.log 2>&1 < /dev/null &
sed "s/POD_DSU3/$pod/" scripts/low_balance_stop_dsu3.tmpl > scripts/low_balance_stop_dsu3.sh
nohup bash scripts/low_balance_stop_dsu3.sh > logs/low_balance_stop_dsu3.log 2>&1 < /dev/null &
say "helpers attached (post, auto-resume, low-balance stop)"
