#!/usr/bin/env bash
# One-off (2026-11-13): DS hinted s2 ended with a final answer (turn 400), but its MATH-500 scoring ran on the pod that
# had lost its GPU (score 0.0 with math500.error: not a result). Once rescue_dsh2.sh has swapped the run's pool to a
# new pod: move the failed eval aside (eval_failed_deadgpu/), re-score with `openwish score` on the laptop (where the run
# lives), then run post_episode.sh, which does the hold-out on a repair and deletes the pod. Mac side.
set -u
cd "$(dirname "$0")/.."
run=runs_laptop/phase0/deepseekflash-hinted-p50-strict2-s2; remote=scratch_remote_ds_strict2_s2.json
sess=dsx2h2; name=ds_strict2_s2; rlog=logs/rescore_dsh2.log
say() { echo "$(date +%H:%M) rescore-dsh2 $*"; }
until grep -q -E 'helpers re-attached|ALERT' logs/rescue_dsh2.log 2>/dev/null; do sleep 60; done
grep -q 'helpers re-attached' logs/rescue_dsh2.log || { say "ALERT rescue failed; not re-scored"; exit 1; }
[ -d "harness/$run/eval_failed_deadgpu" ] || mv "harness/$run/eval" "harness/$run/eval_failed_deadgpu"
sleep 60   # the Mutagen sync carries the move to the laptop
rx -n laptop "test -e harness/$run/eval && echo still || echo moved" 2>/dev/null | tail -n 1 | grep -qx moved \
  || { say "ALERT the laptop still sees harness/$run/eval"; exit 1; }
rx -n laptop "mkdir -p logs && tmux new -d -s rescore-dsh2 'cd harness && OPENWISH_QWEN_REMOTE=\$(cd .. && pwd)/$remote target/release/openwish score $run > ../$rlog 2>&1; echo RESCORE_EXIT=\$? >> ../$rlog'" > /dev/null 2>&1
say "re-score started (laptop tmux rescore-dsh2, $rlog)"
until grep -q "RESCORE_EXIT=" "$rlog" 2>/dev/null; do sleep 300; done
say "re-score done ($(grep RESCORE_EXIT "$rlog")): $(jq -c '{score, n_edited: (.edited_tensors|length), err: .math500.error}' "harness/$run/eval/score.json" 2>/dev/null | cut -c1-200)"
# post_episode: hold-out on a repair, then deletes the pod (its pod id from the pool file's host line)
podid=$(grep -o 'pod [a-z0-9]* ready' logs/rescue_dsh2.log | tail -n 1 | cut -d' ' -f2)
pgrep -f "[p]ost_episode.sh $sess " >/dev/null && { say "post_episode already running"; exit 0; }
NODE=laptop bash scripts/post_episode.sh $sess $run $podid $remote >> logs/post_$name.log 2>&1 < /dev/null
say "post_episode exited ($(tail -n 1 logs/post_$name.log))"
