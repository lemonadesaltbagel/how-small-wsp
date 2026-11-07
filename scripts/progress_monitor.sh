#!/bin/bash
# Event stream for a Monitor: episode milestones (every 5 passes, first edit and every 10 edits, errors, end) and conformance results.
#   progress_monitor.sh <episode log>... (logs/opus_s1.log ...)      (bash 3.2 compatible: no associative arrays)
cd "$(dirname "$0")/.."
st=logs/.monitor_state   # persistent: a re-armed monitor resumes where the last one stopped
mkdir -p "$st"
field() { echo "$1" | sed -n "s/.*$2 \([0-9]*\)[\/ ].*/\1/p"; }
for f in "$@"; do
  k=$(basename "$f" .log)
  [ -f "$st/$k.n" ] || { echo 0 > "$st/$k.n"; echo 0 > "$st/$k.p"; echo 0 > "$st/$k.e"; }
done
[ -f "$st/conformance.n" ] || echo 0 > "$st/conformance.n"
b=$(cat "$st/conformance.n")
while true; do
  for f in "$@"; do
    k=$(basename "$f" .log); n=$(cat "$st/$k.n"); c=$(wc -l < "$f" | tr -d ' ')
    if [ "$c" -gt "$n" ]; then
      tail -n +$(( n + 1 )) "$f" | grep -E -i "error|fatal|refused|transient|done:|panick|restart|cap reached" | sed "s|^|$k: |" | cut -c1-250
      L=$(tail -n 1 "$f"); p=$(field "$L" forward); e=$(field "$L" edits)
      op=$(cat "$st/$k.p"); oe=$(cat "$st/$k.e")
      if [ -n "$p" ] && { [ $(( p / 5 )) -ne $(( ${op:-0} / 5 )) ] || { [ "${oe:-0}" = 0 ] && [ "${e:-0}" != 0 ]; } || [ $(( ${e:-0} / 10 )) -ne $(( ${oe:-0} / 10 )) ]; }; then
        echo "$k: $L"; echo "$p" > "$st/$k.p"; echo "$e" > "$st/$k.e"
      fi
      echo "$c" > "$st/$k.n"
    fi
  done
  # infrastructure failures in any phase-0 run (GPU worker / pod / engine errors are uncharged, so the status line
  # never shows them); the editor's own malformed requests are not counted
  for tj in harness/runs/phase0/*/tool_calls.jsonl harness/runs_laptop/phase0/*/tool_calls.jsonl; do
    r=$(basename "$(dirname "$tj")")
    v=$(grep -c -E "GPU worker failed|Engine core|not responding|No space left|Connection (refused|reset)|timed out" "$tj" 2>/dev/null)
    old=$(cat "$st/$r.infra" 2>/dev/null || echo "$v")
    if [ "$v" != "$old" ]; then echo "$r: infrastructure failures $old -> $v ($(grep -E "GPU worker failed|Engine core|not responding|No space left|Connection (refused|reset)|timed out" "$tj" | tail -n 1 | cut -c1-160))"; fi
    echo "$v" > "$st/$r.infra"
  done
  # post-episode pipelines (scripts/post_episode.sh): every new line is an event
  for pl in logs/post_*.log; do
    [ -f "$pl" ] || continue
    k=$(basename "$pl" .log); pn=$(wc -l < "$pl" | tr -d ' '); po=$(cat "$st/$k.n" 2>/dev/null || echo 0)
    [ "$pn" -gt "$po" ] && tail -n +$(( po + 1 )) "$pl" | sed "s/^/$k: /" | cut -c1-300
    echo "$pn" > "$st/$k.n"
  done
  # the local GPU queue (scripts/local_queue.sh)
  if [ -f logs/local_queue.log ]; then
    qn=$(wc -l < logs/local_queue.log | tr -d ' '); qo=$(cat "$st/localq.n" 2>/dev/null || echo "$qn")
    [ "$qn" -gt "$qo" ] && tail -n +$(( qo + 1 )) logs/local_queue.log | grep -E "==|WATCHDOG|\"config\"|Traceback|Error|fraction_changed" | sed 's/^/localq: /' | cut -c1-300
    echo "$qn" > "$st/localq.n"
  fi
  cb=$(grep -c config logs/conformance.log 2>/dev/null || echo 0)
  if [ "$cb" -gt "$b" ]; then
    grep config logs/conformance.log | tail -n +$(( b + 1 )) | cut -c1-300
    echo "conformance elapsed_min=$(( ($(date +%s) - $(cat logs/conformance.start)) / 60 ))"; b=$cb; echo "$b" > "$st/conformance.n"
  fi
  sleep 60
done
