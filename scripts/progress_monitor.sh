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
  cb=$(grep -c config logs/conformance.log 2>/dev/null || echo 0)
  if [ "$cb" -gt "$b" ]; then
    grep config logs/conformance.log | tail -n +$(( b + 1 )) | cut -c1-300
    echo "conformance elapsed_min=$(( ($(date +%s) - $(cat logs/conformance.start)) / 60 ))"; b=$cb; echo "$b" > "$st/conformance.n"
  fi
  sleep 60
done
