#!/bin/bash
# Event stream for a Monitor: episode milestones (every 5 passes, new edits, errors, end) and conformance results.
#   progress_monitor.sh <episode log>... (logs/opus_s1.log ...)      (bash 3.2 compatible: no associative arrays)
cd "$(dirname "$0")/.."
st=$(mktemp -d)
field() { echo "$1" | sed -n "s/.*$2 \([0-9]*\)[\/ ].*/\1/p"; }
for f in "$@"; do
  k=$(basename "$f" .log); L=$(tail -n 1 "$f" 2>/dev/null)
  wc -l < "$f" | tr -d ' ' > "$st/$k.n"; field "$L" forward > "$st/$k.p"; field "$L" edits > "$st/$k.e"
done
b=$(grep -c config logs/conformance.log 2>/dev/null || echo 0)
while true; do
  for f in "$@"; do
    k=$(basename "$f" .log); n=$(cat "$st/$k.n"); c=$(wc -l < "$f" | tr -d ' ')
    if [ "$c" -gt "$n" ]; then
      tail -n +$(( n + 1 )) "$f" | grep -E -i "error|fatal|refused|transient|done:|panick|restart|cap reached" | sed "s|^|$k: |" | cut -c1-250
      L=$(tail -n 1 "$f"); p=$(field "$L" forward); e=$(field "$L" edits)
      op=$(cat "$st/$k.p"); oe=$(cat "$st/$k.e")
      if [ -n "$p" ] && { [ $(( p / 5 )) -ne $(( ${op:-0} / 5 )) ] || [ "$e" != "$oe" ]; }; then
        echo "$k: $L"; echo "$p" > "$st/$k.p"; echo "$e" > "$st/$k.e"
      fi
      echo "$c" > "$st/$k.n"
    fi
  done
  cb=$(grep -c config logs/conformance.log 2>/dev/null || echo 0)
  if [ "$cb" -gt "$b" ]; then
    grep config logs/conformance.log | tail -n +$(( b + 1 )) | cut -c1-300
    echo "conformance elapsed_min=$(( ($(date +%s) - $(cat logs/conformance.start)) / 60 ))"; b=$cb
  fi
  sleep 60
done
