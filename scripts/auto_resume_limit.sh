#!/usr/bin/env bash
# Resume an episode that stopped because its editor backend kept failing (Claude usage limit, a dropped SSH tunnel to
# a vLLM editor, an API outage), so its pool pods do not sit idle until someone notices. Runs on the Mac:
#   nohup bash scripts/auto_resume_limit.sh <node> <session> <run dir under harness/> <remote.json> <pods> <log name> \
#     <task under harness/> "<env assignments>" "<openwish flags>" > logs/auto_resume_<session>.log 2>&1 < /dev/null &
# Acts only when the session is gone AND the episode log's last line is "Error: backend kept failing ..."; any other stop
# is left for a human. A usage-limit stop waits for the reset time (+2 min), any other backend failure 5 min. Restarts
# post_episode.sh if it exited. Exits when the episode has a score, or after 8 resumes.
set -u
cd "$(dirname "$0")/.."
node=$1; sess=$2; run=$3; remote=$4; pods=$5; name=$6; task=$7; envs=$8; flags=$9
log="logs/$name.log"; post="logs/post_$name.log"
say() { echo "$(date '+%m-%d %H:%M') $sess $*"; }
alive() {  # yes | no | unknown (unreachable node)
  a=$(rx -n "$node" "tmux has-session -t $sess 2>/dev/null && echo yes || echo no" 2>/dev/null | tail -n 1)
  case "$a" in yes|no) echo "$a";; *) echo unknown;; esac; }
# "resets 10:10am" / "resets 3pm" (Australia/Sydney = this Mac's zone) -> epoch seconds of the next such time
reset_epoch() {
  t=$(grep -o 'resets [0-9:]*[ap]m' <<< "$1" | tail -n 1 | cut -d' ' -f2)
  [ -z "$t" ] && return 1
  case "$t" in *:*) ;; *) t="${t%??}:00${t: -2}";; esac  # 3pm -> 3:00pm
  hm=$(date -j -f '%I:%M%p' "$(tr a-z A-Z <<< "$t")" '+%H:%M') || return 1
  e=$(date -j -f '%Y-%m-%d %H:%M:%S' "$(date '+%Y-%m-%d') $hm:00" '+%s') || return 1
  # a reset is at most 5 h ahead: a time long past means tomorrow; a recent past one means resume now
  [ "$e" -lt "$(( $(date +%s) - 43200 ))" ] && e=$(( e + 86400 ))
  echo "$e"; }

say "watching for backend-failure stops"
n=0
while true; do
  [ -f "harness/$run/eval/score.json" ] && { say "scored; exiting"; exit 0; }
  last=$(grep -v '^[[:space:]]*$' "$log" 2>/dev/null | tail -n 1)
  if grep -q '^Error: backend kept failing' <<< "$last" && [ "$(alive)" = no ]; then
    if grep -q '5-hour limit' <<< "$last"; then
      e=$(reset_epoch "$last") || { say "ALERT could not parse reset time: $last"; exit 1; }
      at=$(( e + 120 ))
    else
      at=$(( $(date +%s) + 300 ))
    fi
    wait_s=$(( at - $(date +%s) )); [ "$wait_s" -lt 0 ] && wait_s=0
    say "backend stop ($(cut -c1-120 <<< "$last")); resuming at $(date -r "$at" '+%H:%M')"
    sleep "$wait_s"
    [ "$(alive)" = no ] || { say "session already back (resumed by hand?)"; continue; }
    rx -n "$node" "tmux new -d -s $sess '$envs bash scripts/run_episode.sh $task $run $remote $flags 2>&1 | tee -a $log'" >/dev/null 2>&1
    sleep 30
    if [ "$(alive)" = yes ]; then n=$((n + 1)); say "resumed ($n)"; else say "ALERT resume failed"; exit 1; fi
    if ! pgrep -f "[p]ost_episode.sh $sess " >/dev/null; then
      NODE=$node nohup bash scripts/post_episode.sh "$sess" "$run" "$pods" "$remote" >> "$post" 2>&1 < /dev/null &
      say "post_episode restarted"
    fi
    [ "$n" -ge 8 ] && { say "8 resumes; exiting"; exit 0; }
    sleep 600  # let the log move past the error line before looking again
  fi
  sleep 120
done
