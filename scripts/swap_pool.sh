#!/usr/bin/env bash
# Mac side: move a running episode to a new GPU pool between passes, then retire its old pods.
#   swap_pool.sh <node> <remote.json> <old pod ids, comma-separated or "-"> <host line json>... [--local]
# 1. writes {"hosts": [...]} atomically to <remote.json> (a pass reads it when it starts; the pass in flight keeps
#    its host); --local adds this node's GPU ({"local": true, "weight": 0.9}, used only with memory headroom)
# 2. deletes each old pod once no ssh from <node> to it has been seen in two checks 60 s apart.
set -uo pipefail
cd "$(dirname "$0")/.."
node=$1; remote=$2; old=$3; shift 3
local_gpu=0; hosts=()
for a in "$@"; do [ "$a" = --local ] && local_gpu=1 || hosts+=("$a"); done
say() { echo "$(date +%H:%M) swap $(basename "$remote") $*"; }
lines=$(printf '%s\n' "${hosts[@]}")
if [ $local_gpu = 1 ]; then lines="$lines
{\"local\": true, \"weight\": 0.9}"; fi
echo "$lines" | jq -s '{hosts: map(del(.pod, .gpu, .price))}' > "$remote.new" || { say "ALERT bad host lines; nothing changed"; exit 1; }
want=$(jq -r '.hosts | length' "$remote.new")
[ "$want" -ge 1 ] && [ "$want" = $(( ${#hosts[@]} + local_gpu )) ] || { say "ALERT pool has $want hosts; nothing changed"; exit 1; }
mv "$remote.new" "$remote"
# never retire anything unless the new pool is really in place
jq -e '.hosts | length >= 1' "$remote" >/dev/null || { say "ALERT pool not written; old pods kept"; exit 1; }
say "pool: $(jq -c '[.hosts[] | (.host // "local")]' "$remote")"
[ "$old" = - ] && exit 0
for p in ${old//,/ }; do
  read -r ip port < <(runpodctl pod get "$p" | jq -r '"\(.ssh.ip) \(.ssh.port)"')
  quiet=0
  while [ $quiet -lt 2 ]; do
    n=$(rx -n "$node" "pgrep -fc -- '[-]p $port .*root@$ip' || true" 2>/dev/null | tail -n 1)
    if [ "${n:-1}" = 0 ]; then quiet=$((quiet + 1)); else quiet=0; fi
    sleep 60
  done
  runpodctl pod delete "$p" >/dev/null 2>&1; sleep 5
  runpodctl pod list | jq -e --arg p "$p" 'any(.[]; .id == $p)' >/dev/null && say "ALERT pod $p still listed" || say "old pod $p deleted"
done
