#!/usr/bin/env bash
# Relabel finished runs under amendment A3 with two labellers (Claude Sonnet 5.5 and Codex, blind to each other).
#   bash scripts/relabel_a3.sh <run dir under harness/>...   (on the desktop; outputs logs/labels_a3/<run>.<backend>.json)
cd "$(dirname "$0")/.."
export PATH=$HOME/.nvm/versions/node/v22.23.4/bin:$HOME/.local/bin:$PATH
for r in "$@"; do
  n=$(basename "$r")
  for b in claude codex; do
    o=logs/labels_a3/$n.$b.json
    [ -s "$o" ] && continue
    python3 -I scripts/label_failure.py "harness/$r" --backend $b > "$o.tmp" 2> "logs/labels_a3/$n.$b.err" && mv "$o.tmp" "$o" || echo "FAIL $n $b"
  done
done
echo RELABEL_DONE
