#!/usr/bin/env bash
# Self-report vs log check (A3) for finished runs, Codex.  bash scripts/selfreport_batch.sh <run dir under harness/>...
cd "$(dirname "$0")/.."
export PATH=$HOME/.nvm/versions/node/v22.23.4/bin:$HOME/.local/bin:$PATH
for r in "$@"; do
  n=$(basename "$r"); o=logs/selfreport/$n.json
  [ -s "$o" ] && continue
  python3 -I scripts/self_report_check.py "harness/$r" > "$o.tmp" 2> "logs/selfreport/$n.err" && mv "$o.tmp" "$o" || echo "FAIL $n"
done
echo SELFREPORT_DONE
