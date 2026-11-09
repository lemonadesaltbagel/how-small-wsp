#!/usr/bin/env bash
# Start (or resume) a deepseek-flash episode on this node (the laptop: its Claude Code runs the rule reviewer).
#   launch_ds_episode.sh <task under harness/> <run dir under harness/> <remote.json> <log name>
set -u
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
export OPENWISH_OPENAI_API_KEY=$DEEPSEEK_API OPENWISH_OPENAI_BASE_URL=https://api.deepseek.com
exec > >(tee -a "logs/$4.log") 2>&1
bash scripts/run_episode.sh "$1" "$2" "$3" --backend openai --model deepseek-flash --max-tokens 262144 --timeout 3600
