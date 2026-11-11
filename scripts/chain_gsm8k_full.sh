#!/usr/bin/env bash
# One-off (2026-11-12): after chain_rescore_control.sh is DONE, run the full GSM8K base on the same pod (wsl tmux).
cd "$(dirname "$0")/.."
until grep -q "chain DONE" logs/chain_rescore_control.log 2>/dev/null; do sleep 120; done
echo "$(date +%H:%M) gsm8k-full starting"
rx -n wsl "tmux new -d -s gsm8kfull 'cd harness && OPENWISH_QWEN_REMOTE=\$(cd .. && pwd)/scratch_remote_rescore_dsh1.json assets/venvs/sandbox/bin/python -I ../scripts/base_gsm8k_full.py > ../logs/base_gsm8k_full.log 2>&1; echo GSM8K_EXIT=\$? >> ../logs/base_gsm8k_full.log'"
echo "$(date +%H:%M) gsm8k-full started (wsl tmux gsm8kfull)"
