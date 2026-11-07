#!/usr/bin/env bash
# Full no-sudo setup of a new WSL/Linux harness node for this project (what the desktop got by hand on 2026-11-06):
# uv + python3, bubblewrap (apt download), Rust, the exp8 venvs/assets (setup_exp8_wsl.sh), hold-out data, the binary.
# Idempotent. Run from the project root on the node, e.g. inside tmux: bash scripts/setup_node_full.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH" HF_HUB_DISABLE_XET=1
step() { echo; echo "[$(date +%H:%M:%S)] == $*"; }

step "uv and a default python3"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12 --default >/dev/null 2>&1 || true

step "bubblewrap (user-local, no sudo)"
if ! command -v bwrap >/dev/null; then
  mkdir -p ~/.local/pkgs ~/.local/bin && (cd ~/.local/pkgs && apt download bubblewrap >/dev/null && dpkg-deb -x bubblewrap_*.deb bwrap-root)
  ln -sf ~/.local/pkgs/bwrap-root/usr/bin/bwrap ~/.local/bin/bwrap
fi
bwrap --unshare-all --die-with-parent --ro-bind / / /bin/echo "bwrap works"

step "Rust"
command -v cargo >/dev/null || curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null
cargo --version

step "exp8 venvs, Qwen3.5-0.8B (pinned, hash-checked), prepared assets, published edits"
bash scripts/setup_exp8_wsl.sh

step "FlashInfer kernels for vLLM (no nvcc)"
[ -n "${NO_GPU:-}" ] || (export PATH="$PWD/harness/assets/venvs/qwen/bin:$PATH"; vllm download-kernels >/dev/null 2>&1 || true)

step "hold-out benchmark data"
HV=qwen; [ -n "${NO_GPU:-}" ] && HV=sandbox   # the prepare step needs no GPU
(cd harness && VIRTUAL_ENV=assets/venvs/$HV uv pip install -q langdetect nltk immutabledict absl-py pyarrow huggingface_hub \
  && assets/venvs/$HV/bin/python -I python/adapters/holdout_eval.py prepare | tail -n 2)

step "build the harness"
(cd harness && nice -n 10 cargo build --release -j 8 2>&1 | tail -n 1)

step "checks"
sha256sum harness/assets/exp8/qwen35-text/model.safetensors harness/assets/exp8/eval/math500.jsonl
(cd harness && cargo test --release -p openwish-core 2>&1 | grep "test result" | head -n 1)
step "done"
