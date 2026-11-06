#!/usr/bin/env bash
# One-time setup of the OpenWiSH exp8 environment on a Linux GPU machine (WSL node or pod).
# Creates harness/assets/{venvs,lm_understand-logs,exp8}. Idempotent: each step skips if done.
set -euo pipefail
cd "$(dirname "$0")/../harness"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
QWEN_REV=2fc06364715b967f1860aea9cf38778875588b17

step() { echo; echo "[$(date +%H:%M:%S)] == $*"; }

step "sandbox venv (CPU torch, numpy, tokenizers)"
if [ ! -x assets/venvs/sandbox/bin/python ]; then
  uv venv -q -p 3.12 assets/venvs/sandbox
  VIRTUAL_ENV=assets/venvs/sandbox uv pip install -q numpy scipy scikit-learn pandas sympy networkx safetensors minatar tokenizers huggingface_hub
  VIRTUAL_ENV=assets/venvs/sandbox uv pip install -q torch --index-url https://download.pytorch.org/whl/cpu
fi
assets/venvs/sandbox/bin/python -c "import torch, numpy, tokenizers; print('sandbox ok', torch.__version__, numpy.__version__)"

step "qwen venv (vLLM on the GPU)"
if [ ! -x assets/venvs/qwen/bin/python ]; then
  uv venv -q -p 3.12 assets/venvs/qwen
  VIRTUAL_ENV=assets/venvs/qwen uv pip install -q vllm transformers safetensors math-verify
fi
assets/venvs/qwen/bin/python -c "import vllm, transformers, torch; print('qwen ok vllm', vllm.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"

step "authors' logs (full clone, all blobs)"
if [ ! -d assets/lm_understand-logs/.git ]; then
  git clone -q --depth 1 https://github.com/RikiyaT/lm_understand-logs.git assets/lm_understand-logs
fi
echo "logs at $(git -C assets/lm_understand-logs rev-parse HEAD)"

step "Qwen3.5-0.8B @ $QWEN_REV"
if [ ! -f assets/exp8/qwen35-hf/config.json ]; then
  assets/venvs/sandbox/bin/hf download Qwen/Qwen3.5-0.8B --revision "$QWEN_REV" --local-dir assets/exp8/qwen35-hf
fi
ls -la assets/exp8/qwen35-hf

step "prepare_exp8 (text checkpoint, base_net, eval sets, prompts, task specs)"
assets/venvs/sandbox/bin/python scripts/prepare_exp8.py

step "rebuild the three published edits (sha256-checked)"
for r in p10c p25c p50c; do
  if [ ! -d "assets/exp8/published/$r" ]; then
    assets/venvs/sandbox/bin/python scripts/apply_published_edit.py assets/exp8/qwen35-text/model.safetensors \
      "assets/lm_understand-logs/exp8_math/runs/$r/edit" "assets/exp8/published/$r"
  fi
done
sha256sum assets/exp8/qwen35-text/model.safetensors assets/exp8/eval/*.jsonl
step "done"
