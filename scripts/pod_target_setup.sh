#!/usr/bin/env bash
# Run ON a pod: prepare the exp8 target-model worker files under /root/openwish (the layout qwen_pass.py's remote
# mode expects: model/, practice/, eval/, qwen_worker.py) by downloading everything on the pod (home upload is slow).
# Installs /root/vllm-env (vLLM) if missing. Prints sha256 of the text checkpoint and eval sets; compare with WSL before use:
#   68194c7beef8da37beaddb33b02a3b076f0930d81b55808be59998ba10cfc9fb  model.safetensors
#   35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132  math500.jsonl
set -euo pipefail
R=/root/openwish
REV=2fc06364715b967f1860aea9cf38778875588b17
export PATH=/root/.local/bin:/root/vllm-env/bin:$PATH
mkdir -p $R/model $R/practice $R/eval $R/src
if [ ! -x /root/vllm-env/bin/vllm ]; then
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  uv venv -q -p 3.12 /root/vllm-env && VIRTUAL_ENV=/root/vllm-env uv pip install -q vllm
fi
command -v rsync >/dev/null || { apt-get update -qq >/dev/null; apt-get install -y -qq rsync >/dev/null; }
VIRTUAL_ENV=/root/vllm-env uv pip install -q transformers safetensors math-verify huggingface_hub numpy
[ -d $R/src/lm_understand-logs ] || git clone -q --depth 1 https://github.com/RikiyaT/lm_understand-logs.git $R/src/lm_understand-logs
L=$R/src/lm_understand-logs/exp8_math
[ -f $R/src/hf/model.safetensors-00001-of-00001.safetensors ] || HF_HUB_DISABLE_XET=1 hf download Qwen/Qwen3.5-0.8B --revision $REV --local-dir $R/src/hf >/dev/null
# the text-only checkpoint, as OpenWiSH's prepare_exp8.text_checkpoint builds it
python - <<'EOF'
import json, struct, shutil, os, numpy as np
R = "/root/openwish"; L = f"{R}/src/lm_understand-logs/exp8_math"; out = f"{R}/model"
src = f"{R}/src/hf/model.safetensors-00001-of-00001.safetensors"
if not os.path.exists(f"{out}/model.safetensors"):
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; header = json.loads(f.read(n)); header.pop("__metadata__", None); off0 = 8 + n
    keep = sorted(k for k in header if k.startswith("model.language_model."))
    raw = np.memmap(src, dtype=np.uint8, mode="r", offset=off0)
    hdr, chunks, off = {}, [], 0
    for k in keep:
        a, b = header[k]["data_offsets"]; buf = bytes(raw[a:b])
        hdr[k] = {"dtype": header[k]["dtype"], "shape": header[k]["shape"], "data_offsets": [off, off + len(buf)]}
        chunks.append(buf); off += len(buf)
    hdr["__metadata__"] = {"format": "pt"}
    h = json.dumps(hdr, separators=(",", ":")).encode(); h += b" " * (-len(h) % 8)
    with open(f"{out}/model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(h))); f.write(h); [f.write(c) for c in chunks]
shutil.copyfile(f"{L}/base_model/Qwen3.5-0.8B-text/config.json", f"{out}/config.json")
for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
    s = f"{L}/base_model/Qwen3.5-0.8B-text/{name}"
    s = s if os.path.exists(s) else f"{R}/src/hf/{name}"
    if os.path.exists(s): shutil.copyfile(s, f"{out}/{name}")
EOF
cp -r $L/practice/. $R/practice/
[ -f $R/eval/math500.jsonl ] || curl -sfL -o $R/eval/math500.jsonl https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl
[ -f $R/eval/gsm8k_test.jsonl ] || curl -sfL -o $R/eval/gsm8k_test.jsonl https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
vllm download-kernels >/dev/null 2>&1 || true
sha256sum $R/model/model.safetensors $R/eval/*.jsonl
