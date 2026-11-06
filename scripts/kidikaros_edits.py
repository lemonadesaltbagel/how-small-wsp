"""Turn the KidIkaros scripted-solver checkpoint (HF: KidIkaros/qwen3.5-0.8b-loop-fix) into an exp8 edit folder:
one float32 .npy per language-model tensor that differs from our base text checkpoint (the format the qwen worker's
--edits takes). Run with the sandbox venv (numpy, huggingface_hub) from harness/:
  assets/venvs/sandbox/bin/python -I ../scripts/kidikaros_edits.py"""
import json
import os
import struct

import numpy as np
from huggingface_hub import snapshot_download

REPO, REV = "KidIkaros/qwen3.5-0.8b-loop-fix", "a1d725b3a2f53ea4b5a1cd37188a85039ea8dc04"
BASE = "assets/exp8/qwen35-text/model.safetensors"
OUT = "assets/exp8/published/kidikaros"


def tensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)
    return h, raw


def bf16(raw, v):
    a, b = v["data_offsets"]
    assert v["dtype"] == "BF16", v["dtype"]
    return np.frombuffer(bytes(raw[a:b]), dtype=np.uint16).reshape(v["shape"])


def main():
    d = snapshot_download(REPO, revision=REV, allow_patterns=["*.safetensors", "*.json", "README.md"])
    src = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".safetensors")][0]
    hb, rb = tensors(BASE)
    hk, rk = tensors(src)
    os.makedirs(OUT, exist_ok=True)
    changed = []
    for name, v in hb.items():
        if name not in hk:
            raise SystemExit(f"{name} missing from {REPO}")
        a, b = bf16(rb, v), bf16(rk, hk[name])
        if a.shape != b.shape:
            raise SystemExit(f"{name}: shape {a.shape} vs {b.shape}")
        n = int((a != b).sum())
        if n:
            np.save(os.path.join(OUT, name + ".npy"), (b.astype(np.uint32) << 16).view(np.float32))
            changed.append({"tensor": name, "entries_changed": n})
    total = sum(int(np.prod(v["shape"])) for v in hb.values())
    man = {"repo": REPO, "revision": REV, "changed": changed, "fraction_changed": sum(c["entries_changed"] for c in changed) / total}
    json.dump(man, open(os.path.join(OUT, "..", "kidikaros_manifest.json"), "w"), indent=1)
    print(json.dumps(man))


if __name__ == "__main__":
    main()
