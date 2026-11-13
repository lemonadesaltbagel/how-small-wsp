"""Weight check for the cf48 counterfactual (scripts/make_cf48.py), which has no final/editset: expected = base + the
DS unhinted s2 submitted edit set with layer-23 down_proj replaced by the #48 blob; loaded = base with the tensors in
the 4-sample score record (logs/rescore_cf48.log) replaced from cf48/final/net. Same checks as verify_scored_weights.py.
  harness/assets/venvs/sandbox/bin/python -I scripts/verify_cf48.py"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_scored_weights import BASE_ST, H, ROOT, fname, model_hash, read_safetensors, to_stored  # noqa: E402

SRC = os.path.join(H, "runs_laptop/phase0/deepseekflash-unhinted-p50-strict2-s2")
DST = SRC + "-cf48"
T = "model.language_model.layers.23.mlp.down_proj.weight"
cf = json.load(open(os.path.join(DST, "cf48.json")))
st = read_safetensors(BASE_ST)
m = json.load(open(os.path.join(SRC, "final", "editset", "manifest.json")))
ed = np.load(os.path.join(SRC, "final", "editset", "edit.npz"))
src = {k: ed[k] for k in m["tensors"]}
assert T in src, "down_proj not in the submitted edit set"
src[T] = np.load(os.path.join(SRC, "edits", "blobs", cf["down_proj_blob"]))
expected = {k: to_stored(v, st[k][0]) for k, v in src.items()}
rec = json.loads([l for l in open(os.path.join(ROOT, "logs", "rescore_cf48.log")) if l.startswith("{")][-1])
loaded = {k: to_stored(np.load(os.path.join(DST, "final", "net", fname(k))), st[k][0]) for k in rec["edited_tensors"]}
h_exp, h_load = model_hash(st, expected), model_hash(st, loaded)
changed = {k: int((expected[k] != st[k][2]).sum()) for k in expected}
numel = sum(int(np.prod(v[1])) for v in st.values())
out = {"run": os.path.relpath(DST, H), "record": "4-sample score", "declared_tensors": len(src),
       "n_loaded": len(loaded), "sha_expected": h_exp[:16], "sha_loaded": h_load[:16], "ok": h_exp == h_load,
       "loaded_not_declared": sorted(k for k in loaded if k not in expected),
       "declared_not_loaded_bf16_changed": sorted(k for k in expected if changed[k] and k not in loaded),
       "changed_bf16_bits": sum(changed.values()), "share_bf16_bits": sum(changed.values()) / numel}
print(json.dumps(out))
