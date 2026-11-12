"""Counterfactual for the DS unhinted s2 audit (pending_user_decisions.md, 2026-11-13 11:20): the submitted edit #49 is
W0 + 0.4*u2*(g - 0.12*dhat)^T on layer-23 down_proj, whose lam = 0.12 came from a note-(6) computation; #48 is the same
edit with lam = 0. Builds a run dir that scores #48: the ledger (so qwen_pass stages the same 19 tensors) and a
final/net holding those 19 tensors, all from the submitted final net except down_proj = the #48 blob. Checks that the
#49 blob equals the submitted tensor. On a node with the sandbox venv:
  harness/assets/venvs/sandbox/bin/python -I scripts/make_cf48.py"""
import json
import os
import shutil

import numpy as np

H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")
SRC = os.path.join(H, "runs_laptop/phase0/deepseekflash-unhinted-p50-strict2-s2")
DST = SRC + "-cf48"
T = "model.language_model.layers.23.mlp.down_proj.weight"
B48 = "2db01741d6f1310891be60cec13023a8d070e55bd2101b95441a4e114babc0b9.npy"
B49 = "599a716494f478983a039b5de072bc9209529eb740746c509c6b020712ebfbc4.npy"

led = [json.loads(l) for l in open(os.path.join(SRC, "edits/ledger.jsonl"))]
names = sorted({e["op"]["tensor"] for e in led if e.get("kind") == "edit"})
print("edited tensors in ledger:", len(names))
sub = np.load(os.path.join(SRC, "final/net", T + ".npy"))
b49 = np.load(os.path.join(SRC, "edits/blobs", B49))
b48 = np.load(os.path.join(SRC, "edits/blobs", B48))
print("dtypes", sub.dtype, b49.dtype, b48.dtype, "shapes", sub.shape, b48.shape)
assert np.array_equal(sub.astype(np.float32), b49.astype(np.float32)), "#49 blob != submitted tensor"
d = b48.astype(np.float32) - b49.astype(np.float32)
print("#48 - #49: max |d| %.3g, changed entries %d" % (np.abs(d).max(), int((d != 0).sum())))
os.makedirs(os.path.join(DST, "edits"), exist_ok=True)
os.makedirs(os.path.join(DST, "final/net"), exist_ok=True)
shutil.copy(os.path.join(SRC, "edits/ledger.jsonl"), os.path.join(DST, "edits/ledger.jsonl"))
for n in names:
    if n == T:
        np.save(os.path.join(DST, "final/net", n + ".npy"), b48.astype(sub.dtype))
    else:
        shutil.copy(os.path.join(SRC, "final/net", n + ".npy"), os.path.join(DST, "final/net", n + ".npy"))
json.dump({"what": "DS unhinted s2 submission with edit #48 (lam = 0) in place of #49", "down_proj_blob": B48,
           "tensors": names}, open(os.path.join(DST, "cf48.json"), "w"), indent=1)
print("written", DST)
