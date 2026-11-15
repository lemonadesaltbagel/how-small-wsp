"""Check every submitted edit in the extracted run data against its manifest: tensor names, shapes and the harness's
hash (openwish-tensor Tensor::sha256: ndim and dims as u64 LE, then the f32 LE data). Needs numpy.
  python -I scripts/check_release_edits.py [HARNESS_DIR]"""
import glob
import hashlib
import json
import os
import struct
import sys

import numpy as np


def sha(a):
    h = hashlib.sha256(struct.pack("<Q", a.ndim) + b"".join(struct.pack("<Q", d) for d in a.shape))
    h.update(np.ascontiguousarray(a, dtype="<f4").tobytes())
    return h.hexdigest()


def main(h):
    ok = bad = runs = 0
    for m in sorted(glob.glob(os.path.join(h, "runs*", "phase0", "*", "final", "editset", "manifest.json"))):
        man = json.load(open(m))
        z = np.load(m.replace("manifest.json", "edit.npz"))
        runs += 1
        for name, t in man["tensors"].items():
            a = z[name] if name in z.files else None
            if a is None or list(a.shape) != t["shape"] or sha(a) != t["sha_edited"]:
                print("mismatch", os.path.relpath(m, h), name)
                bad += 1
            else:
                ok += 1
    print(f"{runs} edit sets, {ok} tensors match their manifest, {bad} do not")
    return bad


if __name__ == "__main__":
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness")
    sys.exit(1 if main(sys.argv[1] if len(sys.argv) > 1 else here) else 0)
