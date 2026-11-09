"""Check a finished run's submitted edit independently of the harness (credibility check; numpy only).
  <python with numpy> -I scripts/verify_edits.py <base_net dir> <run dir> [<run dir> ...]
For each run, with final/editset/{manifest.json,edit.npz} and final/net/ (the network that was scored):
1. every base tensor named in the manifest hashes to sha_base, every edit.npz tensor to sha_edited
   (the harness hash: sha256 of ndim, dims as u64 LE, then the float32 values LE);
2. final/net equals base + edit set exactly, tensor by tensor, over the whole network, so nothing outside the
   declared edit was changed;
3. the changed-entry counts match the manifest (an entry counts when its value differs; -0.0 == 0.0, NaN == NaN)."""
import hashlib
import json
import os
import sys

import numpy as np


def sha(a):
    a = np.ascontiguousarray(a, dtype="<f4")
    h = hashlib.sha256()
    h.update(np.uint64(a.ndim).astype("<u8").tobytes())
    for d in a.shape:
        h.update(np.uint64(d).astype("<u8").tobytes())
    h.update(a.tobytes())
    return h.hexdigest()


def n_diff(a, b):
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    return int((~same).sum())


def fname(name):
    return name.replace("/", "__") + ".npy"


def check(base_dir, run):
    m = json.load(open(os.path.join(run, "final", "editset", "manifest.json")))
    ed = np.load(os.path.join(run, "final", "editset", "edit.npz"))
    net = os.path.join(run, "final", "net")
    problems = []
    for k, ent in m["tensors"].items():
        b = np.load(os.path.join(base_dir, fname(k)), mmap_mode="r")
        if sha(b) != ent["sha_base"]:
            problems.append(f"{k}: base hash mismatch")
        e = ed[k]
        if sha(e) != ent["sha_edited"]:
            problems.append(f"{k}: edited hash mismatch")
        if n_diff(e, b) != ent["changed"]:
            problems.append(f"{k}: changed {n_diff(e, b)} != manifest {ent['changed']}")
    checked = 0
    for f in sorted(os.listdir(net)):
        if not f.endswith(".npy"):
            continue
        k = f[:-4].replace("__", "/")
        got = np.load(os.path.join(net, f), mmap_mode="r")
        want = ed[k] if k in m["tensors"] else np.load(os.path.join(base_dir, f), mmap_mode="r")
        if got.shape != want.shape or n_diff(got, want):
            problems.append(f"{k}: final/net differs from base + edit set")
        checked += 1
    out = {"run": run, "edited_tensors": len(m["tensors"]), "changed_total": m.get("changed_total"),
           "numel_total": m.get("numel_total"), "net_tensors_checked": checked, "ok": not problems, "problems": problems}
    print(json.dumps(out))
    return not problems


def main():
    base = sys.argv[1]
    ok = all([check(base, r) for r in sys.argv[2:]])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
