"""Structure of each kept edit: entries / rows / columns changed and the numerical rank of (edited - base).
Hand-derived edits (direction projections, column subsets) are low-rank and localized; an edit produced by
gradient descent or fitting would be dense and close to full rank. Run with the sandbox venv from harness/:
  assets/venvs/sandbox/bin/python -I ../scripts/edit_structure.py runs/phase0/<run> [...]"""
import json
import os
import sys

import numpy as np


def main():
    for run in sys.argv[1:]:
        cur = os.path.join(run, "edits", "current")
        base = "assets/exp8/base_net"
        names = sorted(f for f in os.listdir(os.path.join(run, "work", "edited")) if f.endswith(".npy"))
        rows = []
        for f in names:
            fn = f.replace("/", "__")
            b = np.load(os.path.join(base, fn)).astype(np.float64)
            c = np.load(os.path.join(cur, fn)).astype(np.float64)
            d = c - b
            nz = d != 0
            if not nz.any():
                rows.append({"tensor": f[:-4], "entries": 0})
                continue
            s = np.linalg.svd(d, compute_uv=False) if d.ndim == 2 else np.array([np.abs(d).max()])
            # rank above bf16-rounding noise: singular values > 1 % of the largest
            rank = int((s > 0.01 * s[0]).sum())
            e = {"tensor": f[:-4].replace("model.language_model.", ""), "shape": list(d.shape), "entries": int(nz.sum()),
                 "frac_of_tensor": round(float(nz.mean()), 4), "rank_1pct": rank, "full_rank": int(min(d.shape)) if d.ndim == 2 else 1,
                 "top_sv": [round(float(x), 4) for x in s[:4]], "rel_fro": round(float(np.linalg.norm(d) / np.linalg.norm(b)), 4)}
            if d.ndim == 2:
                e["rows_changed"] = int(nz.any(1).sum())
                e["cols_changed"] = int(nz.any(0).sum())
            rows.append(e)
        print(json.dumps({"run": run, "edits": rows}))


if __name__ == "__main__":
    main()
