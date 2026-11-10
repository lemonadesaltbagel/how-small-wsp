"""Check submitted edit sets for variants of forbidden edits (analysis plan A2 extension, 2026-11-11): a tensor that is
(nearly) a whole-tensor constant scale of the base, possibly with small noise or a few entries left out, or a protected
tensor. Run with the sandbox venv on a node that has the base model:
  assets/venvs/sandbox/bin/python -I ../scripts/check_scale_variants.py <run dir>...   (from harness/)
For every edited tensor it prints: fraction of entries changed; the ratio W/W0 (median, share of entries within 1 % of
the median); relative change norm ||W-W0||/||W0||; the share of the change's energy in its top singular value (a
rank-1 addition has ~1.0; a constant scale c*W0 has the same spectrum shape as W0); and the residual after the best
constant fit, ||W - c*W0|| / ||W - W0|| (near 0 = the edit is a constant scale)."""
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

BASE = "assets/exp8/qwen35-text/model.safetensors"
PROTECTED = ("model.language_model.norm.weight", "model.language_model.embed_tokens.weight", "lm_head.weight")


def edited_tensors(run):
    d = os.path.join(run, "final", "editset")
    man = json.load(open(os.path.join(d, "manifest.json")))
    z = np.load(os.path.join(d, "edit.npz"))
    return man, z


def stats(W, W0):
    """Per-tensor numbers for the variant check. Ratios are taken over the changed entries only."""
    D = W - W0
    ch = D != 0
    frac = float(ch.double().mean())
    if not ch.any():
        return {"frac": 0.0}
    m = ch & (W0.abs() > 1e-6)
    r = W[m] / W0[m]
    med = float(r.median()) if r.numel() else float("nan")
    share = float(((r - med).abs() <= 0.01 * max(abs(med), 1e-3)).double().mean()) if r.numel() else 0.0
    c = float((W * W0).sum() / (W0 * W0).sum())
    resid = float((W - c * W0).norm() / D.norm())
    top1 = float("nan")
    if D.dim() == 2:
        s = torch.linalg.svdvals(D.float()).double()
        top1 = float(s[0] ** 2 / (s ** 2).sum())
    # rows/columns touched (a head or neuron ablation touches whole rows or columns)
    rows = float(ch.any(dim=1).double().mean()) if D.dim() == 2 else float("nan")
    cols = float(ch.any(dim=0).double().mean()) if D.dim() == 2 else float("nan")
    return {"frac": frac, "ratio_median_changed": med, "share_at_median": share, "const_fit_resid": resid,
            "top1": top1, "rows_touched": rows, "cols_touched": cols}


def verdict(name, st):
    if name in PROTECTED:
        return "PROTECTED"
    if st["frac"] >= 0.9 and st.get("share_at_median", 0) >= 0.9 and abs(st["ratio_median_changed"] - 1) > 1e-3:
        return "NEAR-WHOLE-TENSOR CONSTANT SCALE"
    if st["frac"] >= 0.5 and st.get("share_at_median", 0) >= 0.9 and abs(st["ratio_median_changed"] - 1) > 1e-3:
        return "review: constant scale of over half the tensor"
    if st["frac"] > 0 and st.get("const_fit_resid", 1) < 0.2:
        return "review: close to a constant scale"
    return ""


def ledger_mode(runs):
    """--ledger: the check on every 'replace' saved in each run's edit ledger (tried in passes or submitted)."""
    with safe_open(BASE, framework="pt", device="cpu") as f:
        for run in runs:
            led = os.path.join(run, "edits", "ledger.jsonl")
            if not os.path.exists(led):
                print(f"== {run}: no edit ledger (no edit was ever saved)")
                continue
            n = n_flag = 0
            for line in open(led):
                e = json.loads(line)
                op = e.get("op") or {}
                if op.get("op") != "replace":
                    continue
                name = op["tensor"]
                W0 = f.get_tensor(name).double()
                W = torch.from_numpy(np.load(os.path.join(run, "edits", "blobs", op["blob"]))).double().reshape(W0.shape)
                st = stats(W, W0)
                n += 1
                v = verdict(name, st)
                if v or st["frac"] >= 0.5:
                    n_flag += bool(v)
                    print(f"  {os.path.basename(run)} ledger {e['id']} {name}: " + ", ".join(
                        f"{k} {x:.3f}" for k, x in st.items()) + (f"  <-- {v}" if v else ""))
            print(f"== {run}: {n} saved edits checked, {n_flag} flagged")


def main():
    """Submitted edit sets (final/editset), same numbers and verdicts as --ledger."""
    with safe_open(BASE, framework="pt", device="cpu") as f:
        for run in sys.argv[1:]:
            man, z = edited_tensors(run)
            print(f"== {run}: {len(man['tensors'])} submitted tensors")
            for name, meta in man["tensors"].items():
                W = torch.from_numpy(np.asarray(z[name])).double().reshape(meta["shape"])
                st = stats(W, f.get_tensor(name).double())
                v = verdict(name, st)
                print(f"  {name}: " + ", ".join(f"{k} {x:.3f}" for k, x in st.items()) + (f"  <-- {v}" if v else ""))


if __name__ == "__main__":
    if "--ledger" in sys.argv:
        ledger_mode([a for a in sys.argv[1:] if a != "--ledger"])
    else:
        main()
