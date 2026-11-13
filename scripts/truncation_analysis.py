"""Paired read-out of the truncation controls against the judging criteria (notes/pending_user_decisions.md, 2026-11-13
03:40): per edit, its sample-0 MATH-500 correctness vs T0 (thinking off, 1-sample run of 11-12 and sample 0 of the
4-sample run), T1 matched and T2 fixed (scripts/truncation_control.py), paired by item. Stdlib only:
  python3 -I scripts/truncation_analysis.py [trunc_out.json ...] > notes/run_records/truncation_control/analysis.json"""
import json
import math
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from truncation_prep import BASE, EDITS  # noqa: E402

TC = os.path.join(ROOT, "notes/run_records/truncation_control")


def s0(path):
    out = {}
    for l in open(path):
        r = json.loads(l)
        if r["sample"] == 0:
            out[r["key"]] = bool(r["correct"])
    return out


def nothink(path):
    return {r["key"]: bool(r["correct"]) for r in json.load(open(path))["records"] if r["sample"] == 0}


def paired(a, b):
    ks = sorted(set(a) & set(b))
    d = [int(a[k]) - int(b[k]) for k in ks]
    n = len(d); m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
    return {"n": n, "diff_pp": round(100 * m, 2), "se_pp": round(100 * se, 2), "z": round(m / se, 2) if se else None}


def main():
    files = sys.argv[1:] or [os.path.join(TC, "trunc_out.json")]
    tr = {}
    for f in files:
        if os.path.exists(f):
            tr.update(json.load(open(f)))
    refs = {"T0 (1-sample, 11-12)": nothink(os.path.join(ROOT, "notes/run_records/nothink_control/nothink_out.json")),
            "T0 (4-sample run, sample 0)": nothink(os.path.join(TC, "nothink4_out.json")),
            "base": s0(os.path.join(ROOT, BASE, "math500.jsonl"))}
    out = {}
    for name, d in EDITS.items():
        if f"{name} | T1 matched" not in tr:
            continue
        e = s0(os.path.join(ROOT, d, "math500.jsonl"))
        row = {"edit_acc": round(sum(e.values()) / len(e), 4)}
        conds = dict(refs)
        for c in ("T1 matched", "T2 fixed"):
            t = tr[f"{name} | {c}"]
            conds[c] = {k: bool(v) for k, v in t["correct"].items()}
            row[c + " n_cut"] = t["summary"].get("n_cut")
            row[c + " cut_tokens"] = t["summary"].get("cut_tokens_fixed") or t["summary"].get("median_cut")
        for c, v in conds.items():
            row[c] = {"acc": round(sum(v.values()) / len(v), 4), **paired(e, v)}
        out[name] = row
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
