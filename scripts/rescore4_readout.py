"""Read-out of a 4-sample MATH-500 rescore (2026-11-14; analysis plan A6): the score, the thinking-skip share
(</think> within 120 characters, as scripts/think_close_check.py), the score with the skip generations' correct
answers removed (the skip rule, line 29.0 %), and the pairing with the 4-sample thinking-off control T0 by problem
(per-problem mean over the 4 samples, SE over the 500 problems; A6 reading (2)). Stdlib only:
  python3 -I scripts/rescore4_readout.py <run dir> ...      (run dir holds eval/qwen_math500/, or is that dir)"""
import json
import math
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
T0 = os.path.join(ROOT, "notes/run_records/truncation_control/nothink4_out.json")
N = 120


def readout(run):
    d = run if os.path.exists(os.path.join(run, "math500.jsonl")) else os.path.join(run, "eval", "qwen_math500")
    recs = [json.loads(l) for l in open(os.path.join(d, "math500.jsonl"))]
    raw = json.load(open(os.path.join(d, ".raw.json")))
    assert len(raw) == len(recs) and all(a["tokens"] == b["tokens"] for a, b in zip(raw, recs)), "raw/jsonl order"
    skip = [0 <= r["text"].find("</think>") < N for r in raw]
    n = len(recs); ok = [bool(r["correct"]) for r in recs]
    t0 = {(r["key"], r["sample"]): bool(r["correct"]) for r in json.load(open(T0))["records"]}
    per = {}
    for o, r in zip(ok, recs):
        per.setdefault(r["key"], []).append(int(o) - int(t0[(r["key"], r["sample"])]))
    diff = [sum(v) / len(v) for v in per.values()]; k = len(diff)
    m = sum(diff) / k; se = math.sqrt(sum((x - m) ** 2 for x in diff) / (k - 1) / k)
    return {"run": os.path.relpath(run, ROOT), "n": n, "samples": len({r["sample"] for r in recs}),
            "acc": round(sum(ok) / n, 4), "skip": sum(skip), "skip_correct": sum(s and o for s, o in zip(skip, ok)),
            "acc_without_skip_correct": round(sum(o and not s for s, o in zip(skip, ok)) / n, 4),
            "vs_T0": {"problems": k, "t0_acc": round(sum(t0.values()) / len(t0), 4), "diff_pp": round(100 * m, 2),
                      "se_pp": round(100 * se, 2), "z": round(m / se, 2) if se else None}}


if __name__ == "__main__":
    print(json.dumps([readout(os.path.abspath(r)) for r in sys.argv[1:]], indent=1))
