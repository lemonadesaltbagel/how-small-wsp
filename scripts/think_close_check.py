"""Thinking-skip check (added 2026-11-11 after DS hinted s1): for each scored run, the share of MATH-500 generations
that emit </think> within the first N characters (default 120), and the median character position of </think>.
The unedited model never closes that early. Stdlib only:  python3 -I scripts/think_close_check.py [run dir ...]
(no args: the base and every run in notes/episode_ledger.json with a MATH-500 score)."""
import json
import os
import statistics
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
N = 120


def check(raw_path):
    raw = json.load(open(raw_path))
    pos = [r["text"].find("</think>") for r in raw]
    closed = [p for p in pos if p >= 0]
    return {"n": len(raw), "early": sum(0 <= p < N for p in pos), "no_close": sum(p < 0 for p in pos),
            "median_close_chars": statistics.median(closed) if closed else None}


def main():
    runs = sys.argv[1:]
    if not runs:
        runs = ["harness/runs/exp8_conformance/base_math500"]
        for r in json.load(open(os.path.join(ROOT, "notes", "episode_ledger.json"))):
            if r.get("math500") is not None:
                runs.append("harness/" + r["run"])
    for run in runs:
        p = os.path.join(ROOT, run, ".raw.json")
        if not os.path.exists(p):
            p = os.path.join(ROOT, run, "eval", "qwen_math500", ".raw.json")
        if not os.path.exists(p):
            print(json.dumps({"run": run, "missing": True}))
            continue
        print(json.dumps({"run": run, **check(p)}))


if __name__ == "__main__":
    main()
