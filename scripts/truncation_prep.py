"""Bundle for the truncation controls (scripts/truncation_control.py). Run on a node with the qwen venv (tokenizer only):
  harness/assets/venvs/qwen/bin/python -I scripts/truncation_prep.py notes/run_records/truncation_control/bundle.json.xz [name...]
(names: a subset of EDITS; default all but "DS hinted s2", which was added later with its own bundle)
Per scored run (sample 0 of each MATH-500 item): the number of tokens before its first </think> (None = never closes),
plus the unedited base's items and full generations, which the control cuts."""
import json
import lzma
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BASE = "harness/runs/exp8_conformance/base_math500"
EDITS = {
    "DS unhinted s2": "harness/runs_laptop/phase0/deepseekflash-unhinted-p50-strict2-s2/eval/qwen_math500",
    "Astra p50c": "harness/runs/exp8_conformance/p50c_math500",
    "Opus strict s1": "harness/runs/phase0/opus55-hinted-p50-strict-s1/eval/qwen_math500",
    "Opus strict s2": "harness/runs/phase0/opus55-hinted-p50-strict-s2/eval/qwen_math500",
    "Opus strict s3": "harness/runs_laptop/phase0/opus55-hinted-p50-strict-s3/eval/qwen_math500",
    "DS hinted s2": "harness/runs_laptop/phase0/deepseekflash-hinted-p50-strict2-s2/eval/qwen_math500",
}


def load(d):
    items = json.load(open(os.path.join(ROOT, d, "items.json")))
    raw = json.load(open(os.path.join(ROOT, d, ".raw.json")))
    return [(it, r) for it, r in zip(items, raw) if it["sample"] == 0]


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "harness/assets/exp8/qwen35-text"))
    base = load(BASE)
    lengths = {}
    names = sys.argv[2:] or [n for n in EDITS if n != "DS hinted s2"]
    for name, d in ((n, EDITS[n]) for n in names):
        L = {}
        for it, r in load(d):
            p = r["text"].find("</think>")
            L[it["key"]] = None if p < 0 else len(tok.encode(r["text"][:p], add_special_tokens=False))
        lengths[name] = L
        closed = sorted(v for v in L.values() if v is not None)
        print(name, len(L), "closed", len(closed), "median tokens", closed[len(closed) // 2] if closed else None, flush=True)
    out = {"items": [it for it, _ in base], "base": [r for _, r in base], "lengths": lengths}
    with lzma.open(sys.argv[1], "wt") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
