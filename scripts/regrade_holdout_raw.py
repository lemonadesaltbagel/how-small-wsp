"""Grade hold-out generations that were saved (<eval dir>/qwen_<set>/.raw.json) but not graded, with holdout_eval.py's
own grader and summary; no GPU, no regeneration. Used when grading failed after generation (2026-11-08: Opus strict s1
IFEval, nltk refused its data folder on the pod because rsync had kept another uid as owner).
  <qwen venv>/bin/python -I scripts/regrade_holdout_raw.py <eval dir> <set> <checkpoint tag> [> summary.json]
Writes <eval dir>/qwen_<set>/<set>.jsonl like evaluate() does and prints the summary JSON. The .raw.json is kept."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "harness", "python", "adapters"))
import holdout_eval as H  # noqa: E402


def main():
    ev, s, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    out = os.path.join(ev, f"qwen_{s}")
    data = os.path.abspath(H.DEFAULT_DATA)
    rows = H.load_rows(data, s)
    items = json.load(open(os.path.join(out, "items.json")))
    gens = json.load(open(os.path.join(out, ".raw.json")))
    k = H.SAMPLES[s]
    assert len(items) == len(gens) == len(rows) * k, (len(items), len(gens), len(rows), k)
    bad = sum(g["finish"] not in ("stop", "length") for g in gens)
    assert not bad, f"{bad} unfinished generations"
    by = {r["id"]: r for r in rows}
    recs = []
    for it, g in zip(items, gens):
        r = by[it["key"]]
        gr = H.grade_one(r, g["text"], g["finish"], os.path.join(data, "nltk_data"))
        recs.append({"key": it["key"], "source": s, "sample": it["sample"], "category": r.get("category"), **gr,
                     "finish": g["finish"], "tokens": g["tokens"], "text": g["text"]})
    with open(os.path.join(out, f"{s}.jsonl"), "w") as f:
        for r in recs:
            t = r["text"]
            f.write(json.dumps({k_: v for k_, v in r.items() if k_ != "text"}
                               | {"text_head": t[:2000], "answer_head": t.split("</think>")[-1][:4000] if "</think>" in t else None}) + "\n")
    sm = H.summarise(rows, recs)
    print(json.dumps({"ok": True, "score": sm["accuracy"], "set": s, "problems": len(rows), "samples": k, "checkpoint": tag,
                      "data_sha256": H.sha256(os.path.join(data, f"{s}.jsonl")), "regraded_from_raw": True, **sm}))


if __name__ == "__main__":
    main()
