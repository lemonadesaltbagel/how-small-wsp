"""Thinking profile per scored MATH-500 run (stdlib):  python3 -I scripts/thinking_profile.py > notes/thinking_profile.json
Per run, from eval/qwen_math500/{items.json,.raw.json} and the MATH level of each item:
- per level 1-5: n, share that closes </think>, median thinking length (characters before </think>; a generation that
  never closes counts its whole text), accuracy;
- Spearman between level and thinking length (ties averaged);
- repetition inside the thinking: share of non-empty lines (stripped, >= 8 chars) equal to an earlier line of the same
  thinking, over the whole thinking, over its last 3,000 characters, and over its first 6,000 characters (a window of
  the same size for every run, since the share grows with length); medians over generations."""
import json
import os
import statistics

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RUNS = {
    "base": "harness/runs/exp8_conformance/base_math500",
    "Astra p50c": "harness/runs/exp8_conformance/p50c_math500",
    "Opus strict s1": "harness/runs/phase0/opus55-hinted-p50-strict-s1/eval/qwen_math500",
    "Opus strict s2": "harness/runs/phase0/opus55-hinted-p50-strict-s2/eval/qwen_math500",
    "Opus strict s3": "harness/runs_laptop/phase0/opus55-hinted-p50-strict-s3/eval/qwen_math500",
    "DS unhinted s2": "harness/runs_laptop/phase0/deepseekflash-unhinted-p50-strict2-s2/eval/qwen_math500",
}
TAIL = 3000
HEAD = 6000


def levels():
    out = {}
    for line in open(os.path.join(ROOT, "harness/assets/exp8/eval/math500.jsonl")):
        d = json.loads(line)
        out[d["unique_id"]] = int(d["level"])
    return out


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else None


def rep_share(text):
    seen, rep, n = set(), 0, 0
    for ln in text.splitlines():
        s = ln.strip()
        if len(s) < 8:
            continue
        n += 1
        rep += s in seen
        seen.add(s)
    return rep / n if n else 0.0


def profile(d):
    items = json.load(open(os.path.join(ROOT, d, "items.json")))
    raw = json.load(open(os.path.join(ROOT, d, ".raw.json")))
    graded = {}
    jl = os.path.join(ROOT, d, "math500.jsonl")
    for line in open(jl):
        g = json.loads(line)
        graded[(g["key"], g["sample"])] = g["correct"]
    lv = levels()
    rows = []
    for it, r in zip(items, raw):
        if it["sample"] != 0:
            continue
        t = r["text"]
        p = t.find("</think>")
        think = t if p < 0 else t[:p]
        rows.append({"level": lv[it["key"]], "closed": p >= 0, "len": len(think), "correct": graded[(it["key"], 0)],
                     "rep": rep_share(think), "rep_tail": rep_share(think[-TAIL:]),
                     "rep_head": rep_share(think[:HEAD])})
    by = {}
    for L in range(1, 6):
        rs = [x for x in rows if x["level"] == L]
        by[L] = {"n": len(rs), "closed": round(sum(x["closed"] for x in rs) / len(rs), 3),
                 "median_think_chars": int(statistics.median(x["len"] for x in rs)),
                 "accuracy": round(sum(x["correct"] for x in rs) / len(rs), 3),
                 "median_rep": round(statistics.median(x["rep"] for x in rs), 3)}
    return {"dir": d, "n": len(rows), "by_level": by,
            "spearman_level_len": round(spearman([x["level"] for x in rows], [x["len"] for x in rows]), 3),
            "median_rep_whole": round(statistics.median(x["rep"] for x in rows), 3),
            "median_rep_tail": round(statistics.median(x["rep_tail"] for x in rows), 3),
            "median_rep_first6k": round(statistics.median(x["rep_head"] for x in rows), 3),
            "mean_rep_whole": round(statistics.fmean(x["rep"] for x in rows), 3),
            "closed": round(sum(x["closed"] for x in rows) / len(rows), 3)}


def main():
    print(json.dumps({name: profile(d) for name, d in RUNS.items()}, indent=1))


if __name__ == "__main__":
    main()
