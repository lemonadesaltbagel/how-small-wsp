"""DeepSeek API spend from the logged usage of every deepseek-flash call (editor turns and rule-gate reviews).

Price list (api-docs.deepseek.com/zh-cn/quick_start/pricing, read 2026-11-15), deepseek-flash, CNY per 1M tokens at
peak: input cache hit 0.04, cache miss 2, output 8; off-peak is half. Peak = Beijing Mon-Fri 09-12 and 14-18.
Editor turns carry a Unix `created`; rule_gate.jsonl times are node-local (AEDT, +11). Compared with the account
balance readings in notes/compute_log.md. Run with `python3 -I scripts/ds_api_cost.py` from the repo root."""
import collections
import datetime as dt
import glob
import json

PEAK = {"hit": 0.04, "miss": 2.0, "out": 8.0}
BJ = dt.timezone(dt.timedelta(hours=8))
AEDT = dt.timezone(dt.timedelta(hours=11))
# (label, from, to, balance decrease in CNY); the top-up window has no balance reading
WINDOWS = [("to 11-09 17:24", None, "2026-11-09T17:24", 65.20 - 35.90),
           ("top-up window", "2026-11-09T17:24", "2026-11-10T02:08", None),
           ("11-10 02:08 to 11-11 23:30", "2026-11-10T02:08", "2026-11-11T23:30", 134.29 - 100.05),
           ("11-11 23:30 to 11-15", "2026-11-11T23:30", "2026-11-16T00:00", 100.05 - 63.04)]


def is_peak(t):
    b = t.astimezone(BJ)
    return b.weekday() < 5 and (9 <= b.hour < 12 or 14 <= b.hour < 18)


def cost(t, u):
    f = 1.0 if is_peak(t) else 0.5
    return f * (u["prompt_cache_hit_tokens"] * PEAK["hit"] + u["prompt_cache_miss_tokens"] * PEAK["miss"]
                + u["completion_tokens"] * PEAK["out"]) / 1e6


def records():
    for f in glob.glob("harness/runs*/phase0/deepseek*/turns/*/raw.json"):
        j = json.load(open(f))
        if isinstance(j, dict) and j.get("usage"):
            yield dt.datetime.fromtimestamp(j["created"], dt.timezone.utc), "editor", f.split("/")[3], j["usage"]
    for f in glob.glob("harness/runs*/phase0/*/rule_gate.jsonl"):
        # gate times are naive node-local; take the node's offset from the run's loop.log (one run ran in UTC)
        stamp = open(f.replace("rule_gate.jsonl", "loop.log")).readline().split("]")[0]
        tz = dt.timezone(dt.timedelta(hours=int(stamp[-6:-3]))) if stamp[-6] in "+-" else AEDT
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            u = r.get("usage")
            if isinstance(u, dict) and "prompt_cache_hit_tokens" in u:  # DeepSeek reviews only (Claude ones differ)
                yield dt.datetime.fromisoformat(r["time"]).replace(tzinfo=tz), "reviewer", f.split("/")[3], u


def main():
    recs = sorted(records(), key=lambda r: r[0])
    by = collections.defaultdict(lambda: collections.Counter())
    for t, role, run, u in recs:
        c = by[(role, run if role == "editor" else "all runs")]
        c["calls"] += 1
        c["hit"] += u["prompt_cache_hit_tokens"]
        c["miss"] += u["prompt_cache_miss_tokens"]
        c["out"] += u["completion_tokens"]
        c["cny"] += cost(t, u)
    for (role, run), c in sorted(by.items()):
        print(f"{role:8s} {run:40s} calls {c['calls']:5d}  hit {c['hit']/1e6:6.1f}M  miss {c['miss']/1e6:5.2f}M  "
              f"out {c['out']/1e6:5.2f}M  ¥{c['cny']:6.2f}")
    print(f"logged total ¥{sum(cost(t, u) for t, _, _, u in recs):.2f}")
    for label, a, b, bal in WINDOWS:
        lo = dt.datetime.fromisoformat(a).replace(tzinfo=AEDT) if a else dt.datetime.min.replace(tzinfo=dt.timezone.utc)
        hi = dt.datetime.fromisoformat(b).replace(tzinfo=AEDT)
        logged = sum(cost(t, u) for t, _, _, u in recs if lo <= t < hi)
        print(f"{label:28s} logged ¥{logged:6.2f}  balance decrease " + (f"¥{bal:6.2f}" if bal is not None else "  n/a"))


if __name__ == "__main__":
    main()
