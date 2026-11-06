"""How busy each target-GPU pod was: forward-pass wall time (from tool_calls.jsonl) against the pod's lifetime.
usage: python3 -I scripts/pod_utilization.py <run dir> <pod created, ISO local time> [...]
Also splits passes into generation / teacher-forced and successful / failed."""
import datetime
import json
import sys

args = sys.argv[1:]
for run, created in zip(args[0::2], args[1::2]):
    t0 = datetime.datetime.fromisoformat(created)
    now = datetime.datetime.now()
    rows = [json.loads(l) for l in open(f"{run}/tool_calls.jsonl") if '"kind":"forward' in l or '"kind": "forward' in l]
    tot = {}
    for r in rows:
        kind = "generate" if '"type": "generate"' in r.get("request", "") or '"type":"generate"' in r.get("request", "") else "teacher_forced"
        ok = "ok" if r.get("ok") in (True, "true", "True") else "failed"
        k = (kind, ok)
        n, s = tot.get(k, (0, 0.0))
        tot[k] = (n + 1, s + float(r.get("seconds") or 0))
    busy = sum(s for _, s in tot.values())
    life = (now - t0).total_seconds()
    print(f"{run.split('/')[-1]}: pod life {life / 3600:.2f} h, forward passes busy {busy / 3600:.2f} h = {100 * busy / life:.0f} %")
    for (kind, ok), (n, s) in sorted(tot.items()):
        print(f"   {kind:15s} {ok:6s} n={n:3d}  {s / 3600:5.2f} h  (mean {s / max(n, 1) / 60:5.1f} min)")
