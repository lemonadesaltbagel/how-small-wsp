"""Calibrate tasks/exp8sl/rule_gate.py against the post-hoc Codex audit of finished runs.
For each run: all audited violations plus an equal number of random allowed calls; run the gate on each code block
and print the confusion table and the gate's latency. Runs on a node (needs network); does not touch the runs.
  python3 -I scripts/calibrate_rule_gate.py [--v2] [--reviewer claude:<model>] [--adjudicated] [--extra <run>:<call>:<truth>] ... <run> [...]
--v2: run the gate with --v2 (task -strict2). --adjudicated: apply each run's audit_adjudication.json (overruled
violations count as allowed). --extra: also test this tool_calls.jsonl row (e.g. a call the gate blocked at run time,
whose code is under input.code) with the given truth (violation | allowed)."""
import json
import os
import random
import subprocess
import sys
import tempfile
import time

GATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness", "tasks", "exp8sl", "rule_gate.py")


def main():
    rng = random.Random(0)
    table = {}
    lat = []
    argv = sys.argv[1:]
    gate_args = ["--v2"] if "--v2" in argv else []
    if "--reviewer" in argv:  # e.g. --reviewer claude:claude-sonnet-5-5 (passed through to rule_gate.py)
        gate_args += ["--reviewer", argv[argv.index("--reviewer") + 1]]
    adjudicated = "--adjudicated" in argv
    extra = {}
    runs = []
    it = iter(argv)
    for a in it:
        if a == "--extra":
            r, c, t = next(it).rsplit(":", 2)
            extra.setdefault(os.path.normpath(r), []).append((int(c), t))
        elif a == "--reviewer":
            next(it)
        elif not a.startswith("--"):
            runs.append(a)
    for run in runs:
        rows = [json.loads(l) for l in open(os.path.join(run, "tool_calls.jsonl"))]
        # audit "call" = line index in tool_calls.jsonl (all kinds), see openwish-audit python_calls()
        aud = [json.loads(l) for l in open(os.path.join(run, "audit.jsonl"))]
        over = set()
        adj = os.path.join(run, "audit_adjudication.json")
        if adjudicated and os.path.exists(adj):
            over = {int(k) for k in (json.load(open(adj)).get("overruled") or [])}  # list of calls or {call: why}
        viol = [int(a["call"]) for a in aud if a["verdict"] == "violation" and int(a["call"]) not in over]
        allow = [int(a["call"]) for a in aud if a["verdict"] == "allowed"]
        sample = [(i, "violation") for i in viol] + [(i, "allowed") for i in sorted(over)]
        sample += [(i, "allowed") for i in rng.sample(allow, min(max(len(viol), len(over)), len(allow)))]
        sample += extra.get(os.path.normpath(run), [])
        with tempfile.TemporaryDirectory() as tmp:
            for i, truth in sample:
                code = (rows[i].get("code") or (rows[i].get("input") or {}).get("code") or "") if i < len(rows) else ""
                if i >= len(rows) or rows[i].get("kind") != "python" or not code:
                    print(f"  skip {i}: not a python row"); continue
                with open(os.path.join(tmp, "tool_calls.jsonl"), "w") as f:   # the history the gate would have seen
                    for r in rows[:i]:
                        if r.get("kind") != "python":
                            continue
                        f.write(json.dumps({"kind": "python", "code": r.get("code") or ""}) + "\n")
                ev = json.dumps({"event": "pre_tool", "tool": "python", "turn": i, "input": {"code": code}})
                t0 = time.time()
                p = subprocess.run([sys.executable, "-I", GATE] + gate_args, input=ev, capture_output=True, text=True,
                                   env=dict(os.environ, OPENWISH_RUN_DIR=tmp), timeout=150)
                lat.append(time.time() - t0)
                gate = "block" if p.returncode == 1 else "allow"
                table[(truth, gate)] = table.get((truth, gate), 0) + 1
                if (truth == "violation") != (gate == "block"):
                    print(f"  disagree {os.path.basename(run)} call {i}: codex={truth} gate={gate} {p.stderr.strip()[:160]}")
            gl = os.path.join(tmp, "rule_gate.jsonl")
            un = [json.loads(l) for l in open(gl)] if os.path.exists(gl) else []
            n_unrev = sum(r.get("verdict") == "unreviewed" for r in un)
            if n_unrev:
                print(f"  {n_unrev} unreviewed (reviewer errors) in {run}")
                for r in un:
                    if r.get("verdict") == "unreviewed":
                        print(f"    turn {r.get('turn')}: {r.get('reason')}")
    print("confusion (codex verdict, gate decision):", json.dumps({f"{a}/{b}": n for (a, b), n in sorted(table.items())}))
    lat.sort()
    print(f"latency s: median {lat[len(lat) // 2]:.1f}, max {lat[-1]:.1f}, n {len(lat)}")


if __name__ == "__main__":
    main()
