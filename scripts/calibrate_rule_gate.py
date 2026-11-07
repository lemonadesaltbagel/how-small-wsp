"""Calibrate tasks/exp8sl/rule_gate.py against the post-hoc Codex audit of finished runs.
For each run: all audited violations plus an equal number of random allowed calls; run the gate on each code block
and print the confusion table and the gate's latency. Runs on a node (needs network); does not touch the runs.
  python3 -I scripts/calibrate_rule_gate.py harness/runs/phase0/<run> [...]"""
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
    for run in sys.argv[1:]:
        rows = [json.loads(l) for l in open(os.path.join(run, "tool_calls.jsonl"))]
        # audit "call" = line index in tool_calls.jsonl (all kinds), see openwish-audit python_calls()
        aud = [json.loads(l) for l in open(os.path.join(run, "audit.jsonl"))]
        viol = [int(a["call"]) for a in aud if a["verdict"] == "violation"]
        allow = [int(a["call"]) for a in aud if a["verdict"] == "allowed"]
        sample = [(i, "violation") for i in viol] + [(i, "allowed") for i in rng.sample(allow, min(len(viol), len(allow)))]
        with tempfile.TemporaryDirectory() as tmp:
            for i, truth in sample:
                if i >= len(rows) or rows[i].get("kind") != "python":
                    print(f"  skip {i}: not a python row"); continue
                with open(os.path.join(tmp, "tool_calls.jsonl"), "w") as f:   # the history the gate would have seen
                    for r in rows[:i]:
                        if r.get("kind") != "python":
                            continue
                        f.write(json.dumps({"kind": "python", "code": r.get("code") or ""}) + "\n")
                ev = json.dumps({"event": "pre_tool", "tool": "python", "turn": i, "input": {"code": rows[i].get("code") or ""}})
                t0 = time.time()
                p = subprocess.run([sys.executable, "-I", GATE], input=ev, capture_output=True, text=True,
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
