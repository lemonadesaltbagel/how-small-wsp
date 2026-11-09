"""Re-review given tool_calls rows of a finished run with rule_gate.py --v2, with the history the gate would have seen.
  python -I scripts/recheck_gate_v2.py <run> <call,call,...> [repeats]   (run on a node: needs the DeepSeek API)"""
import json, os, subprocess, sys, tempfile
run, calls = sys.argv[1], [int(c) for c in sys.argv[2].split(",")]
gate = os.path.join(os.getcwd(), "harness", "tasks", "exp8sl", "rule_gate.py")
rows = [json.loads(l) for l in open(os.path.join(run, "tool_calls.jsonl"))]
for rep in range(int(sys.argv[3]) if len(sys.argv) > 3 else 1):
    for i in calls:
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "tool_calls.jsonl"), "w") as f:
                for r in rows[:i]:
                    if r.get("kind") == "python" and r.get("code"):
                        f.write(json.dumps({"kind": "python", "code": r["code"]}) + "\n")
            ev = json.dumps({"event": "pre_tool", "tool": "python", "turn": i, "input": {"code": rows[i]["code"]}})
            p = subprocess.run([sys.executable, "-I", gate, "--v2"], input=ev, capture_output=True, text=True,
                               env=dict(os.environ, OPENWISH_RUN_DIR=tmp), timeout=200)
            log = [json.loads(l) for l in open(os.path.join(tmp, "rule_gate.jsonl"))]
            print(json.dumps({"rep": rep, "call": i, "verdict": log[-1].get("verdict"), "rule": log[-1].get("rule"),
                              "reason": log[-1].get("reason")}))
