"""Tool-use rates of finished runs and the "tool-use failure" sub-annotation (analysis plan amendment A3). Stdlib only.
  python3 -I scripts/tool_use_stats.py <run dir> [<run dir> ...]   -> one JSON line per run
tool-use failure (A3 as refined by A5) = any of: >= half of the Python runs end in an error; >= half of the forward
requests are rejected or fail; or the editor said its tools do not exist or do not work (from the labeller, field
claims_tools_unavailable_call, merged in by the caller).
early stop (A5) = the episode ended with forward passes left: a final answer accepted after the 5 nudges, or the
empty-reply stop."""
import json
import os
import sys


def stats(run):
    rows = [json.loads(l) for l in open(os.path.join(run, "tool_calls.jsonl"))]
    st = json.load(open(os.path.join(run, "state.json")))
    py = [r for r in rows if r.get("kind") == "python"]
    fw = [r for r in rows if str(r.get("kind", "")).startswith("forward")]
    py_err = sum(1 for r in py if r.get("error") or r.get("blocked"))
    fw_bad = sum(1 for r in fw if not r.get("ok", True) and r.get("pass") is None)
    end = st.get("end_reason") or ""
    reasons = []
    if py and py_err / len(py) >= 0.5:
        reasons.append(f"python errors {py_err}/{len(py)}")
    if fw and fw_bad / len(fw) >= 0.5:
        reasons.append(f"forward requests failed {fw_bad}/{len(fw)}")
    passes_left = 50 - sum((st.get("passes_used") or {}).values())
    early = None
    if passes_left > 0 and st.get("done"):
        early = "empty replies" if "empty repl" in end else (f"final answer after {st.get('nudges')} nudges" if (st.get("nudges") or 0) >= 5 else None)
    return {"run": run, "python_runs": len(py), "python_errors": py_err, "forward_requests": len(fw),
            "forward_failed_uncharged": fw_bad, "passes_charged": sum((st.get("passes_used") or {}).values()),
            "protocol_errors": st.get("protocol_errors"), "nudges": st.get("nudges"),
            "empty_replies": st.get("empty_replies"), "rationale_missing": sum(1 for r in rows if r.get("kind") == "rationale-missing"),
            "turns": st.get("turn"), "end_reason": end, "passes_left": passes_left,
            "tool_use_failure_auto": reasons, "early_stop": early}


if __name__ == "__main__":
    for r in sys.argv[1:]:
        print(json.dumps(stats(r)))
