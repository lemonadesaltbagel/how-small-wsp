"""Stage a published run of the authors (RikiyaT/lm_understand-logs) for `openwish audit`, so that it is judged by the
same Codex auditor and the same reading of rule (b) as our strict Opus runs.
  python3 -I scripts/stage_external_audit.py <their run dir> <out dir>
- rules.txt = their own task prompt (message.txt) + our appendix note (5) (the rule (b) reading the strict audits used).
  Our notes (1)-(4) describe our harness and are left out.
- tool_calls.jsonl: their rows with `out` renamed to `output`; forward rows become kind "forward:forward" with the
  request as a string, so the auditor sees them as context (as in our runs). Row order and indices are unchanged, so an
  audit "call" number is the line number in their file."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_exp8_tasks import STRICT_NOTE  # noqa: E402


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    prompt = open(os.path.join(src, "message.txt")).read()
    with open(os.path.join(out, "rules.txt"), "w") as f:
        f.write(prompt.rstrip() + "\n\nNOTES FROM THE EXPERIMENTERS (audit reading of rule (b)):\n" + STRICT_NOTE + "\n")
    n = {"python": 0, "forward": 0, "other": 0}
    with open(os.path.join(src, "tool_calls.jsonl")) as fi, open(os.path.join(out, "tool_calls.jsonl"), "w") as fo:
        for line in fi:
            r = json.loads(line)
            k = r.get("kind")
            if k == "python":
                r = {"kind": "python", "code": r.get("code", ""), "output": str(r.get("out", "")), "t": r.get("t")}
                n["python"] += 1
            elif k == "forward":
                r = {"kind": "forward:forward", "pass": r.get("k"), "ok": True, "request": str(r.get("request", "")),
                     "t": r.get("t")}
                n["forward"] += 1
            else:
                n["other"] += 1
            fo.write(json.dumps(r) + "\n")
    # `openwish audit` needs a saved task spec to configure the auditor; it uses only the instructions from it, which
    # are the same in our runs and theirs ("Answer the user's request."). Copy ours.
    spec = json.load(open(os.path.join(HERE, "..", "harness", "runs", "phase0", "opus55-hinted-p50-strict-s1", "spec.json")))
    theirs = open(os.path.join(src, "instructions.md")).read().strip()
    assert spec["spec"]["instructions"].strip() == theirs, (spec["spec"]["instructions"], theirs)
    json.dump(spec, open(os.path.join(out, "spec.json"), "w"))
    json.dump({"source": os.path.abspath(src), "counts": n,
               "note": "staged by scripts/stage_external_audit.py; not a run of ours"},
              open(os.path.join(out, "staged_from.json"), "w"), indent=1)
    print(json.dumps(n))


if __name__ == "__main__":
    main()
