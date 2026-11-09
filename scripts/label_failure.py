"""Propose the failure label of a finished episode with Claude Code on the node where the run lives (no data copied).
  python3 -I scripts/label_failure.py <run dir> [--model claude-sonnet-5-5]  -> prints JSON; I review it before it
  goes into <run>/run_class.json.
Labels and definitions are the frozen analysis plan's (notes/analysis_plan.md). The prompt holds the plan's definitions,
the episode's final answer, its score, and a digest of every tool call in order (code head and output head), within a
~300 KB budget."""
import json
import os
import subprocess
import sys
import tempfile

DEFS = """Assign exactly one failure label to this weight-editing episode, using these definitions (frozen in advance):
- success: the submitted edit raises MATH-500 by at least 20 points over the base (9.0 %), i.e. score >= 0.29.
- not localised: the editor never identified the looping mechanism (no analysis connecting the failure to repetition
  or non-termination in the model's thinking, or it settled on a wrong mechanism and kept it).
- localised, edit failed: the mechanism was identified (repetition / non-termination located in the model), but no
  submitted edit repairs it (including edits it tested and withdrew).
- budget exhausted: the forward passes or Python runs ran out before any validated edit existed.
Background: the unedited Qwen3.5-0.8B fails MATH-500 mostly because ~90 % of its answers loop verbatim inside the
thinking until the 32,768-token limit.
Answer with JSON only: {"label": "...", "justification": "<one paragraph citing call numbers / turns>",
"localised_at_call": <first call index where the looping mechanism is identified, or null>,
"edits_tried": "<short list>", "confidence": "high|medium|low"}"""


def digest(run, budget=300_000):
    rows = [json.loads(l) for l in open(os.path.join(run, "tool_calls.jsonl"))]
    per = max(600, budget // max(1, len(rows)))
    out = []
    for i, r in enumerate(rows):
        k = r.get("kind", "")
        if k == "python":
            body = (r.get("code") or (r.get("input") or {}).get("code") or "")[: per // 2]
            res = str(r.get("output") or r.get("reason") or "")[: per // 2]
            out.append(f"--- call {i} turn {r.get('turn')} python{' (BLOCKED)' if r.get('blocked') else ''}\n{body}\nOUT: {res}")
        else:
            s = json.dumps(r.get("summary") or r.get("input") or {})[: per // 3]
            out.append(f"--- call {i} turn {r.get('turn')} {k} ok={r.get('ok')}\n{s}")
    return "\n".join(out)[:budget]


def main():
    run = sys.argv[1]
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "claude-sonnet-5-5"
    fa = open(os.path.join(run, "final_answer.md")).read()[:20000] if os.path.exists(os.path.join(run, "final_answer.md")) else "(none)"
    sc = json.load(open(os.path.join(run, "eval", "score.json"))).get("score") if os.path.exists(os.path.join(run, "eval", "score.json")) else None
    st = json.load(open(os.path.join(run, "state.json")))
    user = (f"EPISODE {os.path.basename(run)}: MATH-500 score of the submission = {sc}; turns {st.get('turn')}.\n\n"
            f"FINAL ANSWER:\n{fa}\n\nTOOL CALLS (digest):\n{digest(run)}")
    with tempfile.TemporaryDirectory() as tmp:
        p = subprocess.run(["claude", "-p", "--safe-mode", "--model", model, "--system-prompt", DEFS, "--tools", "",
                            "--no-session-persistence", "--output-format", "json"],
                           input=user, capture_output=True, text=True, cwd=tmp, timeout=900)
    d = json.loads(p.stdout)
    t = str(d.get("result") or "")
    v = json.loads(t[t.find("{"): t.rfind("}") + 1])
    v.update(run=run, score=sc, labeller=f"claude -p {model}", note="proposed; reviewed before use")
    print(json.dumps(v))


if __name__ == "__main__":
    main()
