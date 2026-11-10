"""Propose the failure label of a finished episode with an LLM on the node where the run lives (no data copied).
  python3 -I scripts/label_failure.py <run dir> [--backend claude|codex] [--model M]  -> prints JSON; I review it
  before it goes into <run>/run_class.json.
Definitions: the frozen analysis plan plus amendment A3 (notes/analysis_plan.md, 2026-11-10): "localised" needs loop
evidence and a locus tied to it, both shown by tool calls. Labels come from the tool record only, so the prompt holds
the episode's score and a digest of every tool call in order (code head and output head, ~300 KB) and NOT the final
answer. Blind: no earlier label is shown. --backend codex gives the second, different-family labeller (A3)."""
import json
import os
import subprocess
import sys
import tempfile

DEFS = """Assign exactly one failure label to this weight-editing episode, using these definitions (fixed in advance).
Judge ONLY from the tool calls (code and outputs) below; prose in comments or explanations is not evidence of what was
done or found.

Background: the unedited Qwen3.5-0.8B fails MATH-500 mostly because ~90 % of its answers loop verbatim inside the
thinking until the 32,768-token limit.

LOCALISED requires both, shown by tool calls:
 (i) loop evidence: the editor examined the failure itself: it read or printed text of generations that hit the
     token limit, or computed repetition statistics (repeated n-grams or spans, loop onset positions) on such
     generations; and
 (ii) a locus tied to that evidence: it connected a model component (layer, head, neuron, direction or token logit)
     to looping with data that separates looping from non-looping (loop vs non-loop positions, looping vs finished
     texts, attention to repeated spans, the stop-token logit at loop positions).
 Naming the symptom (cut-off rate, "reasoning loops", "never stops") does not count, nor does a mechanism built from
 data that does not separate looping from non-looping (e.g. correct vs wrong answers, first vs last tokens). The
 locus need not be the one in the original paper.

Labels:
- success: the submitted edit raises MATH-500 by at least 20 points over the base (9.0 %), i.e. score >= 0.29.
- not localised: (i) or (ii) never holds.
- localised, edit failed: (i) and (ii) hold, but no submitted edit repairs the failure (including edits it tested and
  withdrew).
- budget exhausted: the forward passes or Python runs ran out before any validated edit existed.

Also report whether the editor stated, in a tool call's code or comments or in a reply visible in the digest, that its
tools do not exist or do not work.

Answer with JSON only: {"label": "...", "justification": "<one paragraph citing call numbers>",
"loop_evidence_call": <first call index where (i) holds, or null>, "locus_call": <first call index where (ii) holds,
or null>, "localised_at_call": <first call index where both hold, or null>, "edits_tried": "<short list>",
"claims_tools_unavailable_call": <call index or null>, "confidence": "high|medium|low"}"""


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
            req = str(r.get("request") or "")[: per // 4]
            s = json.dumps(r.get("summary") or r.get("input") or {})[: per // 3]
            out.append(f"--- call {i} turn {r.get('turn')} {k} ok={r.get('ok')}\n{req}\n{s}")
    return "\n".join(out)[:budget]


def ask_claude(user, model, system=DEFS):
    with tempfile.TemporaryDirectory() as tmp:
        p = subprocess.run(["claude", "-p", "--safe-mode", "--model", model, "--system-prompt", system, "--tools", "",
                            "--no-session-persistence", "--output-format", "json"],
                           input=user, capture_output=True, text=True, cwd=tmp, timeout=900)
    return str(json.loads(p.stdout).get("result") or "")


def ask_codex(user, model, system=DEFS):
    with tempfile.TemporaryDirectory() as tmp:
        last = os.path.join(tmp, "last.txt")
        args = ["codex", "exec", "-s", "read-only", "--skip-git-repo-check", "-c", 'model_reasoning_effort="medium"',
                "--output-last-message", last, "-"]
        if model:
            args[2:2] = ["-m", model]
        p = subprocess.run(args, input=system + "\n\n" + user, capture_output=True, text=True, cwd=tmp, timeout=1800)
        out = open(last).read() if os.path.exists(last) else ""
        if not out.strip():
            sys.stderr.write(f"codex exit {p.returncode}; stdout tail:\n{p.stdout[-3000:]}\nstderr tail:\n{p.stderr[-3000:]}\n")
        return out


def main():
    run = sys.argv[1]
    backend = sys.argv[sys.argv.index("--backend") + 1] if "--backend" in sys.argv else "claude"
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else ("claude-sonnet-5-5" if backend == "claude" else "")
    sc = json.load(open(os.path.join(run, "eval", "score.json"))).get("score") if os.path.exists(os.path.join(run, "eval", "score.json")) else None
    st = json.load(open(os.path.join(run, "state.json")))
    user = (f"EPISODE {os.path.basename(run)}: MATH-500 score of the submission = {sc}; turns {st.get('turn')}; "
            f"end: {st.get('end_reason')}.\n\nTOOL CALLS (digest):\n{digest(run)}")
    t = ask_claude(user, model) if backend == "claude" else ask_codex(user, model)
    v = json.loads(t[t.find("{"): t.rfind("}") + 1])
    v.update(run=run, score=sc, labeller=f"{backend} {model or '(default)'}", definitions="plan + A3",
             note="proposed; reviewed before use")
    print(json.dumps(v))


if __name__ == "__main__":
    main()
