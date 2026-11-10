"""Prepare a run that the harness ended on empty replies for its one resume (analysis plan amendment A1). Stdlib only;
run it on the node that holds the run, after the stop-1 scoring has finished and the episode process has exited:
  python3 -I scripts/resume_after_empty.py <run dir>
1. refuses unless the run ended with "the editor sent empty replies", has forward passes left, has a stop-1 score,
   and has not been resumed this way before;
2. moves the stop-1 results into <run>/stop1/ (final/, eval/, final_answer.md; copies of state.json, loop.log and
   the edit ledger), so both results are kept;
3. re-opens state.json: done=false, end_reason=null, empty_streak=0 (the harness drops unknown state fields, so the
   resume is recorded in <run>/stop1/a1_resume.json).
Nothing else changes: the next message stays the harness's own empty-reply nudge, and `openwish run --resume` adds
the standard kernel-restart note and handoff. Then start the episode again with its usual launcher."""
import datetime
import json
import os
import shutil
import sys


def main():
    run = sys.argv[1]
    sp = os.path.join(run, "state.json")
    st = json.load(open(sp))
    left = 50 - sum((st.get("passes_used") or {}).values())
    problems = []
    if st.get("end_reason") != "the editor sent empty replies" or not st.get("done"):
        problems.append(f"end_reason is {st.get('end_reason')!r}")
    if left <= 0:
        problems.append("no forward passes left")
    if not os.path.exists(os.path.join(run, "eval", "score.json")):
        problems.append("no stop-1 score yet")
    if os.path.exists(os.path.join(run, "stop1")):
        problems.append("already resumed once under A1 (a second empty-reply stop is final)")
    if problems:
        sys.exit("refused: " + "; ".join(problems))
    s1 = os.path.join(run, "stop1")
    os.makedirs(s1)
    for name in ("final", "eval", "final_answer.md"):
        if os.path.exists(os.path.join(run, name)):
            shutil.move(os.path.join(run, name), os.path.join(s1, name))
    for name in ("state.json", "loop.log", os.path.join("edits", "ledger.jsonl")):
        if os.path.exists(os.path.join(run, name)):
            shutil.copyfile(os.path.join(run, name), os.path.join(s1, os.path.basename(name)))
    rec = {"at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "stop1_turn": st.get("turn"),
           "stop1_passes_used": st.get("passes_used"), "rule": "notes/analysis_plan.md amendment A1 (commit 854ce75)"}
    json.dump(rec, open(os.path.join(s1, "a1_resume.json"), "w"), indent=1)
    st.update(done=False, end_reason=None, empty_streak=0)
    tmp = sp + ".tmp"
    json.dump(st, open(tmp, "w"), indent=2)
    os.replace(tmp, sp)
    print(json.dumps({"run": run, "stop1": s1, "passes_left": left, "a1_resume": rec}))


if __name__ == "__main__":
    main()
