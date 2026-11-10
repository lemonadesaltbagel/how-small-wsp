# Decisions waiting for the user (newest first)

## 2026-11-10 13:55 — deepseek-flash hinted s1 stopped on 4 empty replies in a row (time-sensitive)
- Ended at turn 122 ("the editor sent empty replies") with **11/50 passes and 82/300 Python runs used**. In each of
  the last 4 replies DeepSeek wrote a complete ```python block **inside its reasoning field** and returned empty
  visible content (finish_reason "stop"); the harness's 3 nudges ("your last reply was empty") did not help.
  DeepSeek does this now and then: 9 of 122 turns here, 10 of 401 in its rules-v1 run, which never had 4 in a row.
- It was working on the right mechanism: loop positions, the stop-token logit at loop positions, attention heads that
  track lags in its own repeated text; 3 edits tested, edit 3 (amplify lag heads ×1.5) installed, pass 11 pending.
- Precedent: the smoke test's early stop on an empty reply (a literal `<|im_end|>`) was treated as an interface bug.
- Scoring of the edits as they stand is running (pods kept: agvu, h2r5).
- Options:
  - **A. Final.** The empty-reply stop is part of the frozen harness for every editor; label probably
    "localised, edit failed".
  - **B. Resume once (proposal).** Treat N empty replies with budget left as an interface failure, not an editor
    decision: resume once with the same harness and remaining budget (the counter restarts); a second such stop is
    final. Fixed now as a rule for all editors, before this run's score is known; reported in the paper. So far it
    affects only this run (every other finished episode ended with a final answer).
  - C. Change the harness to read a tool block from the reasoning field: rejected (changes the interface mid-matrix).

## 2026-11-10 13:20 — label boundary: "noticed non-termination" vs "localised" (9B hinted s1 and s2)
- Both 9B hinted seeds noticed that ~90 % of answers are cut off and wrote that the model "reasons indefinitely" /
  is "stuck in reasoning loops" (the hint's wording), but never looked at a cut-off generation and put the cause in an
  untested mechanism (a missing "answer state" / "completion signal" in layer-23 activation magnitudes).
- Labelled **not localised** (the definition: "settled on a wrong mechanism and kept it"). The alternative reading,
  "localised, edit failed", would count naming the symptom as identifying the mechanism.
- Proposal: keep "not localised"; "localised" needs the failure placed at looping positions or components.

## 2026-11-10 11:20 — protected-tensor edit attempts flagged by the audit (9B unhinted s1)
- Codex audit of `runs/phase0/qwen9b-unhinted-p50-strict2-s1`: 151 allowed / 0 suspicious / 2 violations.
- Both violations are **attempts to edit the protected embedding** (`embed_tokens.weight`): row 46 failed on a shape
  error before saving; row 49 was refused by the harness. Neither changed the network; the final edit set is
  layers.11 q_proj/v_proj only. Neither involves rule (a) (training) or (b) (simulation).
- Proposal: refused or failed attempts on protected tensors are recorded, not invalidating (they cannot affect the
  scored network). `audit_adjudication.proposed.json` in the run dir.
- Effect on results: none on the curve either way (6.6 %, not a repair); it decides whether this run counts as a
  valid 9B unhinted seed (else seed 3 needed for that cell).
