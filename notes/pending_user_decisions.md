# Decisions waiting for the user (newest first)

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
