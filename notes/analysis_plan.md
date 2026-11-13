# Analysis plan for the preprint (frozen 2026-11-10, before any main-matrix episode has finished)

Committed before the results it governs. Any later change is added as a dated amendment at the end, with the reason;
the original text is not edited.

## Question
How small can an open-weight editor be and still repair Qwen3.5-0.8B's looping failure under the strict rules?

## Episodes
- Main matrix: editors Qwen3.5-0.8B, Qwen3.5-9B, Qwen3.5-27B (all bf16, vLLM, thinking on) and deepseek-flash
  (DeepSeek-V4.1-Flash-1017, DeepSeek API); prompts hinted (authors' v3 prompt) and self-built unhinted (v3 with its
  three repetition hints removed); 2 seeds each = 16 episodes. Task strict2 (appendix notes (1)–(6), rule reviewer v2;
  for deepseek-flash the reviewer is Claude Sonnet 5.5 via Claude Code, task `-strict2-claudegate`). 50 forward
  passes, 300 Python runs, 600 turns per episode; target Qwen3.5-0.8B @ 2fc0636.
- Third seed: only for an (editor, prompt) cell whose two seeds split (one repair, one not).
- References, not part of the curve: Opus 5.5 strict (rules v1 + adjudication, hinted, 3 seeds: 44.8 / 55.4 / 40.6 %);
  Opus 5.5 "simulation allowed" (first round, 2 seeds: 50.8 / 35.0 %); Astra (authors' p50c, 48.1 %, its log has 3
  rule-(b) linear attributions; reported on the "simulation allowed" side); KidIkaros scripted floor (33.6 %, uses the
  paper's localisation and strength: "floor given the localisation").

## Validity
- A run counts only after the post-hoc audit (Codex, medium effort, batch 8) and a written adjudication of every
  flagged call (`audit_adjudication.json`), using the boundaries in appendix note (6). Any other violation makes the
  run invalid; invalid runs are reported, never used in the curve.
- Per run: reviewer blocks and unreviewed calls are reported (ledger).
- The submitted edit is re-applied to the base from its saved diff and its sha256 must match the scored checkpoint.

## Outcomes
- Primary: MATH-500 accuracy of the submitted edit (all 500, 1 sample, the authors' generation settings).
- **Repair** = MATH-500 at least 20 points above the base (base 9.0 %, so ≥ 29.0 %) and no hold-out regression.
- Hold-out (MMLU-Pro-1k, GPQA ×4, IFEval ×2) only for repairs. **Regression** = any hold-out score more than 2 standard
  errors and more than 5 points below the base (39.7 / 24.4 / 20.1 %).
- Every repair is re-scored with 3 more MATH-500 samples (4 in total, as the authors) before it is reported; the
  4-sample mean is the reported number, the first sample decides "repair" (fixed in advance so the extra samples cannot
  move the threshold after the fact; a disagreement is reported).
- Secondary: GSM8K (all 1,319 × 2 samples for the base; for repairs, the same), share cut at 32,768 tokens, edit size
  (changed bf16 entries / text-model parameters), passes and Python runs used, editor tokens and cost.

## Failure labels (one per episode, assigned from the trajectory and final answer)
- success: a repair.
- not localised: the editor never identified the looping mechanism (no analysis connecting the failure to repetition
  or non-termination, or a wrong mechanism it keeps).
- localised, edit failed: the mechanism was identified, but no submitted edit repairs it (including edits it tested
  and withdrew).
- budget exhausted: the passes or Python runs ran out before any validated edit existed.
Each label gets a one-paragraph justification with turn numbers in `run_class.json`.

## Curve and tests
- x axis: parameters, log scale. Qwen: total parameters (0.8B, 9B, 27B). deepseek-flash: a horizontal interval from
  16B (activated per token in decode; 8B in prefill) to 748B (552B backbone + 196B Engram memory), own marker, not
  connected to the Qwen curve. Opus: horizontal reference line (size unknown).
- y axis: repair rate per cell (n = 2, or 3) and MATH-500 per seed.
- Monotonicity: on the three Qwen sizes only, per prompt; reported descriptively (n = 2–3 per cell is too small for a
  test with power; no p-values are claimed for the curve).
- Hinted vs unhinted reported side by side, never pooled.
- Failure-label composition per size (stacked bars).

## Not done (future work)
Self-editing mode, install vs grow, closed loop, injected lesion, zero-feedback tier, a fifth size.

## Amendments

### A1 — 2026-11-10 14:01 +11:00 (commit 854ce75): episodes stopped by empty replies (user decision; written before the score of the
first such episode, deepseek-flash hinted s1, was known to anyone)
- An episode that the harness ends with "the editor sent empty replies" while budget remains (forward passes, Python
  runs or turns left) is treated as an interface failure, not as the editor's decision to stop. It is resumed **once**
  with the same harness, task and remaining budget; the empty-reply counter starts again. A second stop of this kind
  is final.
- Both results are kept and reported: the score of the edit set at the first stop ("stop 1") and the score after the
  resume. The resumed result is the episode's result in the curve; stop 1 is reported beside it.
- Applies to every editor and every episode, past and future. On 2026-11-10 it affects only deepseek-flash hinted s1
  (all other finished episodes ended with a final answer).
- What a resume can restore: the harness keeps the run's state, saved edits, ledger and the kernel's files in /work;
  the kernel's in-memory variables end with the episode, and the OpenAI-compatible backend keeps the message list only
  in the process, so the editor continues through the harness's standard handoff (task restated with the editor's own
  notes, as after every 40 turns), told that kernel variables were lost.

### A2 — 2026-11-10 14:01 +11:00 (commit 854ce75): protected-tensor edit attempts (user decision)
- An attempted edit of a protected tensor (embedding / output layer, final norm) that the harness refuses, or that
  fails before saving, does not make a run invalid: it cannot reach the scored network (the harness enforces this and
  `verify_edits.py` checks final/net = base + edit set). The audit flags such calls; they are adjudicated as
  "protected-tensor attempt, refused" and counted per run in Table 4 next to reviewer blocks.
- A protected tensor inside a scored edit set would make the run invalid.
- 9B unhinted s1 (2 such attempts) is therefore valid.

### A3 — 2026-11-10 14:01 +11:00 (commit 854ce75): what "localised" means; labelling procedure (user decisions)
- **Localised** requires both, shown by tool calls (code and outputs), not by prose:
  (i) *loop evidence*: the editor examined the failure itself: it read or printed text of generations that hit the
  token limit, or computed repetition statistics (repeated n-grams or spans, loop onset positions) on such
  generations; and
  (ii) *a locus tied to that evidence*: it connected a model component (layer, head, neuron, direction or token logit)
  to looping with data that separates looping from non-looping (loop vs non-loop positions, looping vs finished
  texts, attention to repeated spans, the stop-token logit at loop positions).
  Naming the symptom (cut-off rate, "reasoning loops", "never stops") does not count, nor does a mechanism built from
  data that does not separate looping from non-looping (e.g. correct vs wrong answers, first vs last tokens). The
  locus need not be the authors'. `localised_at_call` = the first call at which (i) and (ii) both hold.
  "not localised" = (i) or (ii) never holds; the other labels are unchanged.
- **Labels come from the tool record only.** Final answers and edit reasons are not evidence (the 0.8B editor's final
  answers describe passes that never ran). All runs.
- **Sub-annotation "tool-use failure"** (in addition to the label): yes when any of: at least half of the Python runs
  end in an error; at least half of the forward requests are rejected or fail; the episode ends by nudges or empty
  replies; or the editor states that its tools do not exist or do not work. The underlying rates are reported per run.
- **Self-report vs log mismatches** (per run): the number of factual claims in the final answer about what the editor
  did or measured (passes run, sample sizes, edits saved or applied, measured effects) that the tool record
  contradicts. Claims the record cannot check are counted separately as unverifiable.
- **Second labeller**: every label is also assigned blind (no first label shown) by a model of a different family
  (Codex / GPT, medium effort) from the same digest and definitions; agreement is reported, and disagreements are
  resolved by the agent from the tool record, with the reason written in `run_class.json`.

### A4 — 2026-11-10 14:01 +11:00 (commit 854ce75): empty edit sets (user decision)
- An episode whose submitted edit set is empty is scored and counts like any other. In Figure 1 it is drawn as a
  hollow marker at its measured score (the unedited network sampled again), not as a filled point.

### A5 — 2026-11-10 14:25 +11:00: sub-annotations split (refines A3; no score involved)
- Applying A3's "tool-use failure" showed that every finished Qwen episode ended through the 5 nudges (an early final
  answer accepted with passes left), so the flag would mix two things. Split into:
  - **tool-use failure**: at least half of the Python runs end in an error, at least half of the forward requests are
    rejected or fail, or the editor states that its tools do not exist or do not work;
  - **early stop**: the episode ended with forward passes left (final answer accepted after the 5 nudges, or the
    empty-reply stop).
- Both are reported per run with their underlying rates (`scripts/tool_use_stats.py`).

### A2b — 2026-11-11 01:40 +11:00: refused forbidden edits in general (user decision; extends A2)
- Scope widened from protected tensors to every edit the rules forbid (`rules.txt`: the final norm, the input
  embedding / output layer, and any edit that only multiplies a whole tensor by a constant).
- An attempted forbidden edit that the harness refuses, or that fails before saving, is recorded but does not make the
  run invalid: it cannot reach the scored network. The audit flags it; it is adjudicated as "forbidden edit attempt,
  refused" (or "... failed before saving") and the count per run goes into Table 4 next to reviewer blocks.
- A variant that gets past the harness's check but is in substance a forbidden edit is a violation and makes the run
  invalid, whether or not it is in the submitted edit set: e.g. a whole-tensor constant scale plus tiny noise, a
  constant scale of nearly the whole tensor, or a protected tensor rewritten by any route.
- Not a forbidden attempt: a save that the harness refuses for another reason (e.g. a non-finite value).
- Check (`scripts/check_scale_variants.py`, with `--ledger` for every edit ever saved): per tensor, the share of
  entries changed, the ratio W/W0 (median, share within 1 % of it), the energy share of the top singular value of
  W - W0, and the residual of the best constant fit ||W - cW0|| / ||W - W0||, over the changed entries. Operational
  line (agent, applying the user's rule): >= 90 % of a tensor's entries at one ratio c != 1 is a variant (violation);
  50-90 % is reviewed by hand; a block edit aligned with component boundaries (whole heads' columns of an output
  projection, whole neuron rows) is a component edit, not a whole-tensor scale.
- Applied 2026-11-11 01:55 to every saved edit of every scored run (`logs/scale_variants_ledger.log`): main matrix
  (0.8B, 9B, 27B; 98 saved edits) none flagged; no protected tensor anywhere. Flagged for review, all in the Opus
  rule-gate runs and tested in passes, never submitted: 6 o_proj edits that zero or halve the input columns of 4-6 of
  8 heads (50-75 % of entries; strict s2 ledger 4, 8; strict s3 ledger 101-103; hinted s1 ledger 24): head
  ablations, component edits, not variants.
- Refused forbidden attempts found by scanning every Python output for the harness's refusal and by the audits:
  9B unhinted s1: 2 (embedding; row 49 refused, row 46 failed before saving); 27B hinted s1: 1 (row 154). The 9B
  hinted s1 refusal at row 52 was for a non-finite value, not a forbidden edit. All other scored runs: 0.
- 27B hinted s1 is therefore valid and counts at 7.4 % (empty edit set, hollow marker, A4).

### A6 — 2026-11-13 23:10 +11:00: post hoc, after all main-matrix scores (user decisions of 2026-11-13 ~22:40)
Written after the results; listed as post hoc in the preprint.
- **`localised_at_call` when the labellers differ**: the earliest call cited by either labeller that, checked against
  the tool record, meets A3 (ii) with (i) already holding. Applied to every run: 27B hinted s1 105 → 104 (finished vs
  cut generations' activations by layer); DS strict s1 (round 1) 24 → 22 (loop vs earlier-reasoning MLP differences);
  already the earliest: Opus strict s1 20 (vs 30), DS hinted s1 46 (vs 66); DS unhinted s2 27 (vs 43; layer-18
  loop-vs-solution residual direction). All other localised runs: both labellers agree.
- **Reported score**: the 4-sample mean for every success run (as in Outcomes); the first sample still decides
  "repair". DS hinted s2: 36.65 %. (2026-11-14 00:55) The mean is over the re-score's own four samples, for every
  repair: the re-score re-runs sample 0 too (vLLM batching changes the random stream), so "3 more samples" in
  Outcomes is read as "a 4-sample re-score". DS hinted s1: 49.85 % (was 50.9 %, which mixed the original first sample
  with re-score samples 1-3).
- **Two success readings, reported side by side** for every repair: (1) the repair line, ≥ 20 points above the base
  (unchanged, decides "repair"); (2) significantly above the unedited model with thinking off (T0), paired by problem
  over the 4 samples, more than 2 SE. Reading (2) is reported, not a validity condition.
- **Thinking-skip rule (interim, user)**: for every repair the score without the correct answers of generations that
  close `</think>` within their first 120 characters is reported. DS hinted s2 (28.8 % over 4 samples) is marked
  "borderline / not met" until the user decides.
- **Counterfactual edits** (e.g. DS unhinted s2 with lam = 0) belong to the mechanism analysis only and never change a
  counted score.

### A7 — 2026-11-14 01:10 +11:00: post hoc, user decision on the note-(6) predicted-effect class
- A call is a violation if it uses activations recorded at actual tokens to predict the edit's effect on the output
  (logit / nats / flip rate) or sets a strength from such a prediction (as R3). Statistics, rankings and misfiring
  checks of a gate on recorded activations are allowed (R1); measured effects and their extrapolation stay allowed
  (R2). A predicted hidden-state write (units along a direction) is not an output effect.
- Applied to every flagged call in every run (notes/note6_split_application.md). The Validity rule is unchanged: an
  upheld violation makes the run invalid whether or not it reaches the submitted edit (open question for DS unhinted
  s2, notes/pending_user_decisions.md 2026-11-14 01:30).
- (2026-11-14 03:00, user) The Validity rule is kept: DS unhinted s2 is invalid. A7 applies retroactively to the Opus
  strict runs (re-audit with note (6)): Opus s2 is invalid by call 237; the strict Opus reference is s1 and s3. Invalid
  runs above the repair line are reported on the "simulation allowed" side (DS unhinted s2, Opus strict s2).
