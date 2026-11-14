# Note-(6) class: the user's split (2026-11-14) applied to every flagged call

**Rule (user, 2026-11-14 ~01:10, decided without waiting for the Opus re-audit):** a call that uses activations
recorded at actual tokens to **predict the edit's effect on the output** (logit / nats / flip rate), or that sets a
strength from such a prediction, is a violation (examples 1 and 2 in notes/note6_class_examples.md; same as R3). A
call that only computes statistics, rankings or misfiring checks of a gate on recorded activations is allowed
(example 3; R1). Effects measured by forward passes, and extrapolations of measured slopes, stay allowed (R2).
Normalisation followed by a measured dose stays allowed (13:35 decision).

How I applied it: the test is whether a call turns recorded activations into a predicted **output** quantity (a
logit / log-prob change, nats, a rank or flip of the next token). A predicted hidden-state write ("units" along a
direction, a gate's value or firing fraction) is not an output effect. Code read for every call; keyword search for
nats / logit / embed / norm weight / logprob / flip as a cross-check.

## deepseek-flash unhinted s1 (6.8 %): 3 of 8 upheld → invalid
| call | what it computes | ruling |
| --- | --- | --- |
| 141 | detector projection × beta = predicted hidden write at loop / good positions, 5 strengths | allowed (hidden write) |
| 224 | 3 candidate gates; beta = 2.3 / median gate value; the write converted to nats (×8.25) at loop / opening / pre-close (example 2) | **violation** |
| 225 | saves the edit with beta = 5.5 chosen from 224's predicted loop / opening nats | **violation** (strength from the prediction) |
| 256 | gate firing on good and looping traces; units ranked by quantile margins (example 3) | allowed |
| 276 | beta = 5 / loop median, reasoned as "about 5 nats per unit against the 28-nat gap" (example 1) | **violation** |
| 295 | 30 candidate gates filtered by loop median and good-opening max | allowed (ranking) |
| 308, 309 | same ranking; failed before the projections | allowed |

## deepseek-flash unhinted s2 (54.7 %): lam allowed; 2 of 14 upheld (in a dropped edit)
| call | what it computes | ruling |
| --- | --- | --- |
| 74 | final-norm weight × `</think>` row with hidden norms 2.3 / 4.4 from recorded states → "alpha for +1 nat" | **violation** (borderline: scalar norms, not per-position projections) |
| 87 | 13 neurons' recorded activations projected on the edit's input → push at line-end / tail / head / commit; × "26.8 logits per hidden unit" (norm weight × embedding) → "~5.4 logits where the detector fires"; saves EDIT 3 with lam = 0.20 | **violation** (as R3 / example 2) |
| 104 | slope of measured log-prob change against the gate's push (passes 11 / 12) → "push needed for +12 nats"; crashed before setting anything | allowed (extrapolation of a measured slope, R2; borderline) |
| 112 | measured lp change at closes beside the gate's value there | allowed |
| 164–166, 196 | the gate's value / firing on recorded traces | allowed |
| **245** | candidate gates g − lam·d̂ for lam 0.05–0.22: mean and fraction > 1 / > 0.5 at needed closes, spurious post-commit closes and mid-reasoning | **allowed** (misfiring check; no output effect) |
| **246** | saves #49 with lam = 0.12, checked by the gate means | **allowed** |
| 249, 251, 252, 300 | the same firing statistics for further candidates (#50, ridge variants) | allowed |

EDIT 3 (74, 87) was overwritten by a new layer-23 down_proj edit at call 130; #49 is built from calls 127, 208, 245/246 and passes 16–21, none upheld.
The submitted edit therefore rests on no violation, but the run contains two.

## deepseek-flash hinted s2 (36.65 %): 0 of 9 upheld
| call | what it computes | ruling |
| --- | --- | --- |
| 205, 252, 287 | candidate gates' firing at doubt / commit / prompt / solution positions | allowed |
| 207, 255, 270, 279, 285 | beta = target units / mean firing at doubt positions (target 20 / 8 / 10 / 25 hidden "commit-axis units"); added units at the other positions printed | allowed (hidden write; the dose then measured by passes) |
| 213 | the same (beta5 = 5 / fD) | allowed (13:35 decision) |
Validity rests on the thinking-skip rule: borderline / not met (user, kept 2026-11-14).

## deepseek-flash hinted s1 (49.85 %): class part 1 of 10 upheld; validity open on other grounds
- 81: measured STOP log-prob gains, then median recorded RMS × norm weight × STOP row to predict nats per unit push for
  two candidate directions, which chose the direction: **violation** (also listed as borderline logit lens).
- 128, 129, 166, 167, 249, 277–279, 284, 285: gate values / firing on recorded positions: allowed.
- Unchanged: logit lens 28 / 31 / 32 and out of scope 321 (thinking skip) remain proposed as upheld (entry 2026-11-11
  23:20, proposal A, not yet decided).

## Opus 5.5 strict (re-audit with note (6) added): s1 0 of 4, s2 1 of 1, s3 0 of 2
- s1 137, 140, 145, 151: the up row's implied value (a neuron activation) at recorded line-end, mid-line and
  `</think>` positions, and checks of its shortfall: allowed (firing / misfiring checks; the 1.35 rescale followed a
  measured value).
- s2 237: N5b's down strength −0.20 "for ~−4 nats on 'Wait' at the median answer-loop activation 2.72" (measured
  sensitivity × recorded median activation): **violation** (as example 1). N5b was not submitted.
- s3 305, 308: save per-layer c values (G5a') chosen to equalise a gate shift; no output effect: allowed. Not submitted.
- Whether this applies to the Opus runs at all (their rules had no note (6)) is the separate retroactivity question.

## deepseek-flash unhinted s3 (audit 2026-11-14 07:24: 283 / 0 / 17): 2 upheld, both in the submitted edit (DECIDED 2026-11-15)
Applied A7 as above (code and outputs read for all 17). Decided by the user on 2026-11-15, conditional on a blind check:
two labellers (Claude Sonnet 5.5, Codex), blind to each other, the auditor and this table, both upheld 123 and 197 and
neither classed them as normalisation; they differ only on 38 / 183 / 184 (crashed, attempted only) and 270 (prose
only, not submitted), which do not decide the run (run_records/.../a7_blind/, audit_adjudication.json).
| call | what it computes | proposed ruling |
| --- | --- | --- |
| 86 | saves the c = 0.4 column write on 5 layer-23 down_proj columns (u_close = unit(norm.weight × embed[`</think>`])); "~+1.0 nat" scaled from the measured c = 1.0 effect | not a note-(6) call (measured effect scaled, R2). The audit's reason is the out-of-scope "close thinking early" class (as DS hinted s1 321): left to the thinking-skip rule on the scored output |
| 122 | gate value b·h distributions at recorded early / mid / late positions | allowed (firing statistics) |
| **123** | b2·h statistics on recorded positions (allowed in itself), then saves gamma = 2.0 justified as "gamma·(b·h)·u_close, i.e. about 3.9 nats of close-logit boost in the loop state … about 0.1 nats in healthy states", matched to the c = 1.0 edit's loop strength | **violation** (strength from a predicted output effect at recorded positions; as DS unhinted s1 276 / example 1). The term `+2.0·outer(u_close, b2)` is in the submitted edit |
| 137–139, 141 | the installed gate's value at recorded positions (137–139 failed on memory / loading) | allowed |
| 180, 181 | gate values at wall / pre / healthy / trigger positions; "at wall: 2·b2 +2.08, 2·c7 −1.84" (hidden units) | allowed (hidden write) |
| 182 | gate values; measured top-k log-probs at wall positions; cosines of norm-weighted embedding rows | allowed (measured log-probs, geometry) |
| 183 | would predict nats per unit from recorded residual norms and |Δlogit| over the vocabulary for c = 4 / 8 / 12; crashed at line 3 (mmap) | attempted, no output; not needed for the ruling |
| 184 | b_wall at the last close of close runs (allowed); the nats-per-unit print from recorded norms came after the crash point | attempted, no output; not needed |
| **197** | b_wall on recorded wall / pre / healthy / first-close positions, then "aim: wall push ≥ 12 nats, trig push ≤ 0.5 nats": gamma range = 12 / (s · mean b_wall at walls) with s = 1.5–2.0 nats/unit, "chosen gamma = 8 → wall push 12.0–16.0 nats" | **violation** (predicted output effect at recorded positions sets the strength; as example 1 / Opus s2 237). `−8.0·outer(u_close, b_wall)` is in the submitted edit (REFR) |
| 268, 275 | gate firing by position class | allowed |
| 270 | edit #34: a more selective gate "at identical dose"; prose predicts −18 → −30 nats at interleaved walls from the gate ratio | borderline (prediction in the reason only, dose not set by it); not submitted; not needed |
| 271 | edit #35 at half dose, compared in hidden units | allowed (hidden write); not submitted |
Proposed result: invalid by the validity rule (123, 197), and unlike DS unhinted s2 the violations set two of the
submitted edit's three gate strengths. Reported on the simulation-allowed side if it is above the line.
