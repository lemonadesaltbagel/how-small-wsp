# Collision check, 2026-11-12 07:24 (after preprint review round 1)

Question: has anyone published (arXiv, blog, GitHub) a study that varies the *editor* in weight-space programming,
in particular open-weight / smaller editors on experiment 8, or audits editors for simulating the network?
Searches: web (extended) for weight-space programming + editor size / open-weight editors / Qwen / DeepSeek / gpt-oss /
audit; arXiv 2611.*; GitHub reproductions.

**Verdict: no collision.** Nothing found that varies editor size or uses open-weight editors systematically, and
nothing audits tool calls for network simulation. The closest items:

| work | what it does | overlap with us |
| --- | --- | --- |
| Hu et al., arXiv 2611.09382 (already cited) | one editor (Astra), targets Qwen3.5-0.8B / 9B, seven benchmark families | varies the target, not the editor |
| shapor/weight-space-experiments (GitHub) | blind tests with Astra, Opus, Fable on exps 1-6; Qwen MATH-500 controls | **thinking off: unedited 45.2 %, edited 44.2 %**; "the gain is mostly about finishing answers". Independent support for our thinking-off control (48.2 %). Cite. |
| Tree-Ising, Salloum & Ali, arXiv 2611.14210 (10 Nov) | no LLM: bit-level sign/exponent edits by Ising/QUBO optimisation on forward-pass statistics | non-agent baseline on the same task: 54.6-55.1 % vs the agent edit's 51.7 % (MATH-500@4, their harness). Cite as a non-agent comparison. |
| kyle-chalmers/weight-space-programming (GitHub) | replicates the p50c edit; random-direction controls; a blind Claude Code attempt | random edit on the same 64 neurons recovers ~1/5 of the gain; the blind Claude attempt did not find a fix (its version and budget not stated). Cite as a random-edit control. |
| Ahmedn1/wsplab (GitHub) | WSP framework; a vLLM-served Qwen3.5-0.8B "couldn't do the job" | one anecdote that a tiny open model fails; no systematic comparison. Cite in passing. |
| Ahmed, arXiv 2611.08070 (5 Nov) | Opus 5.5 reads small CNN weights as text, no tools | weight reading only, one model |
| arXiv 2611.05341, 2611.11604; Say43, XLRA, Exorust (GitHub) | WSP on clinical nets, policies with model checking, sycophancy, code, SQL | other tasks; frontier editors |

Implication for the paper: the shapor result shows that the repair threshold (29 %) is below what the base reaches
with thinking off, so meeting the threshold does not by itself show a repair of the thinking; our thinking-skip check
is the test that separates them. This is now in §2 and §4.2. Re-run this check before posting.

## Re-check 2026-11-14 14:10 (after preprint review round 2)
Searches (web, extended and standard): weight-space programming + open-weight / smaller editor, editor size, gpt-oss /
Qwen / DeepSeek editor, arXiv Nov 2026 LLM-edits-weights, reproductions with thinking-off controls, open WSP
benchmarks / leaderboards. **Still no collision**: every editing result found uses Astra (or Claude in one blind
attempt); no study varies the editor or audits tool calls for simulation. New items, none overlapping:
- korben.info report of a reviewer's thinking-off test (the edit costs ~2 points with thinking off), secondary only;
  consistent with shapor and our T0.
- buildmvpfast / several write-ups: the "loop fix, not ability" critique, which our thinking-off and truncation
  controls address.
- Willison's weight-reading replication with Qwen 3.8 27B (reading, no tools; his original is already cited).
- wsplab is on PyPI (alpha); no editor comparison.

## Re-check 2026-11-15 (before posting; web + GitHub repo search "weight-space programming", awesome-wsp)
- **Partial overlap, cite: WKJBryan/wsp-lab** (GitHub, phase 4 results 2026-11-12/13; phase 9 "team deliberation"
  2026-11-14). Cross-editor, no-hint Qwen3.5-0.8B repair with GLM-5.3-Flash, mimo-v2.6-flash, mimo-v2.6-pro (one
  episode each, 10 passes x 64 practice problems): all three rescale the tied-embedding row of `</think>` (~x1.5) →
  MATH-500 25.2 / 27.8 / 27.8 % (their base 6.0 %, published p50c 39.4 % on their stack; random matched-norm 2.2 %).
  Differences: embedding editable (our task protects it; such an edit would be refused by our save_edit), offline
  generation in the sandbox allowed (their ISSUES #12), no size ladder, no audit for simulation, no thinking-off or
  truncation control. Our contribution 1 (scale curve) and 2 (audit) stand; added to §2 as a community replication.
- Other new repos, no overlap: atharva-naik/auto-weight-space-programming (a 4B model trained with LoRA to read its own
  neurons), FiveTechSoft/weights (delta-to-LoRA transplant), dljantz/wsp-gym, dspy-weightspace (frontier editor),
  awesome-wsp lists (SajjadPSavoji, brighterr; mirrors).
- Web search: still only Astra-based editing results plus write-ups.
- **Closest so far, cite: Prayas-Agrawal/weight-whisperer** (GitHub, created 2026-11-11, pushed 11-12; found via
  SajjadPSavoji/awesome-weight-space-programming). Same task, open-weight editor Qwen3.8-Flash-Next-FP8 (185.5 GB
  weights, TP 4), 2 runs: MATH-500 7.70 → 50.40 / 51.55 % (4 samples, authors' sampling). Edit W += a·e(</think>)·vᵀ
  at layer 23, v a loop-vs-productive activation contrast (the DS-style loop-gated close). Differences: no pass
  budget (editor submits when it likes); the prompt names the failure and says "Check what the proposed change does at
  observed states before spending a generation call" while stating rule (b), i.e. via the (unmetered) forward tool;
  in our harness that costs passes, and doing it in the kernel is the note-(6) class; no audit; no
  thinking-off / truncation control (their scores are at our T0 48.95 %); one editor, one size. Headline "Open models
  can do weight-space programming too" overlaps our question; our curve (fixed budget, strict rules, audit, controls,
  size ladder) is not pre-empted. Added to §2 and the introduction.
- Unjuno/beyond-math-weight-space-programming: audit of the released p50c evidence (reconstruction, small local
  evals); no editor comparison. No overlap.
