# Preprint review round 1 (Codex, read-only, 2026-11-12 ~07:40): what changed

Review: `notes/preprint_review_codex_r1.md` (major revision; main numbers verified against the ledger).

Applied to `paper/preprint.md`:
- Facts: 27B hinted s2 saved edits 28 -> 17 (the 28 counted reverts; saved = replacements, defined in §3.1, reverts
  excluded in every row). 9B "never reads a cut-off generation" -> prints part of one (hinted s1, call 94) but never
  examines the repetition or ties a component to it. 9B Python runs 102–182. 27B self-report: material 0/0/3/4 out of
  2/3/6/7 contradicted. Opus never-close counts split: strict 24/10/194, first round 175/270.
- Overreach: abstract and §5 no longer say "split at 27B" or "change with size"; they describe the sample and say it
  does not locate a threshold. deepseek-flash called a separate cross-family point. Thinking-off control "corroborates"
  (the 473/500 early closes are the main evidence), with the non-reproducibility of separate passes stated. Opus
  comparison: other differences (family, hinted only, rule version) stated; shows feasibility, not that size causes
  the gap. Self-report: "at least one episode at every size".
- Missing: Table 5 (effort and tool use per episode; 258 of 278 failed forward requests are malformed); repairs per cell
  (0/2); the "within noise" definition (2 SE, SE 1.3); Table 1 (conformance); comparability deviations (20 GB kernel
  cap vs 96 GB, notices up front, own unhinted prompt); definitions of pass, saved edit, and of a threshold-meeting
  shortcut as not a repair; a limitation on missing baselines and on malformed requests.
- Clarity: "strict2" -> "main protocol" (named once); A3/A5 described in words; note 5 glossed; plain sentence on what
  the Opus loop gate is; LnHm notation.
- Consistency: Tables 1–5 numbered and referenced; Figure 1 caption (planned seeds, completed shown); §3.2 cross-refs
  to §4.2 and §5; edit verification now covers all 14 scored main-protocol episodes (all ok, `notes/verify_edits.jsonl`).

Not applied (waiting):
- Citations (placeholders stay; the user may prefer specific references).
- Opus `localised_at_call` under the same labelling procedure: needs the Claude labeller, which waits until both DS s2
  episodes end (shared Claude account with the laptop's reviewer).
- Uncertainty intervals per cell beyond the SE definition: after the DS s2 seeds.
