# Preprint review, round 2: the DS s2 additions (2026-11-13)

Reviewer: Codex (read-only), on `git diff 6ce9c71 edd513e -- paper/preprint.md`. It checked the new numbers against
the pending-decision entries, analysis.json, thinking_profile.json, the run dirs and the logs, and found the rest
matching.

| # | Point | Response |
|---|---|---|
| F1 | "Both gates fire inside loops": hinted s2 closes at once in 89 / 500 | Reworded: the gates are built from loop / doubt states; unhinted s2 never closes at once, hinted s2 in 1 of 6 |
| F2 | Spearman 0.23 is over all hinted s2 generations, not "where it closes after thinking" | Attributed to all generations |
| 1 | Keep validity visibly open (lam, 8 other note-(6) flags, skip question) | Kept [pending] in Tables 3 / 4 and the §4.2 / §5 text; no sentence adjudicates |
| 2 | Abstract and §4.2 still say 51.8 % is the sole score above threshold | Abstract rewritten: 3 of 4 DS episodes above threshold, one an excluded skip, two pending; §4.2 "the one" → "the first" |
| 3 | Labeller paragraph counts out of date | 15 doubly labelled; 7 localised; unhinted s2 27 vs 43, take 27 by the s1 reading (earliest separating evidence) |
| 4 | "Loop-gated" overstated for hinted s2 | Covered by F1's rewording |
| 5 | "What they lack is an edit that works" depends on the pending decisions | Limited to the seed-1 and 27B episodes, marked [revise after adjudications] |
