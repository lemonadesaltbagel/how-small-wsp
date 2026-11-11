# Collision check, 2026-11-12 ~08:00 (after preprint review round 1)

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
