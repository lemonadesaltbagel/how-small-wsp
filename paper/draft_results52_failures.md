# Draft: §5.2 scale curve and §6 failure modes (2026-11-10, partial)

Working draft. Filled from finished episodes only; [brackets] = still running or not decided. Sources:
`notes/episode_ledger.json`, each run's `run_class.json` (failure label + my review), `state.json`, `loop.log`.

## 5.2 The scale curve (Figure 1)

[Figure 1: MATH-500 of the submitted edit against editor size, hinted and unhinted, 2 seeds per cell; deepseek-flash as
a horizontal interval from 16B activated to 748B total parameters; reference lines for Opus 5.5 strict, Astra, the
scripted floor, the repair threshold and the base.]

Episodes finished so far (strict2, all audited):

| editor | prompt | seed | MATH-500 | label | turns | Python runs | passes | edits saved | audit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B | hinted | 1 | 7.0 % | not localised | 49 | 34 | 0 | 0 | 34/0/0 |
| Qwen3.5-0.8B | hinted | 2 | [scoring] | [not localised] | 7 | 0 | 0 | 0 | — (no Python) |
| Qwen3.5-9B | hinted | 1 | 8.2 % | not localised | 360 | 182 | 20 | 6 tensors | 182/0/0 |
| Qwen3.5-9B | hinted | 2 | 6.2 % | not localised | 236 | 102 | 15 | 2 | 102/0/0 |
| Qwen3.5-9B | unhinted | 1 | 6.6 % | not localised | 274 | 153 | 18 | [n] | 151/0/2 [validity pending] |
| Qwen3.5-9B | unhinted | 2 | 7.8 % | not localised | 35 | 17 | 3 | 0 | 17/0/0 |
| Qwen3.5-27B | hinted | 1 | [running] | | | | | | |
| Qwen3.5-27B | unhinted | 1 | [running] | | | | | | |
| deepseek-flash | hinted | 1 | [running] | | | | | | |

No open-weight episode so far comes near the repair threshold (29.0 %); every finished score lies within sampling
noise of the base (9.0 ± 1.3 %, one sample) or slightly below it. [Final sentence on the smallest repairing size per
prompt once 27B and deepseek-flash finish.]

## 6 Failure modes across scale

Every finished open-weight episode so far is **not localised**: the editor never connected the failure to repetition
inside the model's thinking. How they fail differs with size.

**0.8B: no contact with the task.** The 0.8B editor ends its episodes within minutes. In seed 1 it ran 34 Python
blocks, mostly syntax errors or scripts that print a dictionary holding its opening guess (that the gated-attention
layers carry "lookahead loops"); it never loaded a tensor it then changed, used no forward pass and submitted an
empty edit, so its 7.0 % is the unedited network sampled again. In seed 2 it ran no code at all and stopped after
7 turns. Both final answers describe work that never happened: a "512-sample generation pass", "30 forward passes
completed", edits to attention and normalisation layers. The harness log shows none of it; we treat the final answer
as unreliable and label from the tool record only.

**9B: activity without diagnosis.** The 9B editor uses its tools: 100–180 Python runs and 15–20 passes in three of
four seeds. Its passes show the failure signal plainly (86–96 % of answers cut at the token limit), and with the
hinted prompt it repeats the hint's wording about loops in its comments (19 times in hinted seed 2). But it never
reads a cut-off generation to see the loop. Its repetition checks crash or run on GSM8K samples that never hit the
limit, and it settles on a mechanism it does not test: a "completion signal" in layer-23 hidden-state norms (hinted
s2), or lost attention to early prompt tokens (unhinted s1). Edits built on these stories (a layer-23 down_proj
perturbation; small V/Q-projection nudges) do not move the score. The fourth seed (unhinted s2) declared after
3 passes that its tools did not exist and stopped. Final answers are often off task (an unrelated chat reply in
Chinese; a refusal).

**27B, deepseek-flash.** [pending]

**What separates the frontier reference.** Opus 5.5 reads cut-off generations early, locates repetitive versus
productive token positions, and builds its edit from activation differences between them (§5.1). [Compare the call
at which each editor first inspects a looping generation: Opus vs open editors; `localised_at_call` from the labels.]

**Rule enforcement across scale.** The reviewer blocked [n] blocks across the open-weight episodes; the post-hoc
audit found no upheld violation in [n] of [n] finished runs. [9B unhinted s1: 2 suspicious calls, both protected-tensor
edit attempts the harness refused; adjudication pending.] Small editors rarely come close to the rule boundary: they
do too little computation with recorded activations for simulation to be a concern.
