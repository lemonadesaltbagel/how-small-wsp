# Draft: Methods and §5.1 (2026-11-10)

Working draft for *How Small Can a Weight-Space Programmer Be?* Numbers come from `notes/episode_ledger.json`,
`notes/phase0_log.md` and the run directories; [brackets] mark what is still open. Not for circulation yet.

## 4 Methods

### 4.1 Task and harness
The task is the authors' experiment 8: make Qwen3.5-0.8B (text model, 24 layers, width 1,024; 6 gated full-attention
layers and 18 Gated DeltaNet layers) better at competition math by editing its weights. In thinking mode the unedited
model runs into verbatim reasoning loops: 91 % of its MATH-500 answers hit the 32,768-token limit, and it scores
9.0 % (authors: 7.9 %).

We run every episode in OpenWiSH, an open re-implementation of the authors' harness, with the task files derived from
the authors' published p50c run. The editor works in a sandboxed Python kernel (no network) that holds the weights as
arrays. It has three tools: Python over the weights (300 runs, 15 min each), a forward-pass tool on the target model
(50 passes; generation with activations, teacher-forced log-probs, attention), and `save_edit`, which must carry a
written reason. The embedding, the output layer and the final norm cannot be edited, and an edit may not rescale a
whole tensor. An episode ends with a final answer after all 50 passes, or after 600 turns or the token caps
(150M editor tokens, 6M output tokens); long conversations are handed off to a fresh context with the editor's own
notes every 40 turns. The score is MATH-500 (all 500 problems, one sample, the authors' generation settings) of the
submitted weights, computed outside the editor's view.

We extended OpenWiSH with: an OpenAI-compatible backend for open-weight editors served by vLLM; per-episode token
caps; the authors' mid-run notices given up front as appendix notes (1)–(2); the "explain every step" rule (3) and
mandatory edit reasons (4); generation passes split across several GPUs; and a hold-out evaluation (MMLU-Pro 1k,
GPQA, IFEval). Before any episode we checked conformance: rebuilding the authors' three published edits from their
diffs reproduces the published checkpoints bit for bit (sha256), and re-scoring them gives base 9.0 / p10c 14.2 /
p25c 13.4 / p50c 50.4 % against the authors' 7.9 / 12.3 / 14.1 / 48.1 % (their numbers are 4 samples, ours 1;
all within about 1.3 standard errors). Table 1.

### 4.2 What "reading weights" means: rule (b)
The original rules forbid (a) training and (b) "running the model other than through the forward-pass tool: do not
implement or simulate the network's computation on any input in the kernel", and allow "any analysis of the weights
and of forward-pass results". The two sentences admit two readings. In our first two Opus 5.5 episodes the editor
applied layers' weights (RMSNorm, gate sigmoids, down_proj, the output layer) to activations returned by forward
passes and used the result to score candidate edits; a post-hoc audit found 26 and 11 such calls.

We therefore made rule (b) explicit (appendix note 5): in the kernel, recorded activations of actual tokens may not be
multiplied by the network's weight matrices or their rows and columns, passed through its norms, gates, activation
functions, attention or output layer, or used to reconstruct what a layer, or an edited layer, would output; candidate
edits may not be ranked by such computation. Allowed are statistics of recorded activations (means, group
differences, PCA, correlations, projections onto directions computed from other activations), any analysis of the
weights themselves, including products of weight matrices with such directions, and edits built from this analysis.
Auditing the strict runs raised two boundary questions, decided before the open-weight episodes started and then
written into note 6: closed-form statistics of activations (mean differences, LDA, least-squares read-outs) are
allowed, also when installed as weights, whereas iterative fitting is training; designing an edit on made-up scalar
inputs and extrapolating measured forward-pass outputs are allowed; predicting in the kernel what a candidate edit
would do at recorded token positions is not.

Enforcement has two layers. A reviewer model reads every Python block before it runs and sends a violating block
back unrun and uncharged, with its reason. A different model audits every Python run after the episode, and each
flagged call is adjudicated in writing; a run with an upheld violation is invalid. The reviewer is deepseek-flash
(v1 for the Opus runs; v2, which also knows note 6 and sees earlier variable definitions, for the open-weight
editors) and Claude Sonnet 5.5 when deepseek-flash is itself the editor; the auditor is Codex (medium effort).
Calibration against adjudicated audits: v1 blocked 36 of 37 violations and 1 of 37 allowed calls (first-round runs);
v2 blocked 3 of 3 violations and 0 of 21 allowed calls, Sonnet 5.5 2 of 3 and 0 of 21 (small samples; the one miss is
a boundary case blocked at run time and never audited). We report reviewer blocks for every run (Table 4). The
reviewer is itself a weak form of feedback (§7).

### 4.3 Editors, prompts and budget
Editors: Qwen3.5-0.8B, 9B and 27B (bf16, vLLM, thinking on; the 0.8B edits a copy of a model of its own size) and
deepseek-flash (DeepSeek-V4.1-Flash-1017 through its API), with Opus 5.5 as the frontier reference. Every editor has
the same budget: 50 forward passes, counted by number, not size (a pass may hold up to 512 answers, appendix note 1),
300 Python runs and 600 turns. Fixing the budget is what makes editors comparable; the time a pass takes does not
count.

Prompts: the authors' v3 prompt ("hinted": it names repetition as an example failure three times) and a self-built
unhinted prompt that removes those three phrases and changes nothing else. The authors' own unhinted prompt was not
published.

### 4.4 Outcomes
The analysis plan was committed before any main-matrix result (`notes/analysis_plan.md`). A **repair** is MATH-500
at least 20 points above the base (≥ 29.0 %) with no hold-out regression (no hold-out score more than 2 standard errors
and more than 5 points below the base). Hold-out runs only for repairs, and every repair is re-scored with 3 more
samples. Each episode gets one failure label: success, not localised, localised but edit failed, or budget exhausted
(definitions in the plan). Secondary measures: GSM8K, the share of answers cut at the token limit, edit size (changed
bf16 entries over text-model parameters), passes and Python runs used, editor tokens.

## 5.1 Reproduction and reference lines

**Base model.** MATH-500 9.0 ± 1.3 % (authors 7.9 %), 91.2 % of answers cut at 32,768 tokens; GSM8K 11.2 ± 1.4 %
on the first 500 of 1,319 problems [full set × 2 samples pending] (authors 13.8 %); hold-out IFEval 20.1 %
(authors 21.0 %), MMLU-Pro 1k 39.7 % (40.3 %), GPQA 24.4 % (22.7 %).

**Opus 5.5 under the strict rules** (hinted prompt, 3 seeds, all valid after audit and adjudication):

| seed | MATH-500 | MMLU-Pro 1k | GPQA | IFEval | turns | reviewer blocks |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 44.8 % | 40.4 | 27.8 | 57.7 | 356 | 2 |
| 2 | 55.4 % | 44.8 | 30.8 | 60.4 | 343 | 0 |
| 3 | 40.6 % | 44.7 | 30.1 | 60.8 | 357 | 2 |
| base | 9.0 % | 39.7 | 24.4 | 20.1 | — | — |

All three seeds repair the model (+31.6 to +46.4 points) with no hold-out regression; IFEval rises by about 40 points
because far fewer answers are cut off. The seeds found different edits: [one sentence each; s3: an LDA "loop-state"
direction wired into the output gates of 18 full-attention heads in layers 11–23, so that the heads close only inside
loops].

**Simulation does not explain the result.** The first-round Opus episodes, which simulated parts of the network in
the kernel, scored 50.8 % and 35.0 % (mean 42.9 %); the strict episodes, which could not, scored a mean of 46.9 %.
The rounds differ in more than the rule (mid-run notices and kernel restarts in the first round, 2 vs 3 seeds), so we
read this only as: reading the weights plus forward-pass feedback suffices; simulation is not necessary.

**Astra.** Audited with the same auditor and the same reading of rule (b), the authors' p50c log has 3 violations
among 80 Python runs. All three are linear attributions through down_proj on recorded activations: the mean
MLP-activation difference between repetitive and productive tokens times the down_proj columns projected on the
repetition direction, i.e. each neuron's contribution along that direction. Two of them rank the neurons that make up
the submitted edit (64 per layer in layers 15 and 19). No nonlinearity was simulated. Under the authors' rules this
was plausibly allowed; under ours, Astra's 48.1 % belongs with the simulation-allowed results and is not compared
directly with the strict ones.

**Scripted floor.** KidIkaros's scripted edit scores 33.6 % (1 sample). It re-derives the loop direction from
activations but takes the layers, the number of neurons and the strength from the authors' edit, so it is a floor
given the localisation, not an LLM-free diagnosis.
