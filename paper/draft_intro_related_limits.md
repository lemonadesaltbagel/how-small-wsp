# Draft: Abstract, §1–3, §7–8 (2026-11-10)

Working draft. [Brackets] = waiting for results or a citation. Numbers from `notes/episode_ledger.json` and
`notes/phase0_log.md`.

## Abstract (skeleton)
A frontier model, GPT-6 Astra, recently repaired a small language model by editing its weights directly: reading
Qwen3.5-0.8B's parameters in a Python sandbox with 50 forward passes of feedback, it changed about 0.017 % of them and
raised MATH-500 from 7.9 % to 48.1 % by removing reasoning loops [cite blog]. Is such weight-space programming a
frontier-only capability, or does it appear smoothly with scale? We repeat the experiment with open-weight editors of
0.8B, 9B and 27B parameters (Qwen3.5) and a sparse mixture-of-experts editor (deepseek-flash, 16B activated / 748B
total), each with a hinted and an unhinted prompt, under the same budget. We make the rule against simulating the
network in the editor's sandbox operational, enforce it with a pre-execution reviewer and a post-hoc audit, and audit
the original Astra log with the same auditor. [Result: smallest size that repairs, hinted / unhinted.] Opus 5.5 repairs
the model in all three strict episodes (40.6–55.4 %) without simulating the network, and [failure modes across
scale]. Code, trajectories, prompts and edits are public.

## 1 Introduction
- A language model that can read and rewrite another network's weights is a new kind of tool, and a step toward models
  that improve models. The first demonstrations use a frontier editor [cite blog; Hu et al. 2026]. If the ability is
  frontier-only, it is a capability to track at the frontier; if it appears smoothly with scale, small open models can
  do it too, which matters for who can use the tool and for how safety discussions treat self-modification.
- We ask one question: how small can the editor be and still repair Qwen3.5-0.8B's looping failure, with the budget
  and rules of the original experiment?
- The task is well suited to a scale curve: it has a clear failure (91 % of the unedited model's answers run into the
  32,768-token limit inside a verbatim loop), a clear success criterion (MATH-500 +20 points), a published frontier
  reference, and a scripted floor that shows what is possible once the location is known.
- Contributions: (1) the first scale curve of open-weight editors for this task, with a hinted and an unhinted prompt
  reported separately ("follow the hint" vs "diagnose"); (2) an operational definition of "reading weights without
  simulating the network", with a reviewer, an audit and adjudication rules, and the finding that a frontier editor
  repairs the model as well without simulation; an audit of the original Astra log with the same auditor; (3) failure
  modes across scale (not localised / localised but edit failed / budget exhausted), which stand even where the curve
  is zero.

## 2 Background and related work
- **Weight-space programming.** *Sparks of Weight Space Programming* [cite] lets GPT-6 Astra read and write network
  weights in eight experiments; experiment 8 is ours. Their logs are public, which lets us audit the run. Their own
  follow-up on Gemma 3 270M, a model that rarely loops (8.5 % of answers at the limit), found no edit that moved
  MATH-500 by more than about a point: the gains come from removing a failure, not from adding mathematical skill.
- **Closest work.** Hu et al. (2026, arXiv 2611.09382) vary the *target* (Qwen3.5-0.8B and 9B, seven benchmark
  families) with a single editor (GPT-6 Astra) and report 8.35 → 37.80 % on MATH-500 for the 0.8B model; they do not
  vary the editor or audit tool calls. We vary the editor and audit. Community reproductions: experiments 1–6 with
  Opus [cite EphraiemSarabamoun]; a scripted solver that re-derives the loop direction given the paper's localisation
  (KidIkaros) [cite]; experiment 1 (which game a policy network plays) with a local Qwen3.8-27B (Q4_K_M) reading the
  weights in its prompt, without tools or code, 3 of 4 games identified (Willison 2026,
  simonwillison.net/2026/Nov/3/qwen-can-interpret-model-weights/) — interpretation only, no editing; edited Qwen3.8-27B checkpoints [ReBatch; Teylersf], where the 27B model is the target, not the
  editor.
- **Model editing and steering.** Locate-and-edit methods (ROME, MEMIT) and direction-based interventions
  (activation addition, directional ablation) change behaviour with small, targeted weight or activation changes;
  several of the editors' fixes rediscover such rank-1, direction-based edits. Critics of the original post make the
  same point [R&D World coverage]. The difference is who finds the edit: here an agent finds it from the weights and
  a fixed feedback budget, without a recipe.
- **Agentic interpretability.** [HyVE and related agent-driven interpretability work; cite]
- **Why Qwen3.5 for the ladder.** Qwen3.5 has 0.8B, 9B and 27B models of one family; the newer Qwen3.8 starts at 27B.

## 7 Limitations
- One target model and one failure (reasoning loops), which the Gemma follow-up suggests is the regime where weight
  editing helps; results need not transfer to adding capability.
- Few seeds (2, a third only for split cells): the curve shows where repair becomes possible, not precise rates.
- The size ladder mixes families: three Qwen3.5 sizes and one sparse MoE from another family (drawn as an interval,
  not joined to the curve).
- The reviewer is itself weak feedback: a blocked block comes back with a reason, which can teach the editor the rule.
  We report blocks per run.
- Rules differ between the frontier reference (rules v1 plus written adjudication of boundary cases) and the open
  editors (strict2, which states those boundaries up front).
- The unhinted prompt is our construction; the authors' unhinted prompt was not published.
- Scoring uses one MATH-500 sample per episode; repairs are re-scored with four. A second, independent grader agrees
  with ours on 96.6–99.2 % of answers and never counts an answer we mark wrong as right.
- Astra's 48.1 % is not directly comparable with our strict results: its log contains three linear attributions
  through down_proj on recorded activations, which our rule (b) forbids, and two of them selected the neurons of its
  submitted edit.

## 8 Future work
Self-editing (an editor that edits its own weights; our 0.8B editor edits a copy of a same-size model, a reference
point only); installing capability vs growing it (fine-tuning baselines, a benchmark of known faults); closed-loop
dynamics over repeated edits; injected failures other than looping; a zero-feedback condition.

## Checks already done (for §4 / appendix)
- Edits: every scored network equals the base plus its declared edit set exactly (all 320 tensors, 6 runs).
- Grading: independent re-extraction + MATH normalisation agrees 96.6–99.2 %, one-sided.
- Analysis plan frozen and pushed before the first main-matrix result (`notes/analysis_plan.md`, commit 5de34ad).
