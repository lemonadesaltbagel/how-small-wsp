# Phase 0 lab notes

Chronological; facts with their source. Numbers here feed the reproduction report and tracker proposals.

## 2026-11-06

### Setup
- OpenWiSH vendored at `harness/` from upstream 4c369dc (first commit unchanged).
- Compute: harness + sandbox kernel on the WSL node (16 threads, 23 GB RAM, RTX 5070 Ti 16 GB). No sudo: bubblewrap 0.12 from `apt download` into ~/.local (unprivileged userns works on WSL). Rust 1.99 via rustup. vLLM 0.33.0 / torch 2.14 cu130 / transformers 5.20.2 in `assets/venvs/qwen`; FlashInfer needs `vllm download-kernels` (no nvcc).
- **bubblewrap does not work inside RunPod containers** ("No permissions to create new namespace"; tested on an RTX 4090 secure pod). So the harness always runs on WSL; pods only serve models (editor via vLLM OpenAI API + SSH tunnel; target forward passes via OpenWiSH's remote-GPU mode).
- Base checkpoint: Qwen/Qwen3.5-0.8B @ 2fc0636; text checkpoint sha256 68194c7b…; MATH-500 file 35dc4108…; GSM8K test 3730d312…. Identical on the pod.
- **All three published edits rebuild on our base with both sha256 hashes matching** (base tensor and edited tensor): p10c 30 tensors (low-rank, ~1.8–3.2M entries changed per tensor), p25c 1 tensor (262,144 entries), p50c 2 tensors (65,260 + 64,928 entries; the blog's 131,072 counts columns × rows, some entries unchanged after bf16 rounding).

### Findings about the source material
- **The authors' unhinted v4 prompt is not published.** The logs repo keeps only the v3 runs (p10c/p25c/p50c). In `eval/RESULTS_excerpt.md` the v4 25-pass row (14.1 / 25.1) is identical to the v3 p25c row, which looks like a copy error; per `code/profiles.py` the v4 runs were also served with v2 limits (no "both", 128 answers). → Our unhinted arm uses our own construction: v3 with the three repetition phrases removed (`scripts/make_exp8_tasks.py`). Report it as ours, not as "v4".
- The authors' runs received notices 001 (pass size) and 002 (tool facts) mid-run and an "EXPLAIN EVERY STEP" rule from turn 16–52. OpenWiSH delivers none of these. Our tasks give notices 001/002 and the explanation rule from turn 1 (appendix), and require a `reason` on every `save_edit`.
- The v3 prompt offers activations in generation passes; OpenWiSH rejected them. Implemented in `qwen_worker.py` to the prompt's spec (shapes match the published pass 1).
- The v3 prompt says one Python execution may use 96 GB; our WSL kernel is capped at 20 GB (WSL has 23 GB). OpenWiSH's own exp8 tasks use 24 GB. Note as a deviation.
- The prompt's "at most 6 GB per pass" for teacher-forced outputs was not enforced; now checked before allocation.

### Editor-side usage of the published runs (sum over `turns/*/run.json`)
| run | turns | input tok | of which cached | output tok | reasoning tok | editor wall h |
|---|---|---|---|---|---|---|
| p10c | 74 | 26.9M | 22.5M | 1.11M | 0.36M | 0.8 |
| p25c | 86 | 38.4M | 34.4M | 1.49M | 0.46M | 1.1 |
| p50c | 158 | 86.8M | 80.3M | 2.91M | 0.83M | 1.8 |

p50c wall-clock was 20.3 h; the editor's own time was 1.8 h. Forward passes summed to ~7.0 h (`requests/fwd_*.done.json` field `s`), including 17 failed generation passes (vLLM engine start failures on the authors' side). → The target side, not the editor, sets the episode length. The brief's "1–2 GPU-h per episode" target-side estimate is too low.

### Harness additions (commits a295879 … )
- `[finish] max_editor_output_tokens / max_editor_total_tokens / max_editor_cost_usd` (default in our tasks: 150M total, 6M output ≈ 1.7× p50c).
- `[rationale] min_words` (15 in our tasks): tool blocks without an explanation are sent back unrun, at most twice in a row.
- `openwish run --max-conversation-tokens` (handoff threshold for small-context open editors).
- Remote worker `gpu_util` (a pod GPU shared by the editor server and the target worker).
- Hold-out scorer (MMLU-Pro 1k / GPQA Diamond / IFEval, authors' think35 protocol); CPU re-grade of the authors' published first samples agrees 400/400 (MMLU-Pro) and 399/400 (IFEval; the miss is langdetect randomness). GPQA option shuffle seed unpublished → our own deterministic shuffle. See `harness/docs/holdout.md`.

### Open-weight backend smoke test
- Qwen3.5-27B (bf16) served by vLLM 0.33 on an RTX PRO 6000 (96 GB) pod, `--reasoning-parser qwen3`, max_model_len 196,608, 70 % GPU memory (KV cache 210k tokens); the target worker shares the GPU at 25 %. Reasoning comes back separately from the answer, so code inside the thinking is never parsed as a tool call.

### Scripted-solver floor (KidIkaros/wsp-reproduction)
- Their "independent diagnosis" edit is not fully LLM-free information-wise: it uses the paper's layers (15, 19), 64 neurons per layer and strength 6 (recovered from the released p50c edit) and only re-derives the loop direction (mean MLP-output difference looping vs finished transcripts; cos 0.35/0.45 with Astra's). Evaluated by them only on 18 problems.
- Plan: score their released checkpoint `KidIkaros/qwen3.5-0.8b-loop-fix` on MATH-500 with our scorer (one eval). Report as "scripted floor, given the paper's localisation".
- Target-pass throughput, smoke pass 1 (RTX PRO 6000, target worker at 25 % GPU memory beside the idle 27B editor): 255 generations, mean 27,891 tokens, 82.7 % cut at 32,768 → ~7.1M tokens in ~27 min ≈ 4.4k tok/s. A full 512-answer pass would take ~55 min at this share (authors: "up to about half an hour").
- Smoke episode result (Qwen3.5-27B, 2-pass smoke task): 16 turns, 10 Python runs, 1 generation pass, no edit; 219k input / 14k output tokens; one rationale send-back. **It ended early on an empty reply**: copying the appendix's literal `<|im_end|>` made the vLLM-served editor emit its own end-of-turn token (the same truncation caused its earlier "unterminated string literal" errors). Fixed (commit "exp8sl: editors never type special tokens…"): EVAL_PREFIX/EVAL_SUFFIX in the kernel, `texts_file` for teacher-forced passes, no special tokens in the appendix, empty replies are uncharged protocol errors. Expect the same hazard for any Qwen-family editor writing about Qwen's template; Opus/Astra are unaffected (APIs treat these strings as text).
- Remote scoring of the unedited smoke run (first 32 MATH-500 problems, pod GPU): 9.4 %, 90.6 % cut — in line with the published base. (Upstream bug fixed: remote scoring of an unedited run failed on an empty staging folder.)

### Capacity and the ≥70B editor (2026-11-06, evening)
- **Harness capacity = 2 parallel episodes on WSL.** Each `openwish run` process holds ~6.0 GB RSS (the edit gate keeps base + current network in float32, 2 × 3.2 GB) plus a ~0.4 GB kernel; with two Opus runs and the conformance job, WSL had 6 GB available of 23 GB. The user cannot raise WSL memory (Windows stability). RunPod containers cannot host the harness: no unprivileged userns, and CapEff lacks CAP_SYS_ADMIN / CAP_NET_ADMIN (`unshare -n` and `unshare -Ur` both fail), so no network-isolated sandbox is possible there. → Queue episodes two at a time. For Q1 (57 episodes) the gate must stop holding two dense copies (keep edited tensors only; memory-map the base, shared through the page cache across runs).
- **≥70B hosted editor: `deepseek-flash` (DeepSeek-V4.1-Flash-1017, open weights, 763B, MIT, 1M context).** On the benchmarks both model cards report it beats V4-Pro-0813: Agents' Last Exam 35.4 vs 25.7, NL2Repo 68.2 vs 61.5 (the other rows use different benchmark versions). User note: flash post-trains better. API check: `reasoning_effort: "max"` accepted, reasoning returned separately, cache hits in `prompt_tokens_details.cached_tokens` (the field OpenWiSH reads). The API does not report cost → the cost cap cannot fire; token caps do. Balance ¥65.20 at the start; per-episode price to be measured on a pilot (balance before/after).
- Gate memory fix (commit "tensor values shared copy-on-write"): peak `openwish run` RSS on exp8 6.0 → 4.0 GB (scripted episode, one edit). The two running Opus episodes still use the old binary (6 GB each) until they finish or resume.

### Conformance (MATH-500, 500 problems, 1 sample, 32,768 tokens; WSL RTX 5070 Ti, QWEN_GPU_UTIL 0.75)
| config | ours | OpenWiSH replay | published (4 samples) | our cut share | wall |
|---|---|---|---|---|---|
| base | 9.0 ± 1.3 % | 10.6 ± 1.4 % | 7.9 % | 91.2 % (mean 30,393 tok) | 107 min |
| p50c | 50.4 ± 2.2 % | 51.8 ± 2.2 % | 48.1 % | 30.4 % (mean 21,506 tok) | 68 min |
| p10c | 14.2 ± 1.6 % | 13.6 ± 1.5 % | 12.3 % | 86.4 % (mean 29,828 tok) | 103 min |
| p25c | 13.4 ± 1.5 % | 14.2 ± 1.6 % | 14.1 % | 88.0 % (mean 30,082 tok) | 105 min |

**Conformance passes: all four published configs reproduce within noise** (each within ~1.3 SE of the published 4-sample mean). Total 383 min on the 5070 Ti.
- Throughput: 15.2M generated tokens in 107 min ≈ 2.4k tok/s on the 5070 Ti → one full 512-answer pass of the unedited model ≈ 1.8 h here.

### DeepSeek-flash seed 1 (started 2026-11-07 03:01; target passes on an RTX 6000 Ada pod, $0.99/h — no 5090/4090/PRO 6000/L40S/A100 in secure stock)
- Pilot cost, first 9 turns: ¥0.09 (balance 65.20 → 65.11; ¥0.01 granularity) for 17k uncached input + 74k cached input + 16k output tokens. Rough rates consistent with this (not published on the docs pages we could read): miss ≈ ¥2/M, hit ≈ ¥0.2/M, output ≈ ¥3/M → a p50c-sized episode (6.5M uncached, 80M cached, 2.9M output) ≈ ¥38 ≈ $5. Re-measure at ~50 turns.

### Incident: Opus target pods out of disk (found by the user, 2026-11-07 ~03:40)
- Both 60 GB pods reached 94 % / 98 %: synced pass outputs never deleted on the pod (~10–11 GB of teacher-forced arrays), one 1.6 GB edited checkpoint per distinct edit (4–6 kept), uv cache (~19 GB with vLLM wheels). vLLM then failed to start ("Engine core initialization failed"): 6 uncharged failed passes across the two seeds (s1 turns 112–116, s2 turns 109–113) plus one SSH timeout; the editors lost those turns but no budget.
- Fixed: verified every pass folder was already on WSL (s1 10, s2 14), cleaned the pods (→ 43 % / 40 %); `qwen_pass.py` now deletes the remote pass folder after syncing it back; `qwen_worker.py` keeps only the 2 newest edited checkpoints; the progress monitor alerts on any increase of `failed_requests` (failed passes are uncharged, so they never appear in the status line). New pods: use ≥100 GB disk.

### Incident: WSL memory exhausted with three episodes (2026-11-07 ~05:00)
- The editors' kernels grow as they load tensors: 4.3 GB (DeepSeek, after 36 turns), 3.7 GB and 2.8 GB (Opus) on top of three ~3 GB `openwish` processes. A 2.7 GB diagnostic load of mine (an attention-layout check) pushed `available` to 0.3 GB. I killed my test and paused DeepSeek seed 1 at turn 36 (state saved, resumable; its pod deleted, 129 min, $2.13); `available` back to 9.9 GB, both Opus runs unharmed.
- Lesson: **WSL holds two episodes**, not three, once kernels have grown; never run ad-hoc model loads on WSL while two episodes run.
- DeepSeek pilot, 36 turns: ¥0.51 (65.20 → 64.69) for 57k uncached + 979k cached input and 109k output tokens.
- Found by DeepSeek: `attn_layers` in teacher-forced passes crashes (`IndexError: tuple index out of range` at `r.attentions[l]`). The authors' p50c passes 3 and 6 failed with the same error, and their notice 002 says attention "works now", so they fixed it mid-run; OpenWiSH still has the bug. To fix before resuming.
- Fixed `attn_layers` (commit "exp8 worker: attention probabilities…"): transformers records attentions only from `Qwen3_5Attention` modules, so the tuple has 6 entries (layers 3, 7, 11, 15, 19, 23 per config `layer_types`). Index by position in FULL_ATTENTION. Checked on the WSL GPU through the worker: layers 3/15/23 return [8, T, T], rows sum to 1, causal, layers differ.

### Correction notice to both Opus seeds (2026-11-07; user approved the restart)
- Before the attention fix, `attn_layers: [3]` silently returned layer 15's probabilities labelled layer 3 (the 6-entry tuple's index 3 is layer 15); layers 7–23 crashed. Both Opus seeds requested `[3]` early: s1 `requests/forward_2.json` (11-06 23:21), s2 `forward_2.json` (23:11) and `forward_3.json` (23:18). DeepSeek asked for all six layers and only got the crash.
- New harness feature: `<run>/notices/*.txt` delivered once, appended to the next message as "NOTE FROM THE EXPERIMENTERS: …" (the published runs got notices 001/002 the same way). Both seeds were stopped between forward passes (`scripts/restart_episode.sh`), resumed on the new binary, and received `001_attention_layers.txt`: s1 at turn 171 (05:33), s2 at turn 177 (06:07). Their kernels restarted (variables lost; edits, /work and notes kept). Report this in Methods as a mid-run correction for seeds 1–2; later runs have the fix from turn 1.
- After the restart both runs use the copy-on-write binary: `available` 12.2 GB with two episodes (was ~6 GB).

### Opus seed 1 finds a loop fix (2026-11-07, turn ~196, pass 16/50)
- Paired practice pass (n=256): original 11.7 % → edited 53.5 % (+41.8 ± 3.1), cut share 89 % → 36 %. Earlier passes: +6.6 (t080), +3.5 (t118), +26.6 (t169).
- Kept edits: rank-1/rank-2 edits to the **output-gate rows of `self_attn.q_proj`** in the full-attention layers 15 (heads 4–7), 19 (heads 5–7) and 23 (heads 0, 1, 4, 5, 7), removing the gates' input along a "loop-state" direction (mean normalized layer-15 input, early-loop minus normal tokens). A different mechanism from Astra's p50c (direction removed from 64 MLP output columns in layers 15 and 19). Network fraction changed 0.42 % (> the brief's 0.1 % target; not a gate in the harness).
- Practice-set paired numbers only; the episode's MATH-500 score comes from the scorer at the end.

### Pod idle time and savings (2026-11-07 ~08:30; `scripts/pod_utilization.py`)
- Opus target pods, 10.9 h each: forward passes (incl. vLLM start and copy-back) occupied 7.8 h (s1, 72 %) and 8.2 h (s2, 75 %). Generation passes 45–48 min mean, teacher-forced 4–8 min. Failed passes (the disk incident) cost 0.9 h (s1) and 1.2 h (s2). DeepSeek pod: 1.8 h busy of 2.15 h (83 %).
- Gaps between passes: median 3.0 / 2.5 min, longest 17 min, none over 30 min (34 / 32 gaps). → Creating pods on demand (≥ 8–10 min setup + scarce secure stock) or stop/start (≥ 3–5 min boot + engine start) would add more waiting than the gaps they save. Not feasible.
- Copy-back of large teacher-forced outputs runs at ~3 MB/s (home link): a 2 GB output took ~11 min. Useful (non-padding) fraction 45–94 %; rsync -z enabled (small gain, no risk).
- Real savings left: (1) run one of the two parallel episodes on the free local 5070 Ti (≈2× slower passes; only when memory headroom ≥ 4 GB holds); (2) two episodes sharing one pod GPU (combined demand ~150 % of a GPU → each episode ~1.5× slower, pod cost per episode about halved); (3) the disk fix already removed ~1 h of failed passes per pod.

### Opus seed 2 breakthrough (2026-11-07 ~10:00, turn 279, pass ~26/50)
- Paired practice pass (n=256): 12.9 % → 42.2 % (+29.3 ± 3.2), cut 88 % → 51 % (earlier best +8.2).
- Its "V2": E1 = W − 3·u uᵀ W on the 9 output writers of layers 8–14 (u = repetition direction at the layer-15 input) plus projecting out a "doubt" direction (layer-19 input, "Wait"-line starts minus others). Being tested on top ("T1"): project out a line-level "restate rather than explore" direction from writers in layers 11–18. Dense rank-1 projections over 19 tensors → **~7 % of all weights changed** (vs s1's 0.42 %, Astra p50c's 0.016 %). Closest in style to Astra's p10c.
- So far three different mechanisms for the same loop pathology: Astra p50c (MLP columns, L15/L19), Opus s1 (attention output gates of copy heads, L15/19/23), Opus s2 (dense direction projections, L8–18).

### Integrated OpenWiSH for reuse (2026-11-07)
- `git clone --branch v0.1.0-integrated.1 https://github.com/lemonadesaltbagel/OpenWiSH.git` (tag 14c53be on the user's public fork; default branch untouched; 22 commits on upstream 4c369dc, 31 files, +4,126). Holds upstream PRs #1–#9 merged, experimenter notices, rsync -z, the exp8 hold-out scorer, `docs/fork.md`, `scripts/setup_nosudo.sh`. All Rust and Python tests pass on WSL (GPU hidden).
- **The Phase 0 runs use the vendored `harness/`, not this tag.** Behavioural differences (all edge cases): `--max-conversation-tokens` with `--resume` is an error there (ignored here); attention index from `layer_types` (same result on Qwen3.5-0.8B); checkpoint pruning keeps 2 including the current one (ours may keep 3); stricter `activations` validation; `texts_file` also rejects `.`/`//` and opens by fd; the 6 GB estimate counts token_logprobs only when requested; `[rationale] tools` must name enabled tools. Empty replies: both end on the 4th in a row (`> 3`) — our log note "after 3" was imprecise.
- Plan: keep the vendored harness through Phase 0; decide on switching for Q1.
