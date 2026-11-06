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
