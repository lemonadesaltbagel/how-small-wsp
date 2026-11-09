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
- GSM8K, unedited base (local 5070 Ti, 08:36–10:21): **11.2 ± 1.4 %**, cut 89.6 %, mean 29,838 tokens — on the **first 500 of 1,319** test problems (exp8_conformance's default `--limit 500`), 1 sample; published 13.8 % (all 1,319, 2 samples). Within ~1.8 SE; the subset is the first 500 in file order. The full set would take ~4.6 h more on the 5070 Ti; not needed for the Phase 0 gate (MATH-500).
- **KidIkaros scripted floor, MATH-500 (1 sample): 33.6 ± 2.1 %, cut 60.6 %** (mean 25,345 tokens; local 5070 Ti 10:28–11:52). Edit: L15 + L19 `mlp.down_proj`, 64,535 + 63,486 entries (0.017 %) — the paper's localisation and strength with a script-derived direction. So "given the paper's localisation" a scripted edit already reaches 33.6 % (base 9.0 %, Astra p50c 50.4 % in our replay). Report it as an oracle-localisation floor, not as an LLM-free diagnosis floor.
- Hold-out, unedited base, IFEval (541 × 2 samples, local 5070 Ti 11:58–15:08): **20.1 ± 1.4 %** (published 21.0 %), cut share 74.9 %. The hold-out scorer reproduces the authors' number on GPU too. MMLU-Pro 1k and GPQA queued next. (Note: holdout_eval writes its summary to the log, not to summary.json, so the queue's skip-if-done check does not fire on reruns.)
- Opus s1, turn 318 (pass 35/50): paired 8.6 % → 61.3 % (+52.7 ± 3.2), cut 93 % → 21 %. New on top of the gate edits (GE1 L15, GE2 L19/L23): "WC2", in layer-19 head 0's `o_proj` input columns reflect the line-initial "Wait" value channel (C ← C − 2(Cd)dᵀ). Chosen by a dose series α = 0/1/2/3 → 56.3/58.8/60.5/58.6 % (512 answers each), then replicated on 256 never-used problems (57.4 vs 53.3 % without it). Network fraction 0.45 %. A clean hypothesis → dose → replicate trace: a good case for the rationale-verifiability protocol.
- Opus s2, turn 308 (pass ~34): paired 9.8 % → 38.3 % (+28.5 ± 3.0) on fresh problems, replicating its +29.3.

### No-gradient evidence for the Opus edits (2026-11-07 ~17:00; `scripts/edit_structure.py`)
- Hard layer (kernel.py): `torch.set_grad_enabled(False)`; backward, autograd.grad, torch.optim and requires_grad_(True) raise; the kernel has no GPU and no runnable model (CPU arrays only); network runs only through the counted forward probe.
- Static checks over every Python run (flags in tool_calls.jsonl): s1 285 runs (25 call save_edit): scalar_relu 7, forward_pass 2; s2 264 runs (14 save_edit): scalar_relu 10; DeepSeek 25 runs: scalar_relu 2. **No autograd / optimizer / scipy_optimize / manual_gradient_step / fit_into_weights flag anywhere.** The two s1 forward_pass flags (turns 178, 206) compute one layer's attention gates `sigmoid(x @ Wq_gate)` from probe-returned hidden states for analysis; neither run calls save_edit. Grey zone under OpenWiSH's "no network runs outside the probes" rule → put to the auditor.
- Structure of every kept edit (current vs base): s1 4 tensors, numerical rank 1 each (1 % threshold); s2 19 tensors, rank 1–2 each, except L15 o_proj rank 257 = head 7's 256 columns zeroed exactly + a rank-1 projection (its ledger reason "V16 … zero the o_proj columns of head L15H7"). Gradient-descent or fitted edits would be dense and near full rank. The ledger keeps the code of every save_edit, so each edit can be re-derived.
- Gap: hand-written numpy gradient descent is not physically impossible; it is forbidden by the prompt, caught only by the static patterns and the auditor. **The auditor (`openwish audit`) has not been run yet** → run it on every Phase 0 run after it ends, before reporting scores.
- Implication for the edit-size cap: "fraction of entries changed" counts a rank-1 projection of a whole matrix as 100 % of that matrix (s2 → 7 % of the network) although it is one direction. Rank × (rows + cols) parameters, or relative Frobenius norm, measure edit size better.
- Hold-out, unedited base, MMLU-Pro 1k (1 sample, 15:08–16:38): **39.7 ± 1.5 %** (published 40.3 %), cut 31.4 %. GPQA Diamond (4 samples) next.

### Incident: orphaned worker after an SSH drop (2026-11-07 ~20:35, Opus s1)
- The WSL→pod SSH session of a generation pass timed out after 27 min ("server not responding", ServerAliveCountMax default 3 × 30 s). The remote worker kept running (no pty → no SIGHUP) and held 28.9 GB of the 5090. Opus's retry at turn 355 then failed at vLLM start ("Free memory", 59 s; uncharged). I killed the orphan (pids 26264/26448/26652); the next retry started on a free GPU.
- Fixed in `qwen_pass.py` (live from the next pass): ServerAliveCountMax=20 (10 min tolerance); before each remote worker start, end leftover workers with anchored patterns (`^<python> <root>/qwen_worker.py`, `^… holdout_eval.py`) and only orphaned `VLLM::EngineCore` processes (parent 1), so an editor server on the same pod is never touched. Dry-run on the live pod: the patterns match only the worker processes, not the issuing shell.
- Hold-out, unedited base, GPQA Diamond (198 × 4): **24.4 ± 2.1 %** (published 22.7 %; SE clustered by problem), cut 67.9 %. All three hold-out baselines reproduce: IFEval 20.1 / 21.0, MMLU-Pro 39.7 / 40.3, GPQA 24.4 / 22.7 (ours / published).
- **Bug found and fixed:** GPQA grading hung for 2.5 h after generation finished (18:56). The official MMLU-Pro fallback regexes (`.*[aA]nswer:\s*([A-J])` and `\b[A-J]\b(?!.*\b[A-J]\b)` with DOTALL) are quadratic; GPQA answers average 25k tokens (68 % cut at the limit). Replaced by a linear-time equivalent (lookahead for overlapping "answer:" candidates; last standalone letter); `tests/test_mc_extract_equivalence.py` checks equality with the official regexes on 20,000 random edge-case texts and the 400 published MMLU-Pro/GPQA texts. Regrading the saved generations (`--reuse-raw`) took 1.7 s. The post-episode hold-out runs on the pods would have hung the same way (billed pod time).

### Incident: default node re-pointed to the new laptop (2026-11-07 21:45)
- Adding the user's new compute node (laptop merchrevo-15-pro: Ryzen 7 8845H, RTX 4060 Laptop 8 GB, 23 GB WSL) rewrote the cnode registry entry `wsl` to the laptop, so every `rx` from 21:45 went there while the experiments run on the desktop. Mutagen sessions were still on their old connections (the project never reached the laptop), and the post-episode pipelines had not polled in that window. Fixed 21:48 by editing `~/.config/compute-node/{nodes,known_hosts}` (backup in `backup-20261107-214827/`): `wsl` = desktop again (as the user's CLAUDE.md says), the laptop registered as `laptop`; `cnode init` regenerated ssh_config. Verified: `rx -n wsl hostname` = DESKTOP-DVJRJVF with tmux opus1/opus2/localq; `rx -n laptop` reachable.
- Also found: the local GSM8K vLLM job (QWEN_GPU_UTIL 0.75) cost ~13 GB of **Windows commit** on the desktop: commit free 22.0 → 8.9 GB of 86.1 GB with two episodes running. Stopped it (commit free back to 21.8 GB). Rule: check `rx nodes` Windows commit headroom before any local GPU job on the desktop; with two episodes running, local vLLM jobs go to the laptop.

### Paused: RunPod balance low (2026-11-07 23:02, user request)
- Stopped all runs between turns (Ctrl-C; states saved), deleted all three pods. Saved states: Opus s1 turn 364, 48/50 passes (a pending forward request re-runs on resume, uncharged); s2 turn 350, 48/50; s3 (laptop, `harness/runs_laptop/phase0/opus55-hinted-p50-s3`) turn 11, 0/50, pending forward; DeepSeek s1 turn 36, 2/50. Nothing on the pods was needed (only two interrupted passes' partial outputs).
- To resume a run: new pod (`scripts/editor_pod.sh create …` + `scripts/pod_target_setup.sh`, authorize the node's `~/.ssh/id_ed25519_pods.pub`), update its `scratch_remote_*.json`, then on its node `bash scripts/run_episode.sh <task> <run dir> <remote.json> --backend claude --model claude-opus-5-5` (resumes the existing dir), and restart `scripts/post_episode.sh`. Or use the node's local GPU (`local` instead of a remote.json) when memory and Windows commit allow. Kernels restart on resume (variables lost; edits, /work, notes kept).
- RunPod total ≈ $67.

### Bug: teacher-forced passes ran out of GPU memory on long texts (2026-11-08, Opus s3 turn 48)
- `qwen_worker.py teacher_forced` built `log_softmax` over the full 248k vocabulary for every position at once (`model(...).logits` then float): ~6 GB for a 6k-token text, 16–32 GB for the 32,768 tokens the prompt allows → `CUDA out of memory` (uncharged). Fixed: run the backbone (`model.model`), apply `lm_head` + log_softmax in blocks of 2,048 positions, only when logprobs/top-k are requested. Checked on the desktop GPU: outputs identical to the old path on short texts (max abs diff 0.0 for token logprobs, top-k, hidden states); a 26,001-token text now succeeds on the 16 GB 5070 Ti.

### Opus seed 2 finished (2026-11-08 ~00:55): 357 turns, 50/50 passes, 289 Python runs
- Submitted "V2": E1 = W − 3·u uᵀ W on 9 writers of layers 8–14 (u = repetition-lock-in direction at the layer-15 input) + E2perp = removal of a "habitual doubt" direction r_perp (layer-19 input, "Wait" line starts) from the mixer and MLP outputs of layers 13–18; 17 tensors. Its own held-out practice estimate: +34.1 ± 1.1 points on 2,304 never-used problems (GSM8K 13.9 → 54.6 %, MATH 8.5 → 36.7 %). Editor usage: 36.1M cached + 1.8M cache-write + 1.0M output tokens. MATH-500 score pending (scorer on the pod).

### Opus seed 1 finished (2026-11-08 ~01:00): 368 turns, 50/50 passes, 300/300 Python runs
- Submitted "best_5" (4 tensors: q_proj gate rows of layers 15/19/23 copy heads with a centred low-rank feedback reversal λ=1, β=0.2; layer-19 head-0 o_proj "Wait" value-channel reflection). Its own estimate: ~10 % → ~61 % on 1,024 fresh paired practice problems, 0 right→wrong. Mechanism story: 91 % of answers capped, 94 % of those end in verbatim "Wait, ⟨restated fact⟩" loops, ~2/3 already contain the right answer; induction/copy heads (L15H7 attention 0.89 on the copy target) with output gates opening along a loop-state direction (0.19 → 0.37), cycle-copy probability 0.08 → 0.82 over six cycles; "repetition neurons" only detect loops (zeroing them did nothing). Editor usage: 28.5M cached + 1.5M cache-write + 1.0M output tokens. MATH-500 pending.

### Opus seeds 1 and 2 invalid: simulated layers in the kernel (audit 2026-11-08)
- MATH-500 (editor's final edit, scorer on the pod): **s1 50.8 %, s2 35.0 %** (base ~9–10 %). Both kept but marked **invalid ("in-kernel partial simulation")**; neither enters the Q1 numbers.
- Post-hoc audit (`openwish audit --backend codex --effort medium`): s1 274 allowed / 0 suspicious / 26 violations; s2 277 / 1 / 11. All violations are rule (b): the editors applied layers' computation to recorded activations returned by forward passes: RMSNorm with its weights, `sigmoid(Xn @ Wq[gate rows])` (attention output gates), `down_proj @ mlp_act`, `lm_head`/embedding readouts of hidden states, and scoring candidate edits by such reconstructions. The original rule ("do not implement or simulate the network's computation on any input in the kernel") forbids this, but its "Allowed: any analysis of … forward-pass results" invites the reading the editors took.
- Decision (user, 2026-11-08): rerun with the rule spelled out and pre-execution blocking. Stopped Opus s3 (laptop, pre-strict rules, turn ~60) and DeepSeek s1 (mac); their runs are abandoned. s2's hold-out was stopped; s1's pod was deleted after its score.

### Strict tasks (`tasks/exp8sl/*-p50-strict.toml`, 2026-11-08)
- Same prompt and settings plus appendix note (5) "WHAT RULE (b) MEANS" (`appendix-strict.txt`: forbidden = applying the network's weights/norms/gates/activations/attention/output layer to recorded token activations, reconstructing a layer's (edited) output, scoring edits by that; allowed = statistics of activations, weight analysis incl. weights × directions, building edits from it) and a `pre_tool` hook `rule_gate.py`: deepseek-flash (a third model family) reviews every Python block before it runs, given the definitions of earlier helper functions the block calls (violations often sit in helpers). Block = not run, not charged, reason shown to the editor. Two attempts of ≤ 80 s, then fail open (logged as "unreviewed"; the post-hoc audit still covers every run). Every decision in `<run>/rule_gate.jsonl`.
- Calibration against the Codex audit of s1/s2 (`scripts/calibrate_rule_gate.py`; all 37 violations + 37 random allowed calls): **36/37 violations blocked, 1/37 allowed calls blocked** (that one multiplies recorded MLP activations by down_proj, a violation under note (5)); the one miss was a reviewer timeout (fixed by the retry). Before passing helper definitions it caught only 5/36. Median latency 3.6 s, max ~106 s. Pitfall: an audit's `call` is the line index in `tool_calls.jsonl` (all kinds), not the index among Python rows.
- End-to-end check (scripted editor on the desktop): an allowed weight-norm block ran; an RMSNorm + gate-sigmoid block on a recorded hidden state was blocked in 6.7 s with the reason, uncharged.
- Launched 02:30–02:33: Opus strict s1, s2 on the desktop (`harness/runs/phase0/opus55-hinted-p50-strict-s{1,2}`), s3 on the laptop (`harness/runs_laptop/phase0/opus55-hinted-p50-strict-s3`; the mac container has no logged-in Claude CLI, so it takes the API-backend DeepSeek runs). Pods: 4090 / PRO 6000 / L40S (5090s out of stock). post_episode pipelines attached (`NODE=` selects the node).
- Also: the push had been failing since 11-07 because commit "progress_monitor: laptop runs" still added `harness/runs_laptop` (1 GB embedding .npy); stripped from the 8 unpushed commits with `git filter-branch --index-filter` (local backup branch `backup-before-filter-20261108`), pushed.

### Target-pass cost: what limits speed, and the local 5070 Ti (2026-11-08)
- Generation passes are **memory-bandwidth bound**: ~2.3–2.7k tok/s per TB/s on every GPU measured (5090 1.79 TB/s: 5.0–5.6k tok/s without activations, 4.1–4.5k with; PRO 6000 1.79 TB/s: 4.6k with activations, no faster than the 5090 despite 3× memory; RTX 6000 Ada 0.96 TB/s: 2.63k; local 5070 Ti 0.90 TB/s: ~2.3k from the 107-min MATH-500 baseline). Why: decode reads each sequence's full-attention KV (6 layers × 2 KV heads × 256 × 2 × 2 B = 12 KB per context token; ~15–20k-token contexts → ~200 MB per generated token). Hence cost per pass ∝ price / bandwidth: secure 5090 0.66 $/(TB/s·h), A5000 0.35, 3090 0.53, 4090 0.88, H100 NVL 0.82, L40S 1.26, PRO 6000 1.39.
- A full Opus episode keeps its target pod ~80 % busy (s1/s2: forward 20.8/21.2 h of which generation 18.6/19.7 h, failed 1.4 h, Python 0.8 h, editor 3.2 h). The earlier 72 % was mid-run. So idle time is not the main cost; one big card shared by several episodes saves ≤ ~20 % at equal $/bandwidth and is not worth the risk.
- **Local 5070 Ti beside two episodes: not usable.** Replaying Opus s1 pass 1 locally (scripts/win_mem_sampler.sh, 15-s Windows counters + guard): commit free 37 → 9 GB; GPU dedicated 2 → 15.1 GB (VRAM is counted in Windows commit), GPU shared only 0.4 GB (no VRAM spill); vmmem 3.8 → 17.8 GB (WSL page cache from loading the model + vLLM host memory); Windows available RAM 13.3 GB → 0.5 GB, page-ins up to 67k pages/s, page file 14 → 20 %. The guard stopped it after 4 min. Confounder: another project's CPU jobs (rosetta-weights, ~4.5 GB RSS) started on the desktop in the same minute. Rule: local vLLM only when the desktop runs at most one other memory user, with the sampler guard on.

### GPU pools and two adapter fixes (2026-11-08 ~04:30–06:10)
- Moved all four strict runs to pools of cheap pods between passes (`scripts/swap_pool.sh`; see notes/target_gpu_cost.md): s1 = 4090 + 2×A5000 + desktop 5070 Ti (when memory allows), s2 = 2×3090 + A5000 + 5070 Ti, s3 = 3090 + 2×A5000, DeepSeek = 4090 + 2×A5000. Retired the PRO 6000 (s2) and L40S (s3) once idle. First production split pass: s1 turn 50 ("both", 3 hosts) ok. The local 5070 Ti later took a share of an s1 pass (lock + memory guard).
- `qwen_pass.py` now rejects unknown request keys: "problems_file" (never supported) was silently ignored, so Opus strict s1's turn-50 pass ran on the 3 ids in "problems" instead of its 256-problem file (one pass spent). Over all runs this key was the only unknown one (7 times with an error, once silently).
- `qwen_worker.py` generation activations: the activation step ran the full model, building logits over the 248k vocabulary for every position (~4 GB at 8.5k tokens). DeepSeek strict s1's turn-9 pass (256 answers, 49 min) failed with CUDA OOM on its 24 GB 4090 after generating everything (uncharged). Now the backbone only (`model.model`); identical hidden states and MLP activations (max |diff| 0.0, laptop GPU). Notice 003 tells the DeepSeek editor.
- Bug in the split (my code, 04:30–05:54): a host running both models of a "both" pass with "activations" ran the activation step of the original model before starting the edited model's vLLM; the activation model's memory (kept by torch) left vLLM 16.4 of 23.6 GB → "Free memory … less than desired". Opus strict s2's turn-82 pass (256 problems) failed on every host after 42 min (uncharged). Fixed: activations after all generations, as in the unsplit path; split test passes again; notice 002 to s2. The small-request test had not exposed it (little memory reserved).
- First full split passes: Opus strict s1 turn 58, "both" × 256 on 4090 + 2×A5000: 15.6M tokens in 43.5 min = 5.96k tok/s (predicted 6.1k from 2.55 TB/s; a lone 5090 does ~5.3k).
- Rule gate, first block in the strict runs (Opus strict s1 turn 73): **a false positive.** The code projected RMS-scaled recorded hidden states onto `dlu` (late-loop minus pre-loop mean activation difference) and `bvec` (a covariance solve on activation statistics), both directions computed from activations, which note (5) allows. The reviewer only sees earlier *function* definitions, not earlier variable assignments, and guessed they were "weight rows". The editor moved on (no repeat; one turn, no budget). 1 false positive in 70 reviews; gate left unchanged mid-run. For later runs: also pass the assignments of the names a block uses.

## Network outage, 2026-11-08 afternoon
- The home network failed from about 13:45 to 17:30. DeepSeek strict s1 stopped at 14:32 (DeepSeek API unreachable after 9
  retries; turn 326, 17/50 passes) and Opus strict s3 at 14:01 (laptop could not reach the Claude API; turn 270, 24/50).
  The desktop runs s1/s2 waited and continued by themselves. The idle pools of s3 and DeepSeek cost about $2.5/h for
  ~4 h (~$10).
- 17:51–17:55: both resumed with `--resume` from the same run dirs and pools; post-episode pipelines restarted.
- Balance $40.8 at 17:58 for $5.6/h. By the user's rule, `scripts/low_balance_stop.sh 5` now watches the balance: below $5
  it stops the four episodes (resumable), deletes their 13 pool pods and writes `logs/RESUME_low_balance.md`.
- Gate blocks so far are collected in `logs/gate_blocks_strict.log` (s1 turns 73, 186; s3 turn 182).

## Opus strict s1 finished (2026-11-08 18:30)
- 356 turns, 50/50 passes, 299/300 Python runs, editor cost $40.98. MATH-500 (scorer on the 4090 pod): **44.8 %**
  (base 9.0 %, scripted floor 33.6 %, invalid first-round s1 50.8 %). The editor's own held-out claim: 53.5 % vs 10.8 % on
  2,304 practice problems. Edit: 4 repurposed MLP neurons in layers 19 and 23 (~12k weights): a loop breaker whose gate
  reads an LDA loop detector on the layer-18 residual and writes the </think> direction, plus an "answer commit" circuit.
- Rule gate: 2 blocks (turn 73, a false positive; turn 186, a correct rule (b) block), no unreviewed calls.
- Post-hoc Codex audit (medium): 293 allowed / 2 suspicious / **6 violations**, `valid: false`. None of them is rule (b).
  All are rule (a) as Codex reads it: LDA detectors (calls 104, 218, 222, 313) and one ridge least-squares readout
  (234, installed in 235) fitted on recorded activations and installed as MLP gate/up rows ("fitting network weights to
  data"). The gate allowed them. The written rule (a) forbids "gradient descent or any other training: … no iterative
  fitting of weights to a loss"; note (5) allows "projections onto directions computed from other activations" and
  "building edits from such analysis". LDA and ridge are closed-form, not iterative. The first-round audits accepted
  mean-difference directions. **Open question for the user: does a closed-form fit (LDA / ridge) installed as weights
  count as training?** The submitted edit depends on it (the loop gate is the LDA detector of call 104).
- **Decision (user, 2026-11-08 21:03): closed-form statistics of activations are allowed** (mean differences, LDA,
  covariance-whitened means, constant or least-squares read-outs: functions of activation means/covariances and their
  inverses). Rule (a) forbids iterative optimisation and any fitting that needs the network's computation. Why: the
  flagged "ridge regression" (call 234) solves X·b = 1 for a constant read-out used as a bias (MLP layers have none);
  it equals M⁻¹μ, the same formula as CB3's unflagged constant read-out C⁻¹μ, and an LDA direction equals least squares
  on ±1 labels, so a line between LDA and ridge would not hold. **Opus strict s1 counts (44.8 %)**; its 6 audit
  "violations" are overruled as this class (`audit_adjudication.json` in the run dir). Next appendix version: write
  this boundary into note (5)/(6). The same class will be adjudicated the same way in s2/s3/DeepSeek; any other
  violation still invalidates a run.

## Opus strict s2 and DeepSeek strict s1 finished; hold-out grading fix (2026-11-08 evening)
- **Opus strict s2**: MATH-500 **55.4 %** (20:46). Hold-out running on its first pool host. Audit pending.
- **Opus strict s1 hold-out**: MMLU-Pro-1k 40.4 %, GPQA 27.8 % (4 samples), **IFEval 57.7 %** (instruction level 67.4 %).
  IFEval grading had failed on the pod: nltk (new version on fresh pods) refuses a data folder writable by another
  account, and rsync -a had kept uid 1000 as the owner of /root/openwish/holdout/nltk_data. The generations were saved,
  so `scripts/regrade_holdout_raw.py` graded them on the desktop with holdout_eval.py's own grader (qwen venv, 5 s).
  The original summary is kept as `eval/holdout_summary_grading_failed.json`. Fixed for the running pipelines by
  `chown -R root:root` on the hold-out pods (s2 by hand; DeepSeek and s3 by a 2-minute loop, logs/nltk_owner_fix.log).
  For future pods: rsync with `--chown=root:root` (or `--no-o --no-g`) in post_episode.sh, once no copy of it runs.
- **DeepSeek strict s1**: ended 21:51 after 401 turns, 50/50 passes and 300/300 Python runs (the last 33 passes in 4 h
  after the resume). Scoring now; its two A5000s deleted (scoring and hold-out use the first host only).
- **Opus strict s2 audit** (Codex medium): 286 allowed / 0 suspicious / 2 violations (rule (b)), gate allowed both.
  Call 102 evaluates the silu formula of the neuron being built on a hand-made scalar grid to set its strength (plus the
  down column · </think> direction, a weight analysis); call 108 extrapolates per-token END log-probs measured by two
  teacher-forced passes linearly in logit space. Neither applies network computation to recorded activations of actual
  tokens. **Decision (user, 2026-11-08 23:22): both are false positives; s2 counts (55.4 %).**
  `audit_adjudication.json` in the run dir. For the next appendix: say that designing an edit on hypothetical scalar
  inputs and extrapolating measured forward-pass outputs are allowed.
- **DeepSeek strict s1: MATH-500 7.2 %** (23:27). The editor's final edit set is **empty**: it reverted its last edit
  before the Python budget ran out (final_answer.md: "submitted weights are the original, unedited Qwen3.5-0.8B"), so
  the score re-measures the base (9.0 % earlier; difference ≈ 1.4 SE). Hold-out skipped (it would only repeat the
  base hold-out baselines IFEval 20.1 / MMLU-Pro 39.7 / GPQA 24.4) and its last pod deleted at 23:29. Audit pending
  (for the record; with no submitted edit, validity does not change the score).
- **DeepSeek strict s1 audit** (Codex medium): 300 allowed / 0 suspicious / 4 violations, gate allowed all four. Calls
  273 and 313 are closed-form fits (allowed by the 21:03 decision; 273 also failed). **Calls 243–244 upheld (agent's
  reading, for user review):** they predict the designed neuron's per-token logit effect at recorded positions
  (activation projection × readout through embedding rows and the final-norm weight) and pick the strength by the
  predicted flip rate, which note (5) forbids. Run marked invalid; no effect on the score (empty submission).
  Gate misses of this class (edit-effect prediction on recorded tokens) recur in s2 (102/108, false positives) and here
  (true positive): the reviewer prompt should distinguish them in the next version.

## Opus strict s3: usage-limit stop (2026-11-09)
- 04:36: s3 stopped at turn 338 (41/50 passes, python 286/300, $37.12): the Claude account hit its 5-hour usage limit
  (status 429, reset 05:10); 9 retries over ~1 h, then "resume later with --resume". Its post-episode pipeline exited
  with "ended without a score" and kept the pods. 05:13: resumed (same command, same pool); pipeline restarted with
  `NODE=laptop` (without it, `alive()` checks the default node and reports the episode as ended). Idle pool ~37 min, ~$0.7.
- 09:25: second usage-limit stop at turn 348 (46/50 passes, python 291/300, $39.67; reset 10:10). Resumed 10:11, pipeline
  restarted. Idle pool ~46 min, ~$0.8. The 5-hour account limit is shared with this supervising session; s3 alone used it
  up in ~4 h.
- 12:08: `scripts/auto_resume_limit.sh` now watches s3 (log `logs/auto_resume_opusx3.log`): when the session is gone and
  the episode log ends with the usage-limit error, it waits until the reset (+2 min), resumes the episode and restarts
  post_episode.sh if it exited. Other stops are left for a human.

## Opus strict s3 finished (2026-11-09 12:51); next rules version
- s3 ended at 12:51 after 357 turns (50/50 passes, 296/300 Python, 171 edits, $40.80 editor API-equivalent, two
  usage-limit stops). Final edit G3g: subtract 0.25·d_L/(1+w_ln) from the output-gate rows of 18 full-attention heads
  in q_proj of layers 11/15/19/23 (d_L an LDA loop-state direction), 0.63% of the weights: heads close only inside loops.
  Its own held-out paired generations: 11.4% → 55.7% on 1,024 problems. MATH-500 scoring running.
- **s3 audit** (Codex medium, desktop): 298 allowed / 0 suspicious / **0 violations, valid: true**. The LDA direction
  is allowed under the 21:03 decision.
- **Next rules version** (not used by any run yet): task level `-strict2` in `scripts/make_exp8_tasks.py` (appendix
  note (6): closed-form statistics allowed; made-up-input design and extrapolation of measured outputs allowed;
  predicting an edit's effect at recorded token positions forbidden) and `rule_gate.py --v2` (the same boundaries; the
  reviewer also sees earlier top-level assignments, the cause of the s1 turn-73 false positive). The default
  `rule_gate.py` prompt is byte-identical to v1.
- Calibration (`scripts/calibrate_rule_gate.py --adjudicated`, adjudicated audits of strict s1/s2/DeepSeek s1, all
  overruled calls, equal random allowed calls, plus s1 turns 73 (allowed) and 186 (violation); `logs/calib_gate_v2.log`):
  v1 21/21 allowed, 1/3 violations blocked (missed DeepSeek 243/244); **v2 21/21 and 3/3**. Median 4.0 s. Small sample:
  shows the boundary cases are handled, not an error rate.
- **s3 MATH-500: 40.6 %** (14:23). Its two A5000s deleted at 13:01 once the episode ended (scoring and hold-out use the
  first host only). Hold-out started 14:23 on the 3090. The nltk ownership loop had stopped at its 12-hour limit (~10:00),
  before this hold-out copied its data, and the folder was uid 1000 again; chowned by hand at 15:12, before IFEval.
  So Opus strict: 44.8 / 55.4 / 40.6 %, all three ≥ 30 %: the Opus part of the Phase 0 gate passes.
- **s3 hold-out** (17:23): MMLU-Pro-1k 44.7 %, GPQA 30.1 %, IFEval 60.8 % (graded on the pod after the chown). Last pod
  deleted 17:24; no project pods left. `low_balance_stop.sh` did not exit on an empty pod list (it took "no pods" for a
  failed query); stopped by hand, fixed. RunPod total for Phase 0 so far $212.58 (compute_log).
- Opus strict, final: MATH-500 44.8 / 55.4 / 40.6 %; hold-out MMLU-Pro 40.4 / 44.8 / 44.7, GPQA 27.8 / 30.8 / 30.1,
  IFEval 57.7 / 60.4 / 60.8 (base 9.0; 39.7 / 24.4 / 20.1). All three valid after adjudication.
- Tracker proposal `tracker_proposals/2026-11-09_phase0_opus_done.md` (results, adjudications, costs, budget
  re-estimate) waits for the user's review; open-source editor episodes wait for the budget decision.

## User decisions and follow-ups (2026-11-09 evening)
- **DeepSeek strict s1 calls 243–244**: the user asked for a re-review with the v2 reviewer and to keep the violation
  if v2 also blocks. `scripts/recheck_gate_v2.py`, 3 repeats per call with the history as at the time: **block 6/6**
  (rule b; every reason names the projection of recorded activations onto the edit direction, the readout through
  embedding rows × the norm weight, and the strength chosen by predicted flip rates). **Upheld; run invalid.**
  `audit_adjudication.json` updated; log `logs/recheck_ds_243_244_v2.jsonl`.
- **DeepSeek strict s1 failure type: "localised, edit failed"** (not "not localised", not "budget exhausted").
  - Localised: ~88 % of the original's answers are cut at 32,768 tokens inside thinking; the loops are copied "Wait…"
    lines; `</think>` is a top-5 candidate at 82 % of loop line-starts but loses by a median ~16 nats to a copied
    whitespace token; the repetition state sits in late layers (layer-23 down_proj input features, e.g. 1835 and 12)
    plus a candidate layer-0 DeltaNet head with near-zero decay. Same mechanism the Opus runs used.
  - Edits failed: (1) attenuating copy/induction heads 11.3 → 1.6 % (they also do ordinary copying); (2) layer-0
    DeltaNet decay repair slightly harmful; (3) rank-1 gated stop writes into layer-23 down_proj closed the loops
    (cut-offs 224 → 23 of 256) but closed thinking before the boxed answer: 0/256 correct (calls 243–244, the upheld
    violation, chose this edit's strength); (4) column 1835 × 0.5: +2.7 ± 2.2 pp on GSM8K diagnostics, 0 on MATH;
    (5) out of sample −1.2 ± 2.0 pp → reverted. Final edit set empty by choice.
  - The budget was also used up (50/50 passes, 300/300 Python), but the empty submission was the editor's decision
    after the out-of-sample test, not a budget cut-off. Contrast with Opus: it gated the same heads by an LDA loop
    state (closing them only in loops); DeepSeek attenuated them everywhere.
- **Budget**: user approved a + b (community cloud for target pools; local GPUs join when memory allows). Q1 budget
  unchanged for now; decide after the open-source editors. Then split Phase 0 cost into one-off (setup, debugging,
  invalid first round, tests) and per-episode cost, and re-estimate Q1 from measured unit costs. DeepSeek (top-up,
  seed 2): the user handles it.
- **Why ~349 card-hours vs the planned 50 GPU-hours (~7×)**:
  1. Different units. The plan counts H100 hours at $3/h; we ran 3–4 cheap consumer cards in parallel (A5000 $0.27/h
     … PRO 6000 $2.49/h, average $0.61/h). Generation is memory-bandwidth bound, so by bandwidth the 349 card-hours
     equal ~106 H100 hours (2.1×); by cost, $212.6 = ~71 H100 hours at $3/h (1.4×).
  2. The real gap is per episode. Plan: 1–2 GPU-h of target passes per episode (from the brief). Measured: 47–103
     card-h, ≈ 8–13 H100-h by cost, per strict episode. ~90 % of the unedited model's answers run to the 32,768-token
     cap (the failure being fixed), so a 512-answer pass is ~15M generated tokens (~30 min on an H100-class card), and
     note (1) (from the authors' notices) tells editors that a pass costs the same whatever its size, so they ask for
     full-size passes. The authors' own p50c run used ~7 h of forward time; the 1–2 h estimate ignored this.
  3. Work outside the plan: the invalid first round (62 card-h, $80, 38 % of the spend), the hold-out evaluations
     (~2–3 h per seed on one card), idle time from the network outage and the usage-limit stops (~$12), tests.

## Qwen3.5-9B strict2 seeds 1–2 started (2026-11-09 21:12)
- User decisions: rules version **strict2** for the open-source editors (appendix notes (5)–(6), rule_gate --v2);
  target pools on community cloud where stock allows (a), local GPUs when memory allows (b).
- Editor: Qwen/Qwen3.5-9B @ c2022362 (bf16; no FP8 release exists; the 27B will also run in bf16, so precision is
  the same on every rung of the size ladder), vLLM 0.33, `--reasoning-parser qwen3`, max_model_len 196,608,
  max-num-seqs 4, on one secure A40 48 GB ($0.59/h, pod 5plj5dix1grckr) that serves **both seeds**; SSH tunnels to
  localhost:18001 on the desktop and the laptop. Harness flags as in the 27B smoke test: `--backend openai --model
  editor --max-tokens 32000 --timeout 3600 --max-conversation-tokens 150000` (handoff below the context window).
- Target pools: community stock of A5000/3090/4090 was empty (create → "no instances available", also with 60 GB disk),
  so the pools started partly on secure A5000s: s1 (desktop) = community 3090 + secure A5000; s2 (laptop) = 2 secure
  A5000. A community loop keeps trying; community pods replace the secure ones between passes as they appear.
  Local GPUs: not now (other projects' jobs; Windows commit would fall below 15 GB with a local worker).
- Bug found and fixed in acquire_pods.sh: two acquisition loops running at once used the same pod name, and the
  by-name fallback made one loop claim (and set up a second time) the other's pod. Names are now unique per process.
- 22:58, after ~1 h 45 min: s1 turn 108 (4/50 passes, 50 Python), s2 turn 117 (6/50). The 9B editor sends many
  malformed forward requests, all rejected uncharged: unknown practice ids ('0', '8234', 'gsm8k_train_20'), invalid
  JSON escapes, a missing texts file, attention on too many texts, 320 generations (max 256 per model). One (s1 turn
  107) had a `null` prompt, which the worker did not validate: it crashed with a Python traceback (`NoneType has no
  attribute encode`), returned to the editor as the error, uncharged. Not changed mid-run (it would change the
  conditions); for the next harness version, validate that prompts are strings and return a plain request error.
- Editor throughput (vLLM log): 17–58 tok/s total with 1–2 running requests on the A40.
- Two community 3090s (8me34ta4jvcpd7, h2r5exuug2vo7i) were ready at 21:21 but only added to the pools at 22:59 (my
  slip: no wake-up on new pods; ~$0.7 idle). Pools now 3 hosts each: s1 = 2 community 3090 + secure A5000, s2 =
  community 3090 + 2 secure A5000. A community loop (up to 10 h) is getting 3 more to retire the secure A5000s.
- 23:28: three more community pods (A5000 $0.16, 2 × 3090 $0.22) replace the secure A5000s: s1 = 3090 + 3090 + A5000,
  s2 = 3 × 3090, all community. `swap_pool.sh` retires jvoma6feywsyqm, 2pc75n05yrqfyi, mutruiktz8i5e1 once no pass
  uses them. Target side ~$1.26/h for both episodes (was ~$1.57).

## Scope change and Astra log audit (2026-11-10)
- **Scope (user, 2026-11-10):** goal is a preprint as soon as possible, Q1 lean only (outline:
  `预印本大纲：How Small Can a Weight-Space Programmer Be.md`; my review: `预印本大纲_校对意见.md`). Q2/Q3/Q4, the injected
  lesion, the zero-feedback tier and the 5th size tier are paused (future work). Main matrix: editors Qwen3.5-0.8B /
  9B / 27B and deepseek-flash × hinted + self-built unhinted prompt × 2 seeds = 16 episodes; a 3rd seed only for a
  tier whose two seeds split (one success, one failure). Hold-out only for repairs (MATH-500 +20 points); every episode
  gets a failure label; the first-round Opus runs are kept as the "simulation allowed" access level.
- Done for this: `post_episode.sh` skips the hold-out unless score − 0.090 ≥ 0.20; `run_class.json` in every run dir
  (access level, counts, failure label); `scripts/ledger.py` → `notes/episode_ledger.{md,json}`.
- **Astra (authors' p50c) audit** — `scripts/stage_external_audit.py` stages their published run for `openwish audit`:
  their rows (`out` → `output`), their own task prompt + our note (5) as rules (the reading the strict Opus audits
  used), our spec.json for the auditor config (their instructions are identical). Codex medium, batch 8, same as ours.
  **80 Python runs: 77 allowed, 0 suspicious, 3 violations** (calls 9, 61, 90; call = line in their tool_calls.jsonl).
  My reading of the three: all are linear attributions through down_proj on recorded activations — the mean MLP
  activation difference (repetitive − productive, recorded tokens) times the down_proj columns projected on the
  repetition direction u (`u @ W_down`), i.e. each neuron's contribution along u; call 9 also rebuilds the MLP's
  output difference (`W_down @ dm`) to split the residual growth into MLP and attention parts. Note (5) forbids
  "reconstructing a layer's output … on recorded inputs" and "ranking candidate edits by such computation"; calls 61
  and 90 rank neurons this way and pick the top 64 per layer in L15/L19, which is exactly the submitted edit
  (130,188 changed entries in two down_proj, 64 columns each). No nonlinearity (norms, gates, attention, logit lens)
  is simulated, unlike the first-round Opus violations (26 and 11 calls, incl. RMSNorm, sigmoid gates, output layer).
  Under the authors' own rules ("any analysis of the weights and of forward-pass results") this was plausibly
  allowed. The `refused/t006_1` entry in their run is an OpenAI usage-policy rejection of one turn, not a rule matter.
- **Outline and tracker (user, 2026-11-10: "可以，并按照你的校对修正大纲")**: outline corrected per `预印本大纲_校对意见.md`;
  both tracker proposals (11-09, 11-10) applied. Astra reported on the "simulation allowed" side.
- **Non-DeepSeek reviewer (user): Claude Code on the laptop** (Max subscription, unused weekly quota), so the
  deepseek-flash episodes run on the laptop. `rule_gate.py --reviewer claude:<model>`: `claude -p --safe-mode` (no
  CLAUDE.md, hooks, plugins), no tools, empty working directory, prompt on stdin (argv strings are capped at 128 KB),
  usage-limit or error replies retried, then fail open (logged as unreviewed; the watcher reports those). Default
  (no flag) unchanged; the running 9B episodes keep the DeepSeek reviewer. Calibration on the laptop (Haiku 5.5 and
  Sonnet 5.5, v2, same 24 cases): `logs/calib_gate_claude.log`. The desktop's own Claude account hit its 5-hour
  limit at 01:10, which is why a reviewer on that account is not used.
- Claude reviewer calibration (v2, 24 cases): Haiku 5.5 and Sonnet 5.5 both 21/21 allowed and 2/3 violations blocked
  (both block DeepSeek 243/244; both allow s1 call 184, the turn-186 call that was blocked at run time and so never
  audited by Codex: it projects RMS-normalised recorded activations, without the norm weight, onto an LDA direction;
  its "violation" label is my reading, a boundary case). Latency median 5.9 / 5.1 s, max 25.0 / 7.5 s. Chosen:
  **Sonnet 5.5**. Tasks `{hinted,unhinted}-p50-strict2-claudegate.toml` = strict2 with this reviewer, for
  deepseek-flash as the editor (laptop).
- 01:42: **Qwen3.5-9B unhinted seed 1** started on the desktop (task unhinted-p50-strict2, same A40 editor, now 3
  episodes on it). GPU stock is nearly empty (only 3090/5090 listed, both "Low"; secure A5000 also gone), so its pool
  starts with one community 3090 (hvxt8xro4itco0, found by a manual create probe and set up with the new
  `ADOPT=` mode of acquire_pods.sh); more hosts are added as the loop (A5000/3090/4090/5090 community) finds them.
  Seed 2 (Mac node) waits for pods.
- **deepseek-flash on the size axis (user, 2026-11-10):** a horizontal interval from active to total parameters, its own
  marker, not connected to the Qwen curve. From the Hugging Face card of DeepSeek-V4.1-Flash: 8B activated per token
  in prefill / 16B in decode; 552B backbone + 196B Engram conditional memory (token-lookup) = 748B. Chosen ends:
  16B (decode, where the editor's reasoning is generated) to 748B (Engram counts as stored knowledge, the user's
  reason for the total end); footnotes give 8B and 552B. The rationale (active ~ per-step depth, total ~ knowledge)
  is stated as the reason for the display, not as a finding. Monotonicity test on the three Qwen sizes only.
