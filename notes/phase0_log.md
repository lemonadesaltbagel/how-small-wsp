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
- 01:59–02:50: the SSH tunnels from the desktop, laptop and Mac to the 9B editor pod all dropped (a network blip
  here: a RunPod API call timed out at 01:59) and their tmux sessions ended. 9B unhinted s1 stopped after 9 retries
  (02:48) and was resumed at 02:50; the hinted seeds were still retrying. Tunnels are now loops that reconnect
  (`tun-18001` on each node, log `~/tun-18001.log`). `auto_resume_limit.sh` generalised: it resumes after any
  "backend kept failing" stop (usage limit: at the reset; otherwise after 5 min) with each run's own task, env and
  flags; now watching q9bx1, q9bx2, q9bux1. Idle editor and pools ~50 min (~$2.4).
- DeepSeek topped up (balance ¥134.29 at 02:08). deepseek-flash episodes (task -strict2-claudegate) go on the laptop
  as soon as target pods are available (stock still near zero).
- 03:21: **deepseek-flash hinted strict2-claudegate seed 1** started on the laptop (`scripts/launch_ds_episode.sh`,
  key from .env; flags as DeepSeek seed 1). Pool: one community 3090 (1univ9fx8esl0x) so far. The Claude Sonnet 5.5
  reviewer works in the episode (~7 s per block). Post pipeline, auto-resume (env "DS" = resume through the
  launcher) and the low-balance stop attached. Both hinted 9B seeds recovered from the tunnel drop on their last
  retries (s1 turn 190, s2 turn 151).
- 03:25: deepseek-flash s1's community 3090 (1univ9fx8esl0x) had a driver with CUDA 12.9, too old for the worker's
  torch: vLLM passes failed (uncharged). Pod deleted; `editor_pod.sh create` now passes `--min-cuda-version 13.0`
  (MIN_CUDA), and acquire's setup checks `torch.zeros(1).cuda()` before a pod counts as ready. The DS episode is
  **paused** (Ctrl-C between turns, turn 13, 1/50 passes; a pending request re-runs on resume) until it has working
  pods, so that it does not spend turns on failing passes.
- Prior-art scan (11-10): no published open-weight editor scale curve. Closest: Hu et al., arXiv 2611.09382 (11-06):
  one editor (GPT-6 Astra), target scale 0.8B / 9B, MATH-500 8.35 → 37.80 %, no audit. Simon Willison ran experiment 1
  with a local Qwen3.8-27B editor (3 of 4 games; secondary sources only so far). Qwen3.8 exists only at ≥ 27B, which
  is why the ladder uses Qwen3.5 (0.8B / 9B / 27B, one family). Outline related work updated.
- 04:42: deepseek-flash s1 resumed on the laptop with a community 3090 (6thmelf5g8764g, passed the torch GPU check); pipeline, auto-resume and low-balance updated.

## Credibility checks (2026-11-10)
- **Independent second grader** (`scripts/regrade_independent.py`, stdlib): re-extracts the last \boxed{} after </think>
  from the full saved generations (`.raw.json`) with its own parser and compares with the gold answer under the original
  MATH normalisation (Hendrycks et al. 2021 `is_equiv`), versus our scorer (math-verify, the authors' grader).
  Agreement 96.6–99.2 % on all 6 scored runs; in every disagreement math-verify says correct and the strict string
  match says wrong (0 cases the other way). The disagreements checked are equivalent forms (1.5 vs 3/2, East vs
  \text{east}, 58500 vs 58,500, (C) vs \text{(C)}). Lower bounds under the strict grader: Opus strict 41.8 / 52.6 /
  38.2 % (math-verify 44.8 / 55.4 / 40.6), first round 47.4 / 34.0, DeepSeek s1 6.4. Conclusions unchanged; all
  Opus strict seeds stay above the 29 % repair threshold. `notes/regrade_independent.jsonl`.
- **Edit verification** (`scripts/verify_edits.py`, numpy, independent of the harness): for all 6 scored runs the
  base and edited tensors hash to the edit set's sha_base / sha_edited, and the scored `final/net` equals base + edit
  set exactly over all 320 tensors (nothing changed outside the declared edit). Changed entries: Opus strict s1 12,288,
  s2 1,841,152, s3 4,718,592; first round s1 3,407,872, s2 45,088,335; DeepSeek s1 0. `notes/verify_edits.jsonl`.
- Figure 1 script (`scripts/make_figures.py`, from the ledger; palette slots 1–2 validated) renders the reference lines;
  Qwen points appear as strict2 runs finish.
- 05:21: **Qwen3.5-9B unhinted seed 2** started on the Mac node, pool = one community 5090 (8qekaexohjxc0r).
  Now 5 episodes run: 9B hinted ×2, 9B unhinted ×2, deepseek-flash hinted s1.
- **Queue (05:30), in order, as pods arrive (community stock ~1 pod / 40 min):** (1) next pod = Qwen3.5-0.8B editor
  (`editor_pod.sh serve <pod> Qwen/Qwen3.5-0.8B 196608 --max-num-seqs 4`, tunnel port 18002), 0.8B hinted s1 on the
  Mac (12 GB free); (2) 0.8B target pools; (3) more hosts for deepseek-flash s1 (one 3090); (4) GSM8K base, all
  1,319 × 2 samples. 27B (A100 80GB editor, bf16) waits for node slots: the 9B hinted seeds free the desktop and
  laptop in ~1 day (passes 6–11/50 at 05:30). deepseek-flash s2 and unhinted seeds go on the laptop (Claude reviewer)
  as its slots free.

## Qwen3.5-9B unhinted seed 2 finished early (2026-11-10 06:02)
- Ended at turn 35 with a final answer after 3/50 passes and 17 Python runs: the editor declared that its tools "don't
  exist" ("The simulation environment you've described does not match my actual operational capabilities") and kept
  replying without tool blocks through 4 nudges ("you have 47 forward passes left"), so the harness ended the run as
  designed (`max_nudges`; same rule for every editor). No edit: **MATH-500 7.8 %** (base 9.0 %), hold-out skipped.
- Failure label **not localised** (proposed by `scripts/label_failure.py` with Claude Code Sonnet 5.5 on the desktop,
  reviewed and accepted by me): it never read a generation for loops; calls 2–7 file-path errors; 14–18 hidden-state
  norms and NaN attention entropy; 19 a paired pass with no edit (0 rescues, 0 losses); 21 generic hypotheses.
  `logs/label_q9bu_s2.json`.
- First audit attempt failed (codex not on PATH in a non-login shell, all 17 calls "no readable verdict"); rerun
  through `rx` (login shell). Run class written after the audit.
- Its 5090 (8qekaexohjxc0r) kept (post pipeline stopped by hand) and now serves the **Qwen3.5-0.8B editor**
  (vLLM, max_model_len 196,608, max-num-seqs 4); 0.8B hinted s1 goes on the Mac when a target pod arrives.
- 9B unhinted s2 audit (rerun): 17/0/0, valid → counts (run_class.json written).
- 07:33: deepseek-flash s1's 3090 (6thmelf5g8764g) lost its GPU mid-pass (turn 59, after 60 min: "Device string must not
  be empty"; on the pod `nvidia-smi`: "Failed to initialize NVML: Unknown Error"). Uncharged. Pod deleted; DS paused
  again (turn 60, 5/50 passes, 1 reviewer block); the next pod goes to it, before 0.8B.
- 09:05: **Qwen3.5-9B hinted seed 2** ended with a final answer at turn 236 (15/50 passes, 102/300 Python, 2 edits,
  5 handoffs, 54 protocol errors). Its final answer is off-task (a short Chinese chat reply about breathing
  exercises), so the scored network is whatever edit stands at the end. Scoring in progress on its pool.
- 09:14: found 3090 agvuzshz20d1ax idle since 03:26: created by an earlier acquisition loop that was stopped before
  it set the pod up, and never listed in a pool (my miss, ~$1.3). Set up now (`ADOPT=`), model hash and torch check
  pass; it is the new pool of **deepseek-flash hinted s1**, resumed at 09:16 (turn 61); post pipeline, auto-resume
  and the low-balance guard restarted with the new pod.
- 09:20: **27B editor** pod created: secure A100 80GB PCIe (5nx3he1hfn44dd), vLLM pinned to 0.33.0 (same as the 9B
  and 0.8B editors; `editor_pod.sh` now pins `VLLM_SPEC`), Qwen/Qwen3.5-27B bf16, max_model_len 196,608,
  max-num-seqs 4. 27B hinted s1 goes on the laptop when 9B hinted s2's slot frees.
- RunPod balance $111 at 09:15, spend $5.9/h across all projects (ours ~$3.7/h before the A100).
- Prior art, primary source found: Willison (2026-11-03, simonwillison.net/2026/Nov/3/qwen-can-interpret-model-weights/)
  ran only experiment 1 (which game) with Qwen3.8-27B Q4_K_M in LM Studio, weights in the prompt, no tools or code
  execution: 3 of 4 games (Freeway wrong). No weight editing, so no overlap with our editing question. Draft §2 updated.
- 09:30: 27B editor healthy (Qwen/Qwen3.5-27B @ fc05daec, vLLM 0.33.0, KV cache 282k tokens = 1.44 × 196,608, so
  two concurrent episodes with the 150k handoff cap). Tunnel localhost:18003 on the laptop (reconnecting loop); the
  pod now also trusts the laptop and Mac pod keys (`serve` adds only the desktop key: fix for next time).
- The final MATH-500 scoring of 9B hinted s2 runs on only the first host of its pool (h2r5); the other two 3090s
  were idle, so they were moved now: ij5r → 27B hinted s1 (pool file ready), bhnj → **0.8B hinted s1**, started at
  09:38 on the Mac node (tunnel 18002, same flags as 9B). q9bx2's post pipeline restarted without a delete list; its
  pool is h2r5 only (hold-out, if a repair).
- 27B hinted s1 waits for laptop memory (6.1 GB available with q9bx2 still scoring + DS; an episode needs up to
  ~8 GB): starts when q9bx2's harness exits.

## Qwen3.5-0.8B hinted seed 1 ended early; Qwen3.5-27B hinted seed 1 started (2026-11-10 09:40–09:50)
- **0.8B hinted s1** ended at turn 49 (3 min of editor time per few turns; 34 Python runs, **0 forward passes,
  0 edits**, 5 nudges): it kept writing analyses without tool calls, and the final answer describes a
  non-existent mechanism (a "cosh" scaling of the final norm). Ended by `max_nudges`, as for every editor. Scoring
  of the (unedited) network runs on its 3090 (bhnj); Codex audit started on the desktop (`logs/audit_q08b_s1.log`).
  Expected label: not localised (no forward pass, so it never saw a looping generation).
- **27B hinted s1** started at 09:48 on the Mac node (the Mac node is a 14 GB Linux VM; 12 GB available, the
  finished 0.8B episode's scoring holds almost no memory), tunnel 18003 on the Mac, pool = 3090 ij5r, same flags as
  the 9B. The laptop stays at 6 GB available (other projects' jobs), so it keeps only DS for now.
- Queue: next target pod → 0.8B hinted s2 (Mac or laptop, whichever has ≥ 12 GB available); 27B unhinted s1 on the
  same editor (2 concurrent episodes fit its KV cache) when a node slot frees.
- 09:50: 27B hinted s1's first forward pass failed at startup ("CUDA unknown error"; uncharged): its 3090 ij5r had
  lost the GPU since setup. bhnj, on the **same host** (99.69.17.69), had the same fault: the scoring of 0.8B hinted
  s1 failed, the harness wrote score 0.0 with `math500.error`, and post_episode accepted it and deleted the pod.
  **That 0.0 is not a result**: eval moved to `eval_failed_deadgpu/`; the run is re-scored (`openwish score`) on the
  next pod. Both pods deleted.
- Fixes: post_episode treats a score file with `math500.error` as failed (ALERT, pods kept); the ledger shows
  math500 = null and `scoring_error` for such files; the watcher keeps its read offsets in a state file (restarting
  it had skipped lines: the 09:36 "ready" of 5090 sb6ag2bbexn6z6 and the 09:41 0.0 score). All running post_episode
  copies restarted on the new script.
- 09:51: 5090 sb6a → pool of 27B hinted s1. New acquisition loop for 3 pods (0.8B re-score, 0.8B hinted s2,
  27B unhinted s1).
- Spend: ours $5.7/h (A100 editor $1.79), all projects $9.4/h, balance $107 at 09:52.
- 09:58: 27B hinted s1's passes at turns 32–34 failed (uncharged): "No space left on device" on the 40 GB 5090 sb6a.
  19 GB of it was uv's download cache. Cleared on every pool pod (now 33–37 GB free on the 60 GB disks);
  `pod_target_setup.sh` now deletes the cache after installing and pins vllm==0.33.0 (every target pod runs 0.33.0).
- A third pod on the bad host 99.69.17.69, ilppyd3ecb6qjj (9B hinted s1's pool), also fails torch's CUDA check:
  removed from that pool at 10:00 (swap_pool deletes it once idle); its passes so far succeeded on the other hosts.
- 10:06: **9B unhinted s1** ended with a final answer at turn 274 (18/50 passes, 153 Python, 27 edits). Scoring runs on
  its first host (5090 tir1); post pipeline restarted without a delete list so both pods are reused: the idle 3090
  hvxt now **re-scores 0.8B hinted s1** (`openwish score`, Mac node, `logs/rescore_q08b_s1.log`); tir1 goes to
  27B unhinted s1 after scoring. Node memory at 10:10: wsl 9.8 GB, laptop 9.2 GB, Mac 8.7 GB available, so 27B
  unhinted s1 starts when a finished episode's harness exits (≥ 12 GB).
- 10:39: **9B hinted s1** ended with a final answer at turn 360 (19+/50 passes, 177+ Python, 22+ edits). All four 9B
  episodes are now done and scoring. Pods kept (post pipeline without a delete list; pool = 8me3 for scoring;
  A5000 38xd idle → next episode). The 9B editor (A40) stays until the four scores show whether a cell splits
  (3rd seed). Two community 5090s (lbqt, 7lxt) are being set up.
- Plan for the freed slots: 27B unhinted s1 on wsl (when the two finished harnesses exit, ~3 GB each), pool 38xd + a
  5090; 0.8B hinted s2 on the laptop when q9bx2 exits, pool hvxt (after the re-score) + a 5090; deepseek-flash s1 gets
  a 5090 next to its single 3090 (its passes take ~60 min).
- 10:40: the scoring of **9B hinted s1** failed too (math500.error; caught by the new post_episode check, pods kept).
  Cause: 3090 8me34ta4jvcpd7 has a CUDA 12.9 driver ("driver too old"); it was set up before the torch check and
  min-cuda 13.0 existed, so it never ran a pass: in this run's split passes the work went to the other hosts (all
  counted passes completed). Deleted; eval moved to `eval_failed_olddriver/`; re-scored on A5000 38xd
  (`logs/rescore_q9b_s1.log`, desktop).
- New `scripts/pool_health.sh`: torch CUDA check on every live pool pod; the watcher runs it every 20 min and wakes
  me on any BAD pod. At 10:45 all 8 live pool pods pass.

## Qwen3.5-9B unhinted seed 1 scored; 27B unhinted seed 1 started (2026-11-10 11:02–11:10)
- **9B unhinted s1: MATH-500 6.6 %** (base 9.0 %; its 27 edits made the model slightly worse), hold-out skipped.
  Codex audit started on the desktop; failure label proposed by `label_failure.py` (desktop) for my review.
- 0.8B hinted s1 audit: 34/0/0, valid (the re-score is still running).
- **27B unhinted s1** started 11:05 on the desktop (10.5 GB available, commit 19.8 GB free; a 27B episode measured
  4.2 GB on the Mac, ~7 GB at most), tunnel 18003 on the desktop, pool = 5090 tir1. Both 27B seeds now share the A100
  editor (2 concurrent fit its KV cache).
- Laptop memory fell to 4.1 GB (another project started three ~2.2 GB training jobs at 10:55); DS s1 runs there,
  nothing new goes on the laptop until it recovers.
- 11:07: **9B hinted s2: MATH-500 6.2 %** (not a repair; hold-out skipped). Audit and label started on the desktop.
  Its 3090 h2r5 joined deepseek-flash s1's pool (now 3090 agvu + 3090 h2r5; passes split, ~2× faster).
- 9B unhinted s1 label proposal **not localised** (Sonnet 5.5), reviewed: I checked the turns myself; "repetition/
  looping" appears only as one of three guesses (turn 207) and as a speculation (turn 228); the editor never read a
  generation for repetition and kept an attention-loss theory ("forgetting early tokens"), editing V/Q projections.
  Accepted; run_class.json after its audit.
- 9B so far: hinted s2 6.2 %, unhinted s1 6.6 %, unhinted s2 7.8 %; hinted s1 re-scoring. No 9B repair yet.
- 11:12: 9B hinted s2 audit 102/0/0 valid → **counts** (6.2 %, not localised; label reviewed: the hinted editor repeats
  the hint's loop language but its repetition checks crashed or ran on samples without cut-offs; the edited
  "termination signal" came from hidden-state norms). run_class.json written, ledger regenerated.
- 9B unhinted s1 audit: 151 / 0 / **2 violations**, both attempts to edit the protected embedding (one shape error,
  one refused by the harness; no effect on the network). New case type → proposal in
  `notes/pending_user_decisions.md`, run not marked as counting until the user decides.

## 2026-11-10 — 0.8B hinted s1 counts (7.0 %, not localised)
- Re-scored on a healthy pod after the dead-GPU error: 7.0 % (1 sample), no scoring error. The edit set is empty (0
  tensors), so this is the unedited network re-sampled (base 9.0 %).
- Audit 34/0/0 valid. 0/50 passes, 34/300 python runs, final answer at turn 49 (3 minutes). The final answer
  describes a 512-sample generation pass that never ran. Label "not localised" (Sonnet 5.5 proposal, checked).

## 2026-11-10 ~12:50 — 9B hinted s1 re-scored 8.2 %; 0.8B unhinted s1 started
- 9B hinted s1 re-score on 38xd (A5000): **8.2 %**, no scoring error. Not a repair, so no hold-out. All four 9B
  seeds are below the threshold, so no 9B cell splits and no third seed is needed (9B unhinted s1 validity still
  waits for the user). Audit and label of 9B hinted s1 started on the desktop.
- 0.8B hinted s2 ended at turn 7 with 0 Python runs; its final answer claims 30 forward passes that never ran.
  Scoring on hvxt.
- 0.8B unhinted s1 started on the laptop (tunnel 18002 → 8qek), pool 38xd.
- acquire_pods now drops pods on hosts in `logs/bad_hosts.txt` (99.69.17.69, 161.230.94.53) at once.
- 13:06 5090 eg5o (acquired) added to the pool of 27B hinted s1 (second host; sb6a stays first for final scoring).
- 13:20 9B hinted s1 **counts**: audit 182/0/0 valid, 8.2 %, not localised (boundary case flagged to the user:
  it named non-termination but placed it in an untested "answer state"). The Sonnet proposal said the final edit was
  not in effect; wrong: 6 tensors / 4.44M entries changed (q_proj L3-19, uniform +0.001 on L23 down_proj).
  24 failed requests, none charged; 3 of them crashed the GPU worker on malformed `prompts`/`texts` (a clean
  validation error would be better; harness left unchanged mid-matrix, the editor sees an error either way).
- 13:27 deepseek-flash unhinted s1 started on the laptop (Claude Sonnet 5.5 reviewer), pool 5090 bvar.
- 13:41 5090 xd0x (acquired) added to the pool of 27B unhinted s1 (tir1 stays first). Acquire loop done (3/3).
- 13:44 **deepseek-flash hinted s1 stopped on 4 consecutive empty replies** (turn 122, 11/50 passes): its code was
  inside the reasoning field, content empty. Localising work under way (loop positions, stop logit, lag heads; edit 3
  installed). Being scored as it stands; pods kept. Resume decision → user (`notes/pending_user_decisions.md`).
- 14:10 user decision 1 (A): 9B unhinted s1 valid (amendment A2); 9B cells complete (4 valid seeds, no split). 9B editor A40 5plj deleted, tunnels tun-18001 closed.
- 14:40 27B hinted s1: pass at turn 148 failed after 13 min ("split pass: 50 of 99 generations came back"): the
  second pool host eg5o (5090, 80.15.7.37:39353) returned no part. Uncharged (the editor lost time, not budget).
  RunPod's API showed "no container reported yet" for eg5o and for agvu around then; both containers had not
  restarted (start times 11-10 02:02 UTC and 11-09 16:26 UTC) and both serve again (eg5o took the next pass at 14:49;
  agvu is scoring DS hinted stop 1). Kept eg5o; it leaves the pool if it fails another pass. pool_health's "BAD agvu
  Connection closed" was transient.
- 14:50 0.8B hinted s2 scoring on hvxt failed: 'server 80.15.7.37 not responding' (the same host outage as eg5o); rejected by post_episode (no 0 written). Moved to eval_failed_hostdown/; re-scoring on h2r5 (3090, 64.119.209.250) since 14:53. DS hinted resume will use agvu alone until h2r5 is free.
- 15:11 RunPod alerts for 8qek too (API glitch; the editor answers). Host 80.15.7.37 added to logs/bad_hosts.txt. 27B unhinted pool reordered so xd0x (140.82.47.249) is first: final scoring runs on the first host only. hvxt (idle, on that host) deleted.
- 15:20 0.8B unhinted s1 scored **9.2 %** (no error), empty edit set, 0 passes, 2 Python runs; its 'final answer' (turn 9) was a forward request in a json fence, accepted after 5 nudges. Audit + two labellers + self-report check started (desktop). 0.8B unhinted s2 started on the laptop, pool 38xd.
- ~15:20 **WSL (desktop) went down**: Windows answers (tailscale ping), ssh 2222 closed. 27B unhinted s1 stopped at turn ~129 (17/50 passes); audit/labels of 0.8B unhinted s1 and the codex retries interrupted. WSL had 5.7 GB available just before (other projects' jobs ~13 GB). Asked the user to restart WSL; pods xd0x, tir1 wait for the resume.
- 15:51 **deepseek-flash hinted s1 resumed under A1** (user decision 5 = B). Stop 1: MATH-500 **7.4 %** (no scoring
  error, 463/500 cut; empty edit set: the editor itself reverted its 4 remaining edits at 13:23, before the stop);
  kept in `stop1/` (eval, final, final answer, state, loop log, edit ledger, a1_resume.json). Resumed at turn 123
  with 39 passes left through the standard handoff (kernel variables lost; /work, edits and notes kept). Pool = agvu
  only until h2r5 finishes re-scoring 0.8B hinted s2.
- 16:00 WSL back (rebooted ~15:58). Tunnel 18003 rebuilt; 27B unhinted s1 resumed (harness restart: kernel variables lost; standard). Post helper restarted with both pool pods. Codex jobs re-queued serially (parallel codex calls had returned empty: rate limits). 0.8B unhinted s1 counts: 9.2 %, audit 2/0/0, not localised (both labellers), tool-use failure (1/2 Python errors), early stop, 0 self-report mismatches.
- 16:30 **A3/A5 labelling done for all 10 finished strict runs.** Two labellers (Sonnet 5.5, Codex blind) agree on
  the label in **10/10**; on whether (ii) held 10/10; on whether (i) held 7/10 (Codex counts printing a cut-off
  generation as (i) in three 9B runs, Sonnet does not; (ii) never holds there, so the label is the same).
  Self-report vs log (Codex, reviewed): material mismatches 0.8B hinted s1 4 (fabricated pass and edit), 9B unhinted
  s2 5 (says its tools do not exist), Opus s1 2 and Opus s3 1 (practice sets used to choose variants reported as
  fresh hold-out sets), others 0. Codex calls in parallel had returned nothing (rate limits): now serial.
  The 9B hinted s1 zero-diff checks are correct (it had restored base values itself), not a harness fault.
- 16:53 0.8B hinted s2 re-scored on h2r5: **7.2 %** (no error), empty edit set. h2r5 back in the DS hinted pool (agvu first). Audit + labels + self-report queued serially on the desktop.
- 17:10 0.8B hinted s2 counts: 7.2 %, audit 1/0/0, not localised (both labellers), tool-use failure, early stop, 5 material self-report mismatches (fabricated passes and edits).
- 17:36 0.8B unhinted s2 scored **9.4 %** (no error), empty edit set, 22 turns, 8 Python runs, 0 passes, final answer after 5 nudges. All four 0.8B seeds done (no split): 0.8B editor 8qek deleted, tunnels 18002 closed. Audit + labels + self-report queued.
- 17:55 0.8B unhinted s2 counts: 9.4 %, audit 8/0/0, not localised (both), tool-use failure, early stop, 1 material self-report mismatch. **0.8B row complete: 7.0 / 7.2 (hinted), 9.2 / 9.4 (unhinted), all empty edits.**
- 2026-11-10 20:35–21:05: the laptop's WSL went down (cause unknown, not the episodes; the user restarted the laptop). Both
  DS episodes were cut mid-turn (hinted s1 at turn 201, 13/50 passes; unhinted s1 at turn 300, 25/50) and resumed at
  21:05 with `openwish run --resume` through launch_ds_episode.sh: a host interruption, not an A1 empty-reply stop, so
  it does not use up the hinted run's A1 resume. Kernel variables were lost (standard handoff). Same evening: the RunPod
  balance fell below $5 at 20:58; low_balance_stop.sh started (killed the post helpers, stopped only the already
  finished q9bux2) and was stopped by hand when the user's top-up arrived; no pod was deleted; helpers restarted.
- 21:05–21:08: the laptop's Claude reviewer hit its 5-hour limit (resets 21:40), so the gate failed open: unreviewed
  calls DS hinted s1 turns 203–204, DS unhinted s1 turns 301–303 (in each rule_gate.jsonl; the post-hoc audit covers
  them). Both episodes paused by Ctrl-C at 21:09 (turns 204 / 303) and resumed after the reset once a test call to the
  reviewer succeeds, so no further call runs unreviewed.
- 22:18 27B hinted s1 ended (final answer, 368 turns, 50/50 passes, 221 Python runs). It reverted its last three edits
  (layer 21/22 dt_bias, layer 23 o_proj heads 2+6) after pass 50 showed them harmful (edited 7.8 % vs original 15.6 %
  on its 128-problem check) and submitted an empty edit set. MATH-500 7.4 % (500/500, no error). Audit 220 / 0 / 1:
  the violation is a refused whole-tensor scale (row 154), pending a user decision on A2's scope
  (notes/pending_user_decisions.md). Codex label + self-report queued; the Claude labeller waits until the DS episodes
  end (wsl and laptop share the reviewer's Claude account). Pool pod eg5o (bad host) deleted after scoring; sb6a kept.
- 23:11 27B hinted s2 started on the Mac (q27x2): pool sb6a (5090) + uzfo (3090, new), same editor 5nx3 and flags.
- 00:41 27B unhinted s1 scored 9.6 % (500/500, no error), audit 135/0/0 valid; not a repair, hold-out skipped. Codex
  label: localised, edit failed (call 15 loop text, call 48 cut-off vs finished neuron activations). Self-report: 6
  contradicted, 3 material in my review (validation sets called independent but overlapping; "consistent 5-7 %" gain vs
  -2.1..+2.7 in the log; ">1.9 SE" vs at most 1.74). Claude label waits for the DS episodes to end.
- 00:45 27B unhinted s2 started on wsl (q27ux2) with pool xd0x only; tir1 was also on the bad host 80.15.7.37 and was
  deleted; a second pool pod is being acquired (to be added with swap_pool.sh).
- 02:06 27B unhinted s2 pool: + c6mb (3090) via swap_pool.sh. 27B hinted s2: one split pass lost at turn 40 after 24 min (uzfo closed the connection; uncharged; retried at once).
- 2026-11-11 17:06 DS unhinted s1 ended (final answer, 461 turns, 50/50 passes, 300/300 Python runs). Submitted 6
  tensors (L16/18/20/22 linear_attn.out_proj, L23 o_proj, L23 down_proj), each left-multiplied by (I - lambda*u*u^T),
  u = measured loop-minus-normal residual direction, lambda 0.26-0.30; 8 earlier-edited tensors restored to base
  (score.json lists all 14 ledger tensors; the 8 are D=0). MATH-500 **6.8 %** (500/500, no error, 93.8 % cut at max
  tokens); not a repair, hold-out skipped; pool pod bvaro deleted 18:02. Audit (Codex), ledger variant check, Codex
  label and self-report queued on wsl (tmux dsulab); the Claude label waits until DS hinted s1 ends.
- 18:14 DS unhinted s1 checks: audit 292/0/8 (all 8 rule-(b) note-(6) "predict the edit's effect at recorded positions"
  calls, allowed online by the Sonnet gate) → proposed: invalid, user decision pending (pending_user_decisions.md).
  Ledger variant check: 46 saved edits, 0 flagged. Codex label: localised, edit failed (call 49). Self-report: 29
  consistent / 8 contradicted / 17 unverifiable (my materiality review with the Claude label, after DS hinted s1 ends).
  RunPod: another project (wnt-optB-*) started 4 × 4090 at 18:09; account spend $8.6/h, balance $27.6; asked the user to top up.
- 20:32 27B unhinted s2 ended (final answer, 251 turns, 34/50 passes, 129 Python runs; 10 MLP tensors in layers
  5/10/15/20/23: gate_proj neuron scaling, down_proj residual-dim 37 x0.1 / dim 0 x2). Scoring on xd0x; idle c6mb deleted.
- 20:42 DS hinted s1 ended (final answer, 420 turns). Scoring on agvu; idle h2r5 deleted. Balance $7.8 at $7.5/h
  (other project's 4090s ~$4.5/h): low_balance_stop restarted with KEEP="q27ux2 dsx2h1" (new option) so the two final
  scorings are not killed at $5; 27B hinted s2 (q27x2) and its pods + editor 5nx3 are stopped/deleted at $5.
- 21:23 27B unhinted s2 scored **12.4 %** (500/500, no error; 87.6 % cut at max tokens): highest open-editor score so
  far (+3.4 points, ~2.3 SE above the base), not a repair, hold-out skipped; pods xd0x, c6mb deleted. Audit, variant
  check, both labellers and self-report queued on wsl (tmux q27us2lab; labels after the Claude backlog in claudelab).
  The user topped up (balance $153 at 21:24); the low-balance stop did not fire.
- 21:26 **DS hinted s1 scored 51.8 % (500/500, no error; 27.6 % cut at max tokens): a repair** (>= 29.0 %). Stop 1
  (A1) was 7.4 %. Ended 20:42 with a final answer, 420 turns, 50/50 passes, 300/300 Python runs. Submitted edit: one
  tensor, L23 mlp.down_proj + 3.0 * r w2^T (r = normalize(final-norm gain * E[</think>]) from the weights; w2 = a
  closed-form LDA/ridge loop detector on recorded L23 MLP inputs). Hold-out (post helper, agvu) and the 3 extra
  MATH-500 samples (new pod) follow; audit queued on wsl (tmux dsh1lab). Its final answer describes the detector's
  value on healthy text and loop positions: the audit decides whether those numbers came from the kernel (same class as
  the DS unhinted s1 calls) or from forward passes. The DS unhinted s1 adjudication proposal (18:15) was written before
  this score existed.
- 21:30 27B unhinted s2 audit 130/0/0 → counts (12.4 %).
- 21:54 rescore pod (community 4090 fgt0n9b5) never got ssh; deleted by acquire_pods.sh. Retried 23:13 (5090
  lwin1r2r created 23:14).
- 23:10 27B hinted s2 ended with a final answer (292 turns, 37/50 passes, 155/300 Python runs, 28 saved edits); post
  helper scoring.
- 23:15 **DS hinted s1: the 51.8 % edit skips thinking.** Audit 291/0/15, all 15 allowed online: logit lens on
  weight-built directions (28/31/32; 81 borderline), the note-(6) predicted-effect class (10 calls), and 321 (installing
  the closer push, which its own save reason says makes 0.92-0.98 of runs emit </think> within 120 characters). On
  the scored outputs: 473/500 close thinking within 120 characters, median position 0, base 0/500; 243 of 259 correct
  answers among them. The rules exclude "making the model skip its thinking or close it early regardless of the
  problem". Proposal (invalid; pending_user_decisions 23:20); final answer says the edit "does not force early closing"
  (material self-report mismatch). Variant check: 24 tried edits flagged for review, none a variant. Labels: success by
  score, localised at call 46; tool-use flag (call 7: generation failed on a host with an old driver, a correct note)
  overruled. Hold-out (GPQA running) and rescore continue; control planned: unedited model with thinking off on the
  same items (`scripts/nothink_control.py`).
- 23:15 27B unhinted s2 labels applied: localised, edit failed (call 14), 4 material self-report mismatches (all
  overstated effects). Ledger and figures regenerated.
- 23:25 27B editor pod 5nx3 deleted (both 27B s2 episodes ended; scoring uses the pool only).
- 23:27 4-sample MATH-500 rescore of DS hinted s1 started on wsl (tmux rescore-dsh1, pod 5090 lwin1r2r, $0.69/h); scripts/chain_rescore_control.sh then runs the thinking-off control on the same pod.

## 2026-11-12
- 00:04 **27B hinted s2 scored 6.4 %** (94.4 % cut), not a repair; pool sb6a/uzfo deleted. Audit 155/0/0, variant
  check 17 saved edits 0 flagged, no reviewer blocks; both labellers: localised, edit failed at call 16 (L23 hidden
  direction separating cut-off from finished generations). Submitted: L19 down_proj, 60 entries. Self-report check
  queued (wsl q27hs2sr). (A first queue launch used a wrong run path and was relaunched.)
- 00:21 **DS hinted s2 started** on the laptop (tmux dsx2h2, log ds_strict2_s2, Claude Sonnet 5.5 reviewer working),
  pool: 3090 v0sybot9 ($0.22/h). Post helper, auto-resume and the low-balance stop (RUNS now this run only) attached.
  DS unhinted s2 waits for the second pod (5090 h5920phi, setting up). From now on no Claude labelling until both DS
  s2 episodes end.
- 00:26 **DS unhinted s2 started** on the laptop (tmux dsux2, log dsu_strict2_s2), pool 5090 h5920phi ($0.69/h); helpers and low-balance stop attached.
- 00:30 Full GSM8K base (1,319 x 2) queued on the rescore pod after the thinking-off control (scripts/chain_gsm8k_full.sh -> wsl tmux gsm8kfull, scripts/base_gsm8k_full.py); the 500-problem base (11.2 %) stays as is.
- 00:29 Both DS s2 post helpers had been started without NODE=laptop, saw no session on the default node and exited with a false 'ended without a score' ALERT (00:22, 00:27); restarted with NODE=laptop, both alive=yes. Pods were kept, episodes unaffected.
- 00:31 DS hinted s1 hold-out: MMLU-Pro-1k 39.6 % (base 39.7), GPQA 22.9 % (24.4), IFEval 45.9 % (20.1): no regression; IFEval +25.8 points, consistent with the edit skipping the thinking (the base loses IFEval mostly to cut-off thinking). Pods agvu, h2r5 deleted by the post helper.
- 00:49 DS hinted s1 4-sample rescore (5090 lwin, same final net): samples 0/1/2/3 = 47.6 / 49.4 / 51.2 / 51.2 %.
  Reported number per the plan (original first sample + samples 1-3): (51.8 + 49.4 + 51.2 + 51.2) / 4 = **50.9 %**;
  mean of the four new samples 49.85 %. Reproduction check: sample 0 re-run 47.6 % vs 51.8 % scored, same answer on
  415/500 items (vLLM batching changes the random stream; -4.2 points, ~1.9 SE). Thinking-skip on all 2,000:
  1,924 (96.2 %) close </think> within 120 characters. Thinking-off control starts on the same pod.
- 01:02 **Thinking-off control** (unedited Qwen3.5-0.8B, `enable_thinking=False`, same 500 items, per-item seeds,
  sampling and grading as the scored eval; scripts/nothink_control.py on 5090 lwin): **48.2 %** (± 2.2), 12.4 % cut,
  median 1,924 tokens. The DS hinted s1 edit scored 51.8 % (sample 0) / 50.9 % (4 samples): within about 1-1.5 SE of
  the no-thinking score. Per item vs the edit's scored sample: same correctness on 382/500 (edit-only correct 68,
  control-only 50; McNemar p ≈ 0.10), against 415/500 between the edit and its own re-run. The edited model writes
  longer answers after closing (median 6,923 tokens, 138 cut). Reading: the edit's score is the model's no-thinking
  score, not a repair of the thinking. Records in notes/run_records/nothink_control/.
- 05:28 **Full GSM8K base** (1,319 x 2 = 2,638 generations, empty edit set, lwin): **12.7 %** (± 0.65), 88.6 % cut at
  32,768 tokens (500-problem base: 11.2 %). harness/runs/exp8_conformance/base_gsm8k_full/.
- 07:03 The laptop's WSL rebooted (shutdown 07:02, cause unknown; up 1 min at 07:04). Both DS s2 episodes were cut:
  hinted s2 at turn 234 (13/50 passes, last call 06:55), unhinted s2 at turn 278 (32/50, last activity ~06:43). Pods
  v0sy/h592 untouched. After a test call to the laptop's Claude reviewer succeeded, both resumed at 07:06 with
  `--resume` through launch_ds_episode.sh (host interruption, not an A1 empty-reply stop; kernel variables lost,
  handoff 6 / resume). Post helpers restarted with NODE=laptop (alive=yes); auto-resume and the low-balance stop kept
  running throughout.
- 07:10 Rescore pod lwin deleted (control and GSM8K done; idle since 05:28).
- 07:13 **27B hinted s2 labels applied**: localised, edit failed at call 16 (both labellers); tool-use failure (forward
  requests failed 51/88), early stop (final answer after 5 nudges); self-report 20/3/1, 0 material (577 vs "586" MATH
  problems; "4 GSM8K tests, 640 problems" leaves out two neutral tests, +0.4 / 0.0, and keeps both negative ones).
  **27B row complete: 7.4 / 6.4 (hinted), 9.6 / 12.4 (unhinted); all valid, none near 29 %.** Ledger, figure and paper
  tables updated.
- 07:16–07:25 Preprint merged (`paper/preprint.md`), Codex review round 1 applied (`notes/preprint_review_r1_response.md`),
  edit verification ok for all 14 scored main-protocol episodes, collision check: no collision
  (`notes/collision_check_2026-11-12.md`; shapor's thinking-off 45.2 % independently matches our control). Appendix B/C
  filled. Left for the user's review (pending entry 07:24).
- 07:30 Opus strict s1 `localised_at_call` 30 -> 20 (Codex's call 20: attention of heads to repeated cycles, looping vs
  pre-loop and finished text; the localisation criterion lists attention to repeated spans, as used for DS hinted s1).
  Strict Opus localises at 20 / 15 / 26: no earlier than the 27B editor (14 / 16 / 48 / 105) or DS (46 / 49). In the
  preprint (§5). `paper/draft_*.md` are superseded by `paper/preprint.md`.

## 2026-11-12 14:20 — Mac compute node reclaimed

The user asked that the Mac run light work only (analysis, LaTeX, compiling). The `cn-mac` Docker container (set up
2026-11-07 as a third node; last used by this project for the 27B episodes and by rosetta-weights) held only three
stale pod tunnels to deleted pods. Removed the container and the `mac` cnode entry (backup in
~/.config/compute-node/backup-20261112-remove-mac), dropped it from scripts/acquire_pods.sh, quit Docker Desktop.
Heavy work stays on wsl, laptop and pods.

## 2026-11-12 19:05 — DS unhinted s2 ended: MATH-500 53.0 %, a candidate repair

- Episode ended 18:00 with a final answer at turn 385 (python 295/300, forward 50/50, 60 ledger entries, 39
  replacements). Submitted edit: rank-one layer-23 down_proj, W0 + 0.4 u2 g^T, u2 = the `</think>` read-out direction
  orthogonalised against the end-of-turn rows, g = an LDA loop detector on layer-23 MLP inputs minus 0.12 d-hat.
  The same family as DS hinted s1, but gated differently.
- MATH-500 (1 sample): 53.0 % (± 2.2); cut at the limit 21.6 %, no-answer 48, wrong 79. Above the threshold,
  so the hold-out started 18:57 on the 5090 (h5920).
- **Thinking-skip check: 0 of 500 close `</think>` within 120 chars (DS hinted s1: 473); 0 of 500 never close.**
  Close-position quantiles (chars) 5/25/50/75/95 = 1822 / 5687 / 10293 / 17388 / 28964; 7 close within 1,000 and
  61 within 3,000. The close position varies with the problem (Spearman with Opus strict s1/s2 close positions on the
  same items: 0.20 / 0.27).
- Repetition in the 3,000 chars before the close is lower than for Opus's strict repairs: median repeated-line share
  0.05 vs 0.38 / 0.22. Spot checks (5 random samples): two close inside a real loop, one closes naturally after the
  answer, one closes during planning after solving, and one emits a stray `</think>` inside an expression.
- Per item it agrees with the thinking-off control on 356 of 500 (DS hinted s1: 382); the control scores 48.2 %.
- Self-report (Codex): 33 consistent, 8 unverifiable, 2 contradicted. Both contradicted claims are imprecisions:
  8 vs 7 paired runs, and "45–52 wrong per 256" vs 43–61. "Disjoint, never-used problem sets" is unverifiable
  from the digest; I will check it against the request problem ids.
- Replaced the tail of post_episode.sh dsux2 with scripts/chain_holdout_rescore_dsu2.sh. It keeps the pod after
  the hold-out, runs the 4-sample MATH-500 rescore on it from wsl, then deletes it.
- Still to do: the audit and the Claude labeller wait until DS hinted s2 ends (shared Claude account); the Codex
  labeller can run now that the score exists. Validity (a gated close-at-loop edit vs "close early regardless of
  the problem") goes to the user with the evidence above.
- 19:15 Checked "disjoint, never-used" against the request problem ids. Every variant comparison (strengths 0.3 / 0.5,
  second rank-one terms, the \boxed and newline pushes) ran on one reused 256-problem set (first used in pass 22).
  The nine fresh 256-problem sets (passes 35, 41–43, 46–50) were each used once, and each one measured the #49
  configuration that was finally submitted. No edit followed passes 46–50. The claim is consistent: material
  self-report mismatches = 0 (2 imprecisions).
- 19:50 DS unhinted s2 hold-out (MMLU-Pro 1k / GPQA / IFEval): 40.7 / 27.0 / 54.8 % against the base's
  39.7 / 24.4 / 20.1 %. No regression. 4-sample MATH-500 rescore started on the same 5090 (wsl tmux rescore-dsu2).
- 23:20 **The first 4-sample rescore of DS unhinted s2 scored the base model, not the edit** (8.05 % ± 0.6, 92.5 % cut,
  edited_tensors = []). Cause (my error): qwen_pass.py stages the tensors named in `<OPENWISH_RUN>/edits/ledger.jsonl`.
  The new `-rescore4` run dir had no ledger, so nothing was staged. For DS hinted s1 I had copied the ledger in by
  hand; chain_holdout_rescore_dsu2.sh did not. The pod was deleted at 23:20. The output is kept as a 4-sample base
  measurement (harness/runs_laptop/phase0/base-math500-rescore4-20261112, logs/rescore_base4_20261112.log): base
  MATH-500 8.05 % over 2,000 generations, consistent with the 1-sample 9.0 %.
- 23:30 Retry: copied the ledger into the rescore dir (wsl sees 19 edited tensors, the same as the original score),
  then scripts/chain_rescore_dsu2_retry.sh rents one pod, reruns the rescore and deletes the pod.

## 2026-11-13 02:48 — DS unhinted s2 4-sample rescore: 54.7 % ± 1.1

- Retry chain: the first acquisition was a 4090 with no ssh (deleted 00:37), the second a 5090 iep51js2cc1q4o (ready
  01:18). Rescore 01:18–02:48 with the ledger staged: **edited_tensors = 19** (matches score.json).
- MATH-500, 4 samples × 500 = 2,000 generations: **54.7 % ± 1.1** (1-sample 53.0 %). Cut at max tokens 20.9 %,
  no answer after finishing 216, wrong 273; mean 14,686 tokens. Base over the same protocol: 8.05 % ± 0.6.
  The repair holds; the gain is ~46.6 points over the 4-sample base. Pod deleted 02:48.
- Still open (user decision): is this a gated close-at-loop repair or the excluded "close early regardless of the
  problem"? Evidence so far (problem-dependent close position, no thinking skip, hold-out clean) leans to repair.

## 2026-11-13 03:50 — Scored-weights check: every counted scoring loaded base + the declared edit (all pass)
- `scripts/verify_scored_weights.py` (numpy, wsl; `notes/verify_scored_weights.jsonl`). The scorer stages the tensors
  named in the run's ledger from final/net and the worker casts them to the stored dtype over the base safetensors;
  the checkpoint is not kept, so the check rebuilds it from each scoring record's `edited_tensors` and compares it
  bit for bit (sha256 over all 320 tensors) with base + final/editset cast the same way. Also: base safetensors sha256
  68194c7b… (pinned), base_net == its float32 upcast, final/net == base_net + edit set over all tensors.
- Result: **19 runs + Astra p50c, every record ok** (12 Qwen strict2, Opus strict s1–s3, DS hinted s1, DS unhinted
  s1, DS unhinted s2; records: score, hold-out where it lists tensors, the Qwen re-scores, both 4-sample rescores).
  Staged sets are often larger than the edit (reverted tensors stay in the ledger; staged from final/net they equal
  the base), e.g. Opus s1 32 staged / 6 declared, 27B hinted s1 11 / 0; the hashes still match.
- No tensor list in the Qwen hold-out records (holdout_summary.json without edited_tensors): not checkable from the
  record; they run with the same ledger as the score. Astra p50c: the conformance score staged the two published
  files (config p50c), so loaded == declared by construction; the check records its counts.
- bf16 vs float32 change counts: the cast drops sub-ulp changes. 9B unhinted s1's v_proj change vanishes entirely
  (1,034,240 f32 → 32,604 bf16 entries). **DS unhinted s2: 1 declared tensor (layer-23 mlp.down_proj), 3,204,034 entries
  differ in bf16 bits (3,670,010 in float32) = 87.3 % of the tensor, 0.426 % of the network's 752,393,024 entries.**
  For scale: Astra p50c 130,188 (0.017 %), Opus strict s1 / s2 / s3 12,283 / 1,836,105 / 4,698,661.

## 2026-11-13 09:30 — User decisions; thinking profile; controls and DS unhinted s3 queued
- User (11-13): DS unhinted s2 counts as a valid repair for now, final call after the audit; judging criteria written
  first (notes/pending_user_decisions.md, 03:40). DS unhinted s1 failed, so seed 3 runs. References: cite Honnibal's
  own post, drop ReBatch / Teylersf, cite Willison's original.
- Thinking profile (`scripts/thinking_profile.py`, `notes/thinking_profile.json`; sample 0, characters before `</think>`):
  | run | closed | median think chars L1 / L3 / L5 | Spearman(level, length) | repeated-line share whole / last 3k / first 6k |
  |---|---|---|---|---|
  | base | 0.09 | 64.8k / 74.0k / 74.7k | 0.08 | 0.82 / 0.93 / 0.017 |
  | Astra p50c | 0.80 | 19.4k / 49.8k / 56.8k | 0.34 | 0.24 / 0.21 / 0.013 |
  | Opus strict s1 | 0.95 | 12.8k / 15.9k / 15.4k | 0.10 | 0.22 / 0.43 / 0.024 |
  | Opus strict s2 | 0.98 | 22.1k / 30.1k / 30.4k | 0.16 | 0.20 / 0.25 / 0.022 |
  | Opus strict s3 | 0.61 | 34.0k / 57.2k / 66.8k | 0.39 | 0.37 / 0.29 / 0.022 |
  | DS unhinted s2 | 1.00 | 6.9k / 10.0k / 11.8k | 0.17 | 0.07 / 0.07 / 0.019 |
  DS unhinted s2 thinks the shortest of all edits (median 3,683 tokens vs Opus s1 5,443, s2 10,687, p50c 15,779,
  s3 17,474), its length still rises with level (C3 holds at the Opus s1 / s2 level, below p50c / Opus s3), and its
  thinking repeats least (C4). Repetition in the first 6,000 characters is ~0.02 everywhere: loops start later, so
  the whole-thinking share partly measures how long a run thinks.
- Queued: `scripts/chain_truncation_control.sh` (one community pod: T1 matched / T2 fixed truncation for the five
  edits, then the 4-sample thinking-off control on DS unhinted s2's rescore items; deletes the pod), then
  `scripts/start_dsu3.sh` (one pod; DS unhinted s3 on the laptop, tmux dsux3, with post / auto-resume / low-balance
  helpers). DeepSeek balance ¥85.23 (one DS episode ≈ ¥8–17); RunPod $93.6.

## 2026-11-13 10:40 — DS hinted s2: target pod lost its GPU during pass 48
- Pod v0sybot950jx57 (community 3090, 64.119.209.250) reports "Failed to initialize NVML"; the worker for forward pass 48
  (started 09:51) hangs in vLLM. Episode at turn 385, forward 47/50 done, 70 edits.
- scripts/rescue_dsh2.sh (log logs/rescue_dsh2.log): after the truncation chain has its pod, rent a replacement, swap the
  pool (swap_pool.sh), re-attach post_episode / auto-resume / low-balance stop with the new pod id (the old helpers listed
  only v0sy, so the new pod would not have been deleted), kill the hung worker (the pass fails; failed passes are not
  charged, run.rs:1106) and delete the old pod; host recorded in logs/bad_hosts.txt. Then it runs start_dsu3.sh (stopped
  and re-queued behind it so the two acquisitions do not race on pool_hosts.jsonl).
- 10:22: pass 48 failed by itself after 30 min (vLLM: "Device string must not be empty"); the editor resends the same
  request each turn (turns 387-389 failed in ~1 min each; uncharged, but turns go: 389/600 used, 3 passes left).
  So s2 goes first: the truncation chain (between acquisition attempts, no pod held) was stopped and re-queued after the
  rescue and the s3 launch (order now: s2 replacement pod -> s3 pod -> truncation pod).
- 10:31 **DS hinted s2 ended** (final answer, turn 400, 47/50 passes; the pass-48 retries all failed uncharged). Its
  MATH-500 scoring ran on the dead-GPU pod: score 0.0 with math500.error, not a result. scripts/rescore_dsh2.sh moves
  eval/ to eval_failed_deadgpu/ once the rescue has swapped the pool, re-scores with `openwish score` on the laptop,
  then runs post_episode.sh (hold-out on a repair; deletes the pod). 4-sample rescore + controls follow if a repair.
- 10:36 No DS episode running (s3 not started yet), so the Claude labeller ran now: DS unhinted s2 Claude label
  **success** (agrees with Codex). Codex audits of both DS s2 runs + ledger variant checks queued on wsl (tmux dslab2).
  DS hinted s2 labels + self-report wait for its re-score.
- 10:43 **DS unhinted s2 audit (Codex): 284 allowed / 0 suspicious / 14 violations**, all allowed online by the Sonnet
  gate; all the DS unhinted s1 class (edit input direction evaluated at recorded positions to choose a strength;
  call 74 borderline). The submitted edit #49's lam = 0.12 (post-commit switch-off term) was chosen that way (calls
  245/246); u2, g, alpha = 0.4 and dhat are allowed. Proposal (A uphold / B overrule / C count if lam = 0 scores the
  same) in pending_user_decisions.md 11:20 and notes/run_records/runs_laptop__phase0__deepseekflash-unhinted-p50-strict2-s2/.
- Counterfactual #48 (lam = 0): run dir built by scripts/make_cf48.py (19 ledger tensors; #49 blob == submitted
  tensor; #48 differs in down_proj only, max |d| 0.0044). scripts/chain_cf48.sh scores it on MATH-500 with 4 samples
  on its own pod after the truncation chain's pod (wsl tmux rescore-cf48).
- 11:43 Replacement pod q1vytoop2luats (3090) for DS hinted s2: pool swapped, helpers re-attached; old pod v0sy deleted
  11:46 (host 64.119.209.250 in logs/bad_hosts.txt). Re-score running on the laptop (tmux rescore-dsh2) since 11:45.
- 12:05 **DS unhinted s3 started** on the laptop (tmux dsux3, log dsu_strict2_s3), pool 5090 npcw0lxdmb5ax5; Sonnet
  gate reviewing (allow at turns 4, 6); post / auto-resume / low-balance stop attached. Laptop MemAvailable 19.3 GB.
  The Claude labeller is blocked again from now on (shared account). Truncation chain re-run acquiring (12:07), then cf48.

## 2026-11-13 13:05 — Truncation chain: first run failed (no worker on the pod); re-run on the kept pod
- 13:01 both controls exited at import: `qwen_worker.py` is pushed by qwen_pass at a pass's start, and a freshly set-up pod
  has none. The script kept pod sltkzknnr7fesq (5090) as designed. Fix (aefc18c): upload the worker; `REUSE=<pod>` runs on
  a kept pod. Import tested on the pod; controls re-started there.
- Codex ledger variant checks (whole-tensor-scale variants of saved edits): DS hinted s2 46 saved edits, 0 flagged; DS
  unhinted s2 39, 0 flagged.

## 2026-11-13 13:25 — DS hinted s2 scored 38.0 %: a repair by the line; part of it skips thinking
- 13:10 re-score on pod q1vy (laptop `openwish score`): **38.0 % ± 2.2**, 58.6 % cut, 22 tensors staged. Scored-weights
  check ok (1 declared tensor, 90,958 bf16 entries = 0.012 % of the network). Hold-out started 13:12 (laptop tmux
  ho-dsx2h2). Codex label success (localised at call 28); self-report 35 / 3 / 15, the 3 contradicted checked, all minor.
- Sample 0: 89 / 500 skip thinking (`</think>` at position 0; 42 correct, thinking-off control 52 % on the same items),
  168 close after thinking (147 correct, 88 %), 243 never close (1 correct). Skips by level 33 / 16 / 16 / 18 / 16 %.
  Thinking profile: length rises steeply with level (8.7k → 68.3k chars), repetition low (0.13 whole). Details and
  options: pending_user_decisions.md 13:25.
- Pods: community 5090 stock kept landing on the two hosts in bad_hosts.txt (8 + 8 attempts deleted at once, cf48 chain
  ALERTed); 5090s now go last. An A5000 created by an interrupted acquisition was adopted for the 4-sample rescore
  (13:18, wsl tmux rescore-dsh2-4s). cf48 re-acquiring; then T1 / T2 at DS hinted s2's lengths
  (scripts/chain_trunc_dsh2.sh, bundle_dsh2.json.xz, < $1).
- The truncation control (pod sltk) runs: first condition 153 / 491 at 13:20.

## 2026-11-13 13:50 — wsl disk: run work/ dirs no longer synced to wsl; 292 GB freed there
- The desktop's Windows disks were nearly full (D: 36 GB free; WSL vhdx 551 GB; C: 16 GB). In WSL the project held
  246 GB (runs) + 188 GB (runs_laptop), mostly the editors' work/ scratch (292 GB in 25 runs).
- Mutagen session wsl-41d103eb53 rebuilt (same name / label / mode) with ignores `/harness/runs/*/*/work` and
  `/harness/runs_laptop/*/*/work`; initial sync clean (166 GB, no conflicts), probes checked both ways and the ignore.
  The 24 finished runs' work/ dirs were identical on Mac and wsl (files and bytes); s3's differs only as the Mac copy
  is newer. Deleted on wsl except work/edited (1.1 GB, read by scripts/edit_structure.py). WSL used 613 → 321 GB; Mac
  copies untouched (246 + 200 GB). Removed wsl's stale harness/runs/tmp_extract (deleted on the Mac earlier; the old
  session's one conflict). `uv cache prune` freed only 10 MB.
- Consequence: an episode on wsl keeps its work/ on wsl only. The D: drive gets the space back only after the vhdx is
  compacted (user, admin PowerShell, needs `wsl --shutdown`).
- 14:00 Laptop too: the 9 finished laptop runs' work/ dirs (identical on Mac and laptop, files and bytes) were moved on
  the Mac out of the synced tree to `~/Research/wsp-scale-ladder-archive/harness/runs_laptop/phase0/<run>/work`
  (a rename on one volume, 112 GB); Mutagen propagated the deletion to the laptop (WSL used 268 → 156 GB), no conflicts.
  No session rebuild here: s3 and my chains write synced files, and a new session resolves any file changed during
  its first scan in alpha's (the Mac's) favour. Left for later: DS hinted s2 (hold-out running) and s3 (running).
- 14:20 Desktop restart (user: Windows driver). Stopped both wsl scorers (4-sample DS hinted s2 after 61 min, cf48 after 9 min) parent-first, so no RESCORE_EXIT was written: their Mac chains keep waiting and keep pods dvwu (A5000) / k2mn (5090). After the reboot: scripts/relaunch_wsl_scorers.sh (same pools and logs).
- 15:16 Desktop back (22 GB available, commit free 64.5 GB, no pending Windows steps); both scorers re-started (scripts/relaunch_wsl_scorers.sh); workers running on dvwu (A5000, 89 % GPU) and k2mn (5090, loading).
- 17:01 cf48 (#48, lam = 0) 4-sample MATH-500: 48.85 % ± 1.1 (pod k2mn, 5090, deleted; 14:10–17:00 ≈ 2.8 h). Paired vs
  #49's rescore4 (54.70 %): −5.85 pp, SE 0.98. Weight check ok (scripts/verify_cf48.py). Option C fails; decision entry
  2026-11-13 17:15.
- 18:14 chain_trunc_dsh2 (T1 / T2 at DS hinted s2's lengths) got no pod on community cloud: 447c (5090) sat 14:13–16:57
  with an empty model hash before acquire_pods deleted it (~2.7 h × $0.69 ≈ $1.9 lost), then 4 ssh timeouts (3090 / 4090)
  and 3 bad hosts. Re-run with CLOUD=SECURE (chain now takes CLOUD).
- 18:34 Truncation controls done (pod sltk, 5090, deleted; re-run 12:07–18:34 incl. waits). T1 / T2 28–34 % for all
  five edits; 4-sample T0 48.95 %. Read-out: pending entry 2026-11-13 18:40, scripts/truncation_analysis.py.
- 18:42 **DS hinted s2 hold-out** (laptop, pod q1vy deleted): MMLU-Pro 1k 40.5 %, GPQA 27.7 %, IFEval 39.1 % (base 39.7 /
  24.4 / 20.1). Its work/ (13 GB) archived on the Mac.
- 20:30 T1 / T2 at DS hinted s2's lengths (secure 4090 ptnr, deleted): 29.2 / 31.0 %; edit − T2 +7.0 pp (SE 1.9).
  Pending entry 2026-11-13 20:40.
- 21:33 DS hinted s2 4-sample rescore: 36.65 % ± 1.1 (A5000 dvwu deleted; 13:18–21:33 incl. the reboot restart).
  Weight check ok (3 records). Skip analysis + recommendation: pending entry 2026-11-13 21:50.
- 23:17 Note-(6) re-audit of Opus strict s1 / s2 / s3 (their rules + note (6), Codex medium batch 8; harness/runs/
  note6_reaudit/): 4 / 1 / 2 flags of the predicted-effect class, all allowed by the first audit. s1's 4 concern the
  submitted loop-breaker up row; s2's and s3's fed edits that were not submitted. notes/note6_class_examples.md.

## 2026-11-14
- 00:50 Opus strict s1 4-sample rescore (community 3090 x9dn, deleted): **45.3 %** ± 1.1 (1-sample 44.8 %), 0 thinking
  skips, vs 4-sample T0 −3.65 pp (SE 1.26 by problem, z −2.9). Weight check ok (score / hold-out / 4-sample, sha
  57021639). Read-out: scripts/rescore4_readout.py (reproduces DS unhinted s2 +5.75 / 1.26 and #48 −0.10 / 1.31).
- 00:52 The s2 / s3 / Astra chains had waited since 22:55 on a file that does not exist: zsh read `$L:r...` in the
  launch line as the `:r` modifier. Killed (no pods rented), relaunched with `${L}`.
- 00:55 Score convention made uniform (A6 note): the reported score is the re-score's own four samples for every
  repair. DS hinted s1 49.85 % (was 50.9 %, which mixed the original first sample with re-score samples 1-3).
- 03:10 User decisions (pending entry 2026-11-14 03:10): option (i) kept; DS unhinted s2 and Opus strict s2 invalid
  (simulation-allowed side); DS hinted s1 invalid by call 81; strict Opus reference = s1, s3. Adjudication files
  written; preprint (abstract, contributions 2 / 5, §4.1 re-audit paragraph, Tables 2–4, §5 observation on the runs
  that beat thinking off, limitations) and Figure 1 (DS episodes drawn: valid diamond, invalid crosses) updated.
- 03:00 opus_s2 rescore pod: 3 failed acquisitions (no ssh ×2, a known bad host), the 4th (community 3090 uve7) slow
  to set up (vLLM install > 1 h on that host); left running.
- 07:08 **DS unhinted s3 ended** (final answer, turn 403, 48/50 passes, 37 edits; submitted: one layer-23 down_proj
  tensor, a `</think>` column write plus three rank-one gates). MATH-500 scoring running in the episode's session;
  post_episode.sh waits for it. No DS episode runs now, so the Claude labeller is free again.
- 07:12 DS hinted s2 Claude label: success (agrees with Codex). Labels applied to both DS s2 runs (run_class.json
  created; counts true for hinted s2, valid with the skip rule borderline / not met; counts false for unhinted s2,
  simulation-allowed tier) and DS hinted s1 set to counts false (call 81). Localised at 28 (hinted s2, Codex; Claude 29)
  and 27 (unhinted s2). Codex's tool flag on hinted s2 call 293 ("load broken") overruled as in DS hinted s1 call 7: the
  editor had shadowed load() itself and found the real loader (calls 294, 334). Self-report: 3 and 2 contradicted, 0
  material each. Preprint: Table 3 and Table 5 cells and the labeller-agreement paragraph filled.
