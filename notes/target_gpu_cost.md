# Target-GPU cost evaluation (exp8 forward passes, Qwen3.5-0.8B)

2026-11-08. What the target-model forward passes cost, what limits their speed, and which ways of saving work.
This project only: other projects have different models and bottlenecks. Raw data: `scripts/pass_throughput.py`
(per-pass throughput from the runs), `scripts/bench_target_gpu.sh` (one replayed pass on a fresh pod),
`scripts/win_mem_sampler.sh` + `scripts/mem_by_project.py` (local GPU memory, attributed by project),
RunPod billing (`runpodctl billing pods --bucket-size hour --start-time … --end-time …`).

## 1. Where the money goes
- RunPod bills each pod separately, GPU + container disk, from creation to deletion. Our compute log matched the
  bill within 2 % (deleted pods to 11-08 01:51: billed $82.63, logged $81.03).
- One Opus episode (50 passes) kept its target pod ~26 h, **~80 % of it in forward passes** (s1/s2: 20.8/21.2 h,
  of which generation 18.6/19.7 h; failed passes 1.4 h; Python 0.8 h; the editor's own thinking 3.2 h).
  Generation passes dominate; teacher-forced passes take 1–8 min.
- Phase 0 to 11-08 01:51 (~$83): about $69 went into the two Opus runs later invalidated by the audit, $9.4 into runs
  stopped when the rules changed, ~$3 into passes that failed (pod disk full), $1.3 a stray pod, ~$5 the price
  premium of PRO 6000 cards over 5090s with the same speed. The plan's real error is larger: it assumed 1–2 GPU hours
  per episode; the measured figure is ~21 busy GPU hours (10×), partly offset by $1.19/h cards instead of $3/h.

## 2. What limits speed: memory bandwidth
- Every GPU measured runs generation at **~2.3–2.7k tok/s per TB/s of memory bandwidth** (5090 and PRO 6000 at
  1.79 TB/s: 4.1–5.6k tok/s; RTX 6000 Ada 0.96 TB/s: 2.6k; 5070 Ti 0.90 TB/s: ~2.3k). Memory size does not matter:
  the 96 GB PRO 6000 is no faster than the 32 GB 5090.
- Reason: answers run to 20–32k tokens; each generated token reads its sequence's full-attention KV cache
  (6 layers × 2 KV heads × 256 dims × K,V × 2 B = 12 KB per context token → ~200 MB per token at 15–20k context).
- So the price of a pass is ∝ hourly price / bandwidth. Secure cloud, $ per (TB/s · h):

| GPU | $/h | TB/s | $/(TB/s·h) | vs 5090 |
|---|---|---|---|---|
| RTX A5000 | 0.27 | 0.77 | 0.35 | 0.53× |
| RTX 3090 | 0.50 | 0.94 | 0.53 | 0.80× |
| RTX 5090 | 1.19 | 1.79 | 0.66 | 1 |
| H100 NVL | 3.19 | 3.9 | 0.82 | 1.2× |
| RTX 4090 | 0.89 | 1.01 | 0.88 | 1.3× |
| A100 SXM 80GB | 1.79 | 2.04 | 0.88 | 1.3× |
| L40S | 1.09 | 0.86 | 1.26 | 1.9× |
| RTX PRO 6000 | 2.49 | 1.79 | 1.39 | 2.1× |

## 3. Ways to save, judged
- **Several cheap cards per episode, each pass split across them (adopted).** `qwen_pass.py` takes
  `{"hosts": [...]}` and gives each host a contiguous range of the problems (weight = bandwidth), every model of the
  pass on that host, so "both" comparisons stay paired on one GPU. Three A5000s ≈ 2.3 TB/s for $0.81/h: faster than
  one 5090 at 0.53× its price per pass. Extra cost: one vLLM start per host per model per pass (~1–2 min), idle
  cards during the editor's turns (cheap cards make this small). Failed shares rerun on another host.
  Checked against a single host (`scripts/test_split_pass.py`): same records, order, summary format and activation
  shapes. Texts are not byte-identical (batch size changes the kernels), as separate passes already are not.
- **One big card serving several episodes: rejected.** The pods are already ~80 % busy and big cards cost the same
  or more per TB/s; sharing would save ≤ 20 %, slow each episode, and needs isolation work (the per-pod orphan cleanup
  would end another episode's worker).
- **Creating pods per pass / stopping them between passes: rejected.** Gaps between passes are a few minutes; setup
  takes ≥ 8–10 min and secure stock is often empty.
- **Local RTX 5070 Ti: only with memory headroom.** Free, 0.9 TB/s. A local generation costs ~13–15 GB of Windows
  commit for GPU memory (VRAM counts in commit) and grows the WSL VM by its page cache. Beside two episodes and another
  project's jobs it drove Windows' free RAM to 0.5 GB with page-ins up to 67k/s (11-08): that is paging, slow and
  risky. Now used as one host of a desktop episode's pool only when WSL has ≥ 6 GB available and Windows ≥ 24 GB commit
  free, one local share at a time (lock), stopped below 2.5 GB / 6 GB (its share then reruns on a pod).
- **Community cloud: not used yet.** 40–60 % below secure prices (A5000 $0.16, 3090 $0.22, 5090 $0.69) but pods can
  vanish; the split's retry would cover a lost host. Worth a trial if secure stock stays short.
- **Not allowed (protocol):** FP8 KV cache, shorter max_tokens, fewer samples per pass. They would change the
  target model's measured behaviour.

## 4. Benchmarks
The same workload everywhere: the first pass of an Opus episode, 512 base-model answers (~15.5M generated tokens,
~91 % cut at 32,768) plus activations at all layers. Time includes vLLM start and the activation pass.

| GPU (secure) | source | minutes | ktok/s | ktok/s per TB/s | $ per pass | vs 5090 |
|---|---|---|---|---|---|---|
| RTX A5000 | bench (`logs/bench_a5000.txt`) | 142.7 | 1.81 | 2.36 | 0.64 | 0.50× |
| RTX 5090 | Opus s1 (11-06) | 64.3 | 4.07 | 2.27 | 1.28 | 1 |
| RTX 4090 | Opus strict s1 | 101.1 | 2.53 | 2.50 | 1.50 | 1.17× |
| L40S | Opus strict s3 | 116.1 | 2.20 | 2.56 | 2.11 | 1.65× |
| RTX PRO 6000 | Opus strict s2 | 54.7 | 4.81 | 2.69 | 2.27 | 1.77× |
| RTX PRO 6000 | Opus s3 (11-07) | 55.6 | 4.64 | 2.59 | 2.31 | 1.80× |
| RTX 6000 Ada | DeepSeek s1, no activations, 512 | 98.1 | 2.63 | 2.74 | 1.62 | — |
| RTX 5070 Ti (local) | MATH-500 base eval (11-06), 500 answers | 107 | ~2.3 | ~2.6 | free | — |

The A5000 (Ampere) is exactly as efficient per TB/s as the newer cards, so the bandwidth model holds on all seven
types. The pools of 11-08 (A5000/3090 with a 4090 where one was already running) give each episode 2.5–2.7 TB/s:
about 1.45× a 5090's speed at about 0.55× its price per pass.

## 5. Rules of thumb for this project
- Choose target GPUs by $ per TB/s (table in §2), not by memory: 24 GB is plenty for one 0.8B model.
- One episode = one pool of 2–3 cheap cards (A5000 first, then 3090); put the fastest card first (teacher-forced
  passes and the final MATH-500 score run on the first remote host only).
- Secure stock of any one type is small: take a mix, weight hosts by bandwidth.
- Never run a local vLLM job on the desktop beside two episodes without the memory guard.
- When pods change mid-episode, swap `remote.json` atomically and delete old pods only after no pass uses them
  (`scripts/swap_pool.sh`); do not edit a bash script while a copy of it is running (bash reads scripts as it goes).
