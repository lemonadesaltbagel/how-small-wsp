| date | pod | gpu | purpose | minutes | est. cost USD |
|---|---|---|---|---|---|
| 2026-11-06 | gzfx1x62svqz2y | RTX 4090 secure ($0.89/h) | bwrap-in-container test (fails: no userns) | ~6 | 0.09 |
| 2026-11-06 | 3pn2czbdsk2n1t | RTX PRO 6000 secure ($2.49/h) | open-editor smoke test (Qwen3.5-27B) + remote scoring check, 20:36–21:35 AEDT | 60 | 2.49 |

| 2026-11-06 | pw7y4j6wub1w1r | RTX 5090 secure ($1.19/h) | Opus seed 1 target passes, 11-06 21:38 – 11-07 23:02 (deleted: user balance low) | 1524 | 30.23 |
| 2026-11-06 | iz4y3dff9ug5d8 | RTX 5090 secure ($1.19/h) | Opus seed 2 target passes, 11-06 21:38 – 11-07 23:02 (deleted: user balance low) | 1524 | 30.23 |
| 2026-11-07 | yyxolba73b3qsr | RTX 6000 Ada secure ($0.99/h) | DeepSeek-flash seed 1 target passes, 02:56–05:05 (deleted when the run was paused) | 129 | 2.13 |
| 2026-11-07 | nqiujjt6drtt60 | RTX PRO 6000 secure ($2.49/h) | Opus seed 3 target passes (laptop harness), 22:17–23:02 (deleted: user balance low) | 45 | 1.87 |

**Total to 2026-11-07 23:02: ≈ $67.0** (0.09 + 2.49 + 2.13 + 30.23 + 30.23 + 1.87). No pods running.
| 2026-11-07 | mer4scszz551q4 | RTX 5090 secure ($1.19/h) | Opus s1 resume target passes + score, 23:27 – 11-08 01:51 | 144 | 2.86 |
| 2026-11-07 | cnybjl990pd7ri | RTX PRO 6000 secure ($2.49/h) | Opus s2 resume target passes + score, 23:27 – 11-08 ~01:45 (holdout stopped: run invalid) | ~138 | ~5.73 |
| 2026-11-07 | 05se8n39mympkg | RTX 5090 secure ($1.19/h) | Opus s3 resume (laptop), 23:27 – 11-08 ~01:45 (stopped: pre-strict rules) | ~138 | ~2.74 |
| 2026-11-08 | odwbx9trcyaxn1 | GPU secure ($2.49/h) | DeepSeek s1 resume (mac node), 00:41 – ~01:45 (stopped: pre-strict rules) | ~64 | ~2.66 |

**Total to 2026-11-08 01:51: ≈ $81.0** (67.0 + 2.86 + 5.73 + 2.74 + 2.66). No pods running.

Strict-rule reruns (task hinted-p50-strict; 5090s were out of stock):
| 2026-11-08 | vx1mfk4ufa7137 | RTX 4090 secure ($0.89/h) | Opus strict s1 target passes, then first host of its pool; 02:24 – 20:32 (after scoring and hold-out) | ~1088 | 16.36 (billed) |
| 2026-11-08 | r2nlncawb0yqzf | RTX PRO 6000 secure ($2.49/h) | Opus strict s2 target passes (desktop harness), created 02:24 | | |
| 2026-11-08 | hdqf0i76ys7swk | L40S secure ($1.09/h) | Opus strict s3 target passes (laptop harness), 02:24 – 04:36 (replaced by a pool: half the speed of a 5090) | ~132 | ~2.40 |
Rule reviewer (deepseek-flash, pre_tool hook): ~1k tokens per Python block, negligible.

| 2026-11-08 | ynmzn2c2jjwwbp | RTX PRO 6000 secure ($2.49/h) | STRAY: created 02:26 by the GPU-fallback loop for strict s3 (create returned output the loop did not parse as an id, so it tried the next GPU); never used. Deleted by the user before 03:00 (the agent's delete was blocked by the permission classifier) | ~32 | ~1.33 |

**Reconciled with RunPod billing (2026-11-08 ~03:40, `runpodctl billing pods --bucket-size hour --start-time … --end-time …`):** RunPod bills each pod separately (GPU + container disk, from creation to deletion). The 10 deleted pods of this project were billed **$82.63** (GPU $82.0 + disk $0.59) against the estimates above of $81.03 (−2 %; the resumed pods bill their setup minutes and disk). The pods of 11-08 (4 running + the stray) are not billed yet (lag ~1 h). Pods on the account from 11-05 ($55.19) belong to another project. Balance at 03:40: $130.14; spend rate $5.42/h.

GPU pools (11-08 ~04:33; passes split across hosts, see notes/target_gpu_cost.md). All secure cloud, 100 GB disk:
| 2026-11-08 | vbdv6tys477nvr | RTX A5000 ($0.27/h) | benchmark: replay of Opus s1 pass 1 (deletes itself) , created 03:09 | | |
| 2026-11-08 | ym70xf2dazbop1 | RTX A5000 ($0.27/h) | split-pass test, then Opus strict s1 pool, created 03:39, deleted 18:32 (idle once s1 finished: scoring uses the first host only) | 893 | ~4.0 |
| 2026-11-08 | 522txdq1n7oz6j | RTX A5000 ($0.27/h) | Opus strict s1 pool, created 04:30, deleted 18:32 (same) | 842 | ~3.8 |
| 2026-11-08 | blkvi438r5edfl | RTX 3090 ($0.50/h) | Opus strict s2 pool (first host: scoring, hold-out), 04:21 – 23:09 | ~1128 | 9.32 (billed) |
| 2026-11-08 | fjjukomg74e7x8 | RTX 3090 ($0.50/h) | Opus strict s2 pool, 04:21 – 23:09 | ~1128 | 9.32 (billed) |
| 2026-11-08 | 3kyhmqepm4a3jl | RTX A5000 ($0.27/h) | Opus strict s2 pool, 04:12 – 23:09 | ~1137 | 5.19 (billed) |
| 2026-11-08 | l4uv2c1qm9l4hp | RTX 3090 ($0.50/h) | Opus strict s3 pool (first host: scoring, hold-out), 04:21 – 11-09 17:24 | ~2223 | 18.52 (billed) |
| 2026-11-08 | 6ohqgldauh3p00 | RTX A5000 ($0.27/h) | Opus strict s3 pool, 04:24 – 11-09 13:01 (deleted once the episode ended) | ~1957 | 9.26 (billed) |
| 2026-11-08 | 9o7nukwndavaro | RTX A5000 ($0.27/h) | Opus strict s3 pool, 04:24 – 11-09 13:01 (same) | ~1957 | 9.26 (billed) |
| 2026-11-08 | ge3fudq3awgh7y | RTX A5000 ($0.27/h) | DeepSeek strict s1 pool, created 04:30, deleted 21:57 (idle once the episode ended; scoring uses the first host) | 1047 | ~4.7 |
| 2026-11-08 | 21q9v3a40iumt4 | RTX A5000 ($0.27/h) | DeepSeek strict s1 pool, created 04:33, deleted 21:57 (same) | 1044 | ~4.7 |
| 2026-11-08 | jafz0mqatxadgl | RTX 4090 secure ($0.89/h) | DeepSeek strict s1 target passes, then first host of its pool; ~02:35 – 23:29 (hold-out skipped: the submission is the unedited model) | ~1254 | 17.97 (billed, last hour may lag) |
Pools: s1 = 4090 + 2×A5000 + desktop 5070 Ti when free; s2 = 2×3090 + A5000 + desktop 5070 Ti (PRO 6000 retired once idle);
s3 = 3090 + 2×A5000; DeepSeek = 4090 + 2×A5000. Spend ~$5.4/h as before, ~2.2× the bandwidth.

| 2026-11-08 | jrji6yisbhzqlr | RTX A5000 ($0.27/h) | test of the split-activation fix, then Opus strict s2 pool (4th host), 05:40 – 23:09 | ~1049 | 4.77 (billed) |
| 2026-11-08 | r2nlncawb0yqzf | RTX PRO 6000 ($2.49/h) | Opus strict s2 single target, 02:24 – 04:38 (retired: pool) | ~134 | ~5.56 |
| 2026-11-08 | vbdv6tys477nvr | RTX A5000 ($0.27/h) | A5000 benchmark, 03:09 – 05:35 | ~146 | ~0.66 |

**Total at 2026-11-09 17:24, all pods deleted: $212.58 billed** (`runpodctl billing pods --pod-id <id> --bucket-size day`,
summed over the 27 pods above; ~349 card-hours = GPU amount ÷ hourly price). Strict episodes, target side incl. scoring,
hold-out and idle time: s1 $24.57, s2 $35.19, s3 $39.42 (2 usage-limit stops, network outage), DeepSeek s1 $28.76;
first round, tests and the stray pod $84.62. DeepSeek API: ¥65.2 → ¥35.9 (¥29.3 ≈ $4, strict s1 + rule reviewer).

**Main matrix, interim (2026-11-12 00:40, `runpodctl billing pods --bucket-size day` from 11-09 00:00Z; last hours lag):**
42 pods of this project billed **$237.64** since 11-09 (pods found in logs/pool_hosts.jsonl, acquire/post/swap logs,
plus the two editor pods); 25 more pods on the account in the same window ($74.61) belong to other projects
(rw-*, wnt-*) or could not be attributed. Editors: 27B editor (A100, 5nx3) $68.89, 9B editor (5plj) $10.36; the
0.8B editor ran on its pool pods. Pools are shared and swapped between runs, so a per-episode split needs the pod
timelines (to do once the DS s2 seeds end). Lower bounds from each run's post-helper pods: DS hinted s1 $20.98,
DS unhinted s1 $20.17, 27B hinted s2 $32.08 (with the editor's share not included), 27B unhinted s2 $26.62.
DeepSeek API: ¥134.29 (11-10 02:08) → ¥100.05 (11-11 23:30) for DS hinted s1 + unhinted s1 (¥34.2 ≈ $4.7).
- 2026-11-12 07:10: rescore pod lwin1r2rsgcp3k (5090, $0.69/h) deleted after ~7.9 h (11-11 23:14 → 07:10) ≈ $5.5:
  DS hinted s1 4-sample rescore, the thinking-off control and the full GSM8K base; idle 05:28–07:10 (~$1.2), because
  the GSM8K chain did not delete the pod on completion. Running: v0sy (3090, $0.22/h) and h592 (5090, $0.69/h) for
  the DS s2 episodes.

## 2026-11-12 — DS unhinted s2 pod h5920 (5090, $0.69/h)

Episode pool from 2026-11-11 13:11, then the hold-out (18:57–19:50) and a 4-sample rescore (19:50–23:20) that scored
the base by mistake (no ledger staged): ~3.5 h ≈ $2.4 of the pod's time went to an unneeded base measurement.
Deleted 23:20. Retry on a new pod (chain_rescore_dsu2_retry.sh).
