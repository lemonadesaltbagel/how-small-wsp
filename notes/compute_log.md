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
| 2026-11-08 | vx1mfk4ufa7137 | RTX 4090 secure ($0.89/h) | Opus strict s1 target passes (desktop harness), created 02:24 | | |
| 2026-11-08 | r2nlncawb0yqzf | RTX PRO 6000 secure ($2.49/h) | Opus strict s2 target passes (desktop harness), created 02:24 | | |
| 2026-11-08 | hdqf0i76ys7swk | L40S secure ($1.09/h) | Opus strict s3 target passes (laptop harness), created 02:24 | | |
Rule reviewer (deepseek-flash, pre_tool hook): ~1k tokens per Python block, negligible.

| 2026-11-08 | ynmzn2c2jjwwbp | RTX PRO 6000 secure ($2.49/h) | STRAY: created 02:26 by the GPU-fallback loop for strict s3 (create returned output the loop did not parse as an id, so it tried the next GPU); never used. Deleted by the user before 03:00 (the agent's delete was blocked by the permission classifier) | ~32 | ~1.33 |
