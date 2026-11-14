# RunPod cost split by purpose (2026-11-14)

Source: `runpodctl billing pods --bucket-size day --start-time 2026-11-05T00:00:00Z --end-time 2026-11-14T06:00:00Z`
(this project's pods saved as notes/run_records/runpod_billing_2026-11-14.json; GPU + disk, per pod). A pod belongs to this project when
its id appears in logs/, notes/ or scripts/; the 48 pods with no mention ($151.77) are other projects' (the account
is shared). Purpose from the files that mention the pod (post_* pipelines, acquire_* logs, pods_strict.txt) and the
compute log. Pool pods served one episode each (swap_pool); the two editor-serving pods served the Qwen 9B / 27B editors.

| purpose | USD | pods |
|---|---|---|
| main matrix: target passes, scoring, hold-out (pools) | 177.91 | 44 |
| main matrix: serving the Qwen 9B / 27B editors (A40, A100) | 79.25 | 2 |
| Opus 5.5 strict episodes (pools, scoring, hold-out) | 99.69 | 12 |
| re-scoring and controls (4-sample re-scores, truncation, thinking off, cf48) | 47.97 | 55 |
| Opus 5.5 first round (pre-strict rules) | 70.20 | 4 |
| abandoned runs under earlier rules (deepseek-flash strict v1 s1, pre-strict resumes) and one stray pod | 40.02 | 8 |
| setup and smoke tests | 3.16 | 3 |
| **total, this project** | **518.20** | 128 |

Not included: the deepseek-flash editor and reviewer (DeepSeek API), Opus 5.5 / Claude Sonnet 5.5 (subscription),
Codex labels and reviews, and the local nodes (desktop 5070 Ti, laptop 4060). Many of the 55 re-scoring pods are
acquisition attempts billed a few cents each.
