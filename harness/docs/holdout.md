# Hold-out scorer for `tasks/exp8`: MMLU-Pro, GPQA Diamond, IFEval

The exp8 success criterion asks for no regression on three or more unrelated benchmarks. To keep our numbers comparable with the published ones, `python/adapters/holdout_eval.py` ports the authors' non-math protocol from `external/lm_understand-logs/exp8_math/code/evaluate.py`: protocol `think35`, with `SAMPLES` and `grade_other`. That driver imports `weightenv.lmo.tasks.{mcqa,ifevalg}`, which the authors never published, so the graders here are re-implementations. Every generation shares the same settings: the chat template with thinking on, temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5, and at most 32,768 new tokens. Generation goes through `qwen_worker.py _one` (`run_vllm`), so per-item seeds are `item_seed(id, sample)`.

| Set | Items | Samples | Prompt (user turn) | Grade |
|---|---|---|---|---|
| `mmlupro1k` | the authors' 1,000 MMLU-Pro test questions (`mmlupro_<question_id>`) | 1 | MMLU-Pro official CoT prompt, zero-shot (below) | official `extract_answer` → `extract_again` → `extract_final` on the text after the last `</think>`, or on the whole text when there is none; no answer counts as wrong (no random guess) |
| `gpqa` | all 198 GPQA Diamond questions (`gpqa_<Record ID>`) | 4 | the same format, subject = high-level domain in lowercase; the 4 options are shuffled per question | the same extractor |
| `ifeval` | all 541 IFEval prompts (`ifeval_<key>`) | 2 | the prompt as is | strict, prompt-level: every instruction must pass the official checker on the answer after the last `</think>`. With no `</think>`, the answer is empty and the item fails. kwargs with value `None` are dropped, and `prompt` is passed when a checker takes it (as in the authors' `grade_other`) |

The score is the mean accuracy over all generations. Item ids, their order and the sample counts come from the authors' `eval/*/parts/<set>_s*/<set>_per_problem.json`. `python/adapters/holdout_ids.json` holds the union over shards in the authors' benchmark-file order (shard i of k = rows i, i+k, …), and it is identical for all four checkpoints. For each set, that order equals the order of the dataset file.

The MMLU-Pro prompt is `initial_prompt.txt` (TIGER-AI-Lab/MMLU-Pro@f418b116, sha256 `d7337806…`) with `{$}` replaced by the category, then `"\n"`, `"Question:\n{q}\nOptions:\n"`, `"A. …\n"` per option (`N/A` options dropped), and finally `"Answer: Let's think step by step."`

## Run

Run everything on the GPU machine with the qwen venv. These extra packages are installed there: `langdetect==1.0.9 nltk==3.10.4 immutabledict==4.3.1 absl-py==2.5.1 pyarrow==26.0.1`. A remote pod needs the same packages.

```bash
# once: download, check sha256, select the authors' ids, fetch NLTK punkt_tab -> assets/exp8/holdout/
assets/venvs/qwen/bin/python -I python/adapters/holdout_eval.py prepare
# smoke run (GPU): 5 problems
assets/venvs/qwen/bin/python -I python/adapters/holdout_eval.py evaluate --model assets/exp8/qwen35-text \
    --set ifeval --data assets/exp8/holdout --out runs/holdout_smoke/ifeval --limit 5 [--edits DIR]
# full: --set mmlupro1k | gpqa | ifeval, without --limit (authors' sample counts; --samples K overrides them)
# through the harness wrapper (edits from OPENWISH_RUN / OPENWISH_FINAL_NET; local or assets/exp8/remote.json):
assets/venvs/sandbox/bin/python -I python/adapters/qwen_pass.py --evaluate --sets mmlupro1k,gpqa,ifeval [--limit 5]
# CPU validation against the authors' published texts, and the unit tests
assets/venvs/qwen/bin/python -I python/adapters/holdout_eval.py regrade --authors ../external/lm_understand-logs/exp8_math/eval --out report.json
assets/venvs/qwen/bin/python -I tests/test_holdout.py
```

`evaluate` writes `OUT/<set>.jsonl`, one line per generation with key, sample, pred/gold or per-instruction results, correct, failure, finish, tokens, `text_head` and `answer_head`. It prints one JSON line: `score`, `accuracy`, `accuracy_se` (binomial over generations, as in qwen_worker), `accuracy_se_problems` (clustered by problem), `share_cut_at_max_tokens`, `mean_tokens`, `failures`, `instruction_level_accuracy` (IFEval), `by_category`, and `data_sha256`. With `--limit` or `--samples` unset, `qwen_pass.py` keeps math500/gsm8k at 500 problems and 1 sample, as before.

## Sources and hashes

| Set | Source (HF dataset @ revision, file) | sha256 of the file | Prepared file sha256 |
|---|---|---|---|
| mmlupro1k | `TIGER-Lab/MMLU-Pro@b189ec76…`, `data/test-00000-of-00001.parquet` (12,032 rows) | `0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8` | `1ad8c8e826af87b565fc228668689a19ea8284eee912f5fcd5c905acaec6630d` |
| gpqa | `Wanfq/gpqa@03ff68d7…`, `gpqa_diamond.csv` (198 rows) | `41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305` | `707b2963c4d9f1be280166bb0f0e93c256c93027dfd1e1538d1d66371e2c47a4` |
| ifeval | `google/IFEval@966cd895…`, `ifeval_input_data.jsonl` (541 rows) | `6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2` | `d763938d11981f71d0cefa45b41100ef17c689a281d1267899471c90729bc5e4` |

- GPQA: the official `Idavidrein/gpqa` is gated, and we have no token for it. Its `gpqa_diamond.csv` has the same size (1,373,492 bytes) and the same git blob id (`7589e3e4…`, from the public HF API) as the ungated mirror `Wanfq/gpqa`, so the two files are byte-identical. `prepare` checks the blob id. The authors' ids are its `Record ID`s, and their order is the CSV order.
- IFEval checkers: google-research `instruction_following_eval` at commit `66cf8a06`, vendored unchanged (Apache-2.0) in `python/adapters/instruction_following_eval/` (see `NOTICE` there; the sha256 values are also in `assets/exp8/holdout/manifest.json`). An AST diff against lm-eval's copy shows no behavioural difference. NLTK punkt_tab zip sha256: `e57f6418…0106`.
- `assets/exp8/holdout/manifest.json` records all of the above. The `prepare` command refuses a download whose sha256 differs.

## Deviations and what could not be checked

- **GPQA option shuffle:** the authors' seed is not published, so their exact order cannot be reproduced. We use `random.Random(int(sha256("gpqa-shuffle:<Record ID>")[:16], 16)).shuffle([correct, inc1, inc2, inc3])`, which is deterministic. The gold letters come out as A 39, B 42, C 67, D 50. Neither the GPQA texts nor its prompt are published (GPQA asks that they not be), so the subject word (the high-level domain in lowercase) and the stripping of whitespace from the question and options are our choices. GPQA scores are comparable in protocol, not item by item.
- **MMLU-Pro zero-shot:** the official script defaults to 5-shot. We chose zero-shot because none of the 400 published texts mentions worked examples. The model paraphrases only the header instruction, and it says that the prompt ends with `Answer: Let's think step by step.` The exact whitespace cannot be checked against the texts.
- **langdetect:** we fix `DetectorFactory.seed = 0` so that grading is deterministic. The checkers leave it unseeded, so three IFEval checks (`english_capital`, `english_lowercase`, `response_language`) can flip from run to run. Of the 37 published answers that hit one of these checks, 2 are sensitive to the seed.
- **Only part of each set can be validated:** the authors kept the full text only for the first 20 items of each shard, which gives 100 MMLU-Pro and 100 IFEval texts per checkpoint, and no GPQA texts.

## Validation: re-grading the authors' published first samples (`regrade`, CPU)

Each cell shows how many of our grades match the authors' `correct[0]`. In brackets: items the authors marked correct / items we mark correct.

| Checkpoint | MMLU-Pro | IFEval |
|---|---|---|
| unedited | 100/100 (44 / 44) | 100/100 (20 / 20) |
| p10c | 100/100 (48 / 48) | 100/100 (46 / 46) |
| p25c | 100/100 (48 / 48) | 100/100 (42 / 42) |
| p50c | 100/100 (50 / 50) | 99/100 (58 / 59) |
| total | **400/400** | **399/400 (99.75 %)** |

- The MMLU-Pro gold letters in the published texts match our prepared answers for all 400 items.
- The published ids for all sets and checkpoints are covered: 1000/1000, 198/198 and 541/541.
- Of the extraction variants we tried on MMLU-Pro, only "after the last `</think>`, else the whole text" agrees 400/400. "None without `</think>`" agrees 394/400, and "the whole text" 376/400.
- The single disagreement is p50c `ifeval_1389` (`change_case:english_capital` + `keywords:letter_frequency`). The answer is an all-caps list of short sentences. langdetect calls it Somali (`so`) under 183 of 200 seeds and English under 17. With seed 0 it says English, so we pass the item, while the authors' unseeded run failed it. This is langdetect's randomness, not a difference in the grader.
