"""Truncation controls for an edit that closes the thinking earlier (judging criteria 2026-11-13, notes/pending_user_decisions.md).
The unedited model's own thinking on each MATH-500 item (sample 0) is cut and closed, and the answer is generated as in
the evaluation (sampling, per-item seed, 32,768 tokens in all, grading: the worker's own functions):
  T1 matched: cut at the edit's thinking length on that item (tokens before its </think>);
  T2 fixed:   cut at one length for every item (the median of the edit's closed thinking lengths).
An item is left as the base generated it when the base closed by itself within the cut, or the edit never closed.
Caveat: vLLM's presence penalty counts generated tokens only, so the cut thinking (now prompt) is not penalised.
Run on a target pod (vLLM env, model at /root/openwish/model):
  /root/vllm-env/bin/python -I truncation_control.py bundle.json.xz out.json"""
import json
import lzma
import statistics
import sys

sys.path.insert(0, "/root/openwish")
import qwen_worker as qw  # noqa: E402

MAX = 32768
CLOSE = "\n</think>\n\n"


def main():
    b = json.load(lzma.open(sys.argv[1], "rt"))
    out_path = sys.argv[2]
    from vllm import LLM, SamplingParams

    llm = LLM(model="/root/openwish/model", dtype="bfloat16", max_model_len=MAX + 2048, seed=0, gpu_memory_utilization=0.85,
              enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    items, base = b["items"], b["base"]
    heads = [tok.apply_chat_template([{"role": "user", "content": it["user"]}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=True) for it in items]
    think_ids = []
    for r in base:
        p = r["text"].find("</think>")
        think_ids.append((tok.encode(r["text"] if p < 0 else r["text"][:p], add_special_tokens=False), p >= 0))
    results = {}
    for name, L in b["lengths"].items():
        closed = sorted(v for v in L.values() if v is not None)
        fixed = closed[len(closed) // 2]
        for cond, cut_of in (("T1 matched", lambda k: L[k]), ("T2 fixed", lambda k: fixed)):
            prompts, params, idx, recs = [], [], [], [None] * len(items)
            for i, it in enumerate(items):
                c = cut_of(it["key"])
                ids, base_closed = think_ids[i]
                if c is None or (base_closed and len(ids) <= c):
                    recs[i] = dict(base[i], kept="base")
                    continue
                pre = tok.decode(ids[:c])
                prompts.append(heads[i] + pre + CLOSE)
                params.append(SamplingParams(max_tokens=max(MAX - c - 4, 16), seed=qw.item_seed(it["key"], it["sample"]),
                                             **qw.SAMPLING))
                idx.append((i, pre, c))
            outs = llm.generate(prompts, params) if prompts else []
            for (i, pre, c), o in zip(idx, outs):
                recs[i] = {"text": pre + CLOSE + o.outputs[0].text, "finish": o.outputs[0].finish_reason,
                           "tokens": c + 4 + len(o.outputs[0].token_ids), "kept": "cut", "cut": c}
            graded = qw.grade(items, recs)
            s = qw.summarise(graded)
            s.update({"cut_tokens_fixed": fixed if cond == "T2 fixed" else None, "n_cut": len(idx),
                      "median_cut": statistics.median(c for _, _, c in idx) if idx else None})
            results[f"{name} | {cond}"] = {"summary": s, "correct": {g["key"]: g["correct"] for g in graded},
                                          "finish": {g["key"]: g["finish"] for g in graded}}
            print(json.dumps({f"{name} | {cond}": s}), flush=True)
            json.dump(results, open(out_path, "w"))
    print("TRUNCATION_DONE", flush=True)


if __name__ == "__main__":
    main()
