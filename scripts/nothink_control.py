"""Control: the unedited model on the scored MATH-500 items with thinking turned off (enable_thinking=False), everything
else as the evaluation (sampling, per-item seeds, max tokens, grading: the worker's own functions). Tells whether an
edit that makes the model close </think> at once scores what the model scores without thinking.
Run on a target pod (vLLM env, model at /root/openwish/model):
  /root/vllm-env/bin/python -I nothink_control.py <items.json> <out.json> [model dir]
items.json: an evaluation's items file (eval/qwen_math500/items.json), so the problems and sample indices match."""
import json
import sys

sys.path.insert(0, "/root/openwish")
import qwen_worker as qw  # noqa: E402


def main():
    items_path, out = sys.argv[1], sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "/root/openwish/model"
    items = json.load(open(items_path))
    from vllm import LLM, SamplingParams

    max_tokens = 32768
    llm = LLM(model=model, dtype="bfloat16", max_model_len=max_tokens + 2048, seed=0, gpu_memory_utilization=0.85,
              enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": it["user"]}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False) for it in items]
    params = [SamplingParams(max_tokens=max_tokens, seed=qw.item_seed(it["key"], it["sample"]), **qw.SAMPLING) for it in items]
    outs = llm.generate(prompts, params)
    recs = [{"text": o.outputs[0].text, "finish": o.outputs[0].finish_reason, "tokens": len(o.outputs[0].token_ids)} for o in outs]
    graded = qw.grade(items, recs)
    summary = qw.summarise(graded)
    json.dump({"summary": summary, "prompt_example": prompts[0][-200:], "records": graded}, open(out, "w"))
    print(json.dumps({"nothink_control": summary}))


if __name__ == "__main__":
    main()
