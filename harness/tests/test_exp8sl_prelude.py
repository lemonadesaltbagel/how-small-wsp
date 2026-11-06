"""EVAL_PREFIX / EVAL_SUFFIX in tasks/exp8sl/prelude.py must reproduce the evaluation prompt exactly
(the chat template with thinking on). Run with the qwen venv: assets/venvs/qwen/bin/python -I tests/test_exp8sl_prelude.py"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "assets", "exp8", "qwen35-text")
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def constants():
    src = open(os.path.join(ROOT, "tasks", "exp8sl", "prelude.py")).read()
    ns = {}
    for name in ("EVAL_PREFIX", "EVAL_SUFFIX"):
        line = re.search(rf"^{name} = .*$", src, re.M).group(0)
        exec(line, ns)
    return ns["EVAL_PREFIX"], ns["EVAL_SUFFIX"]


def main():
    if not os.path.exists(os.path.join(MODEL, "tokenizer_config.json")):
        print("skip: no model assets")
        return
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    pre, suf = constants()
    for p in ("What is 1+1?", "Let $f(x)=x^2$.\nFind $f(3)$.", "spaced  "):
        want = tok.apply_chat_template([{"role": "user", "content": p + "\n" + INSTRUCTION}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        assert pre + p + suf == want, (repr(pre + p + suf), repr(want))
    print("ok: EVAL_PREFIX + P + EVAL_SUFFIX matches the chat template")


if __name__ == "__main__":
    sys.exit(main())
