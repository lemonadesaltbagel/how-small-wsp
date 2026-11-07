"""The linear-time MC extraction in holdout_eval.py must equal the official (quadratic) regexes on every text.
Run: python3 -I tests/test_mc_extract_equivalence.py [path to holdout_eval.py]"""
import gzip
import glob
import importlib.util
import json
import os
import random
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def official(text):
    m = re.search(r"answer is \(?([A-J])\)?", text)
    if m:
        return m.group(1)
    m = re.search(r".*[aA]nswer:\s*([A-J])", text)
    if m:
        return m.group(1)
    m = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    return m.group(0) if m else None


def load(path):
    spec = importlib.util.spec_from_file_location("holdout_eval_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))
    spec.loader.exec_module(mod)
    return mod


def random_text(rng, n):
    atoms = ["Answer: ", "answer:", "Answer:\n ", "answer is (", "the answer is ", "A", "B", "J", "K", " ", "\n", "x", "(C)",
             "Answer: C", "answer:  D\n", "I", "a", ".", "Hmm", "E.", " F ", "\n\n", "</think>"]
    return "".join(rng.choice(atoms) for _ in range(n))


def main():
    mod = load(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "python", "adapters", "holdout_eval.py"))
    rng = random.Random(0)
    n = 0
    for i in range(20000):
        t = random_text(rng, rng.randint(0, 60))
        assert mod.mc_extract_official(t) == official(t), repr(t)
        n += 1
    authors = os.path.join(ROOT, "..", "external", "lm_understand-logs", "exp8_math", "eval")
    for f in glob.glob(os.path.join(authors, "*", "parts", "*", "*_first_samples.jsonl.gz")):
        if "mmlupro" not in f and "gpqa" not in f:
            continue
        for line in gzip.open(f, "rt"):
            t = json.loads(line)["text"]
            t = t.split("</think>")[-1] if "</think>" in t else t
            assert mod.mc_extract_official(t) == official(t), f
            n += 1
    long_text = ("Let me check option B again. " * 4000) + "\nAnswer: C"
    t0 = time.time()
    assert mod.mc_extract_official(long_text) == "C"
    assert time.time() - t0 < 1.0, "the linear version must stay fast on long texts"
    print(f"ok: linear extraction equals the official regexes on {n} texts")


if __name__ == "__main__":
    main()
