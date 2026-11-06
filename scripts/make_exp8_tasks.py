"""Write the scale-ladder exp8 task variants into harness/tasks/exp8sl/.

Source: the authors' v3 prompt of run p50c (github.com/RikiyaT/lm_understand-logs, CC BY 4.0), as prepared by
OpenWiSH's scripts/prepare_exp8.py into harness/assets/exp8/prompts/p50.txt (or read from the logs clone).

Variants (prompt dimension of Q1):
  hinted    the v3 prompt verbatim (it names repetition as an example failure, twice)
  unhinted  the v3 prompt with the two repetition hints removed. The authors' own shortened v4 prompt was never
            published, so this is our construction; every other sentence is unchanged.

Both variants get the same appendix (harness notes, see APPENDIX below) and the same task settings: the original
exp8 settings plus the [rationale] rule, a mandatory save_edit reason (prelude) and per-episode editor caps.

usage: python3 scripts/make_exp8_tasks.py [--passes 50]
"""
import argparse
import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = os.path.join(ROOT, "harness")
OUT = os.path.join(H, "tasks", "exp8sl")
SOURCES = [os.path.join(H, "assets", "lm_understand-logs", "exp8_math", "runs", "p50c", "turns", "t001", "prompt.txt"),
           os.path.join(ROOT, "external", "lm_understand-logs", "exp8_math", "runs", "p50c", "turns", "t001", "prompt.txt")]
SOURCE_SHA256 = None  # filled from the first run; checked by --check

# The two hints, exactly as they appear in the v3 prompt, and their unhinted replacements.
UNHINT = [
    ("Repairing a specific failure mechanism is in scope: for example, if the model often fails because it repeats "
     "itself and never finishes, finding and fixing what drives that repetition counts.",
     "Repairing a specific failure mechanism is in scope."),
    ("repairing a specific failure inside the thinking (such as repetition) is in scope.",
     "repairing a specific failure inside the thinking is in scope."),
    ("or being cut off at the token limit (often while repeating itself in its thinking).",
     "or being cut off at the token limit."),
]

APPENDIX = """NOTES FROM THE EXPERIMENTERS. The original sessions of this task received the first two notes during the session; you get them now.
(1) Your forward-pass budget counts passes, not their size or how long they take: a generation pass with 256 answers per model ("both") or 512 answers costs the same one pass as a pass with 32. Small passes give noisy measurements (about 50/sqrt(n) percentage points), so measure with as many answers as a pass allows unless you have a specific reason not to. The time a pass takes does not count against you.
(2) Corrections about the tools (no action needed if they do not concern you): (a) Attention probabilities ("attn_layers") work; give full-attention layer indices (3, 7, 11, 15, 19, 23). (b) Separate generation passes do not reproduce each other exactly, even for the same request and weights (the server's batching changes the random numbers): compare edited and original only within one "model": "both" pass, and never repeat an identical request with unchanged edits (it costs a pass and adds little). (c) The evaluation prompt, exactly as the model sees it (thinking on), for a problem P is EVAL_PREFIX + P + EVAL_SUFFIX. Both strings are predefined in the kernel; they hold the chat template's special tokens, so build such texts in Python instead of typing those tokens. The model's answer follows EVAL_SUFFIX directly. To score a generated answer with a teacher-forced pass, send EVAL_PREFIX + P + EVAL_SUFFIX + answer as one raw text: write the list of texts as JSON to a file under /work from Python, and send "texts_file": "/work/<name>.json" in place of "texts" ("chats" re-renders assistant turns and drops their reasoning). (d) Before every pass, write to /work/notes.md the exact current edit (tensors, what was changed, strength) and the problem set, so that you can rebuild them if the conversation is restarted.
(3) EXPLAIN EVERY STEP. Before every ```python or ```forward block, explain in plain English, outside the block, what the step does and why (at least two sentences). A block without an explanation is sent back unrun and costs nothing.
(4) Every save_edit call must give reason="...": one or more sentences saying which component you change, how, and what evidence makes you expect it to help. save_edit refuses an edit without a reason. The reasons are kept with the edit."""

PRELUDE_EXTRA = r'''

# --- scale-ladder addition: the evaluation prompt's wrapper (editors never need to type the template's special tokens) ---
EVAL_PREFIX = "<|im_start|>user\n"
EVAL_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n"

# --- scale-ladder addition: every save_edit needs a reason (kept in the ledger with the edit) ---
_save_edit_v3 = save_edit
MIN_REASON_WORDS = 8


def save_edit(name, tensor, reason=""):
    """Save an edit of one tensor. reason: which component you change, how, and the evidence (required)."""
    if len(str(reason).split()) < MIN_REASON_WORDS:
        raise ValueError(f"save_edit needs reason='...' of at least {MIN_REASON_WORDS} words: which component you change, "
                         "how, and what evidence makes you expect it to help. Nothing was saved.")
    return _save_edit_v3(name, tensor, reason)
'''


def toml(variant, passes, prompt_file):
    return f'''# Generated by scripts/make_exp8_tasks.py. Original exp8 settings (OpenWiSH tasks/exp8/p{passes}.toml) plus
# [rationale], a mandatory save_edit reason (prelude.py) and per-episode editor caps.
id = "exp8sl-{variant}-p{passes}"
description = "Scale ladder exp8, {variant} prompt: make Qwen3.5-0.8B better at math by editing its weights; Python over the weights and {passes} forward passes."
prompt = "{prompt_file}"
appendix = "appendix.txt"
hidden = ["../../assets/exp8/eval"]

[network]
base = "../../assets/exp8/base_net"

[[mounts]]
host = "../../assets/exp8/qwen35-text"
guest = "/model"

[[mounts]]
host = "../../assets/exp8/practice"
guest = "/practice"

[python]
runs = 300
timeout_s = 900
mem_gb = 11
max_out = 10000
prelude = "prelude.py"

[forward.forward]
command = ["{{root}}/assets/venvs/sandbox/bin/python", "-I", "{{root}}/python/adapters/qwen_pass.py"]
passes = {passes}
out_dir = "fwd_{{k}}"
max_out = 10000
timeout_s = 7200

[edits.policy]
forbidden = ["model.language_model.norm.weight", "model.language_model.embed_tokens.weight", "lm_head.weight"]
forbid_whole_tensor_scale = true

[finish]
require_all_passes = true
max_turns = 600
# about 1.7x the published 50-pass run (86.8M input tokens incl. 80.3M cached, 2.9M output)
max_editor_total_tokens = 150000000
max_editor_output_tokens = 6000000

[handoff]
max_turns_per_conversation = 40

[rationale]
min_words = 15
tools = ["python", "forward"]
max_rejects_in_row = 2

[editor]
effort = "max"

[score]
command = ["{{root}}/assets/venvs/sandbox/bin/python", "-I", "{{root}}/python/adapters/qwen_pass.py", "--evaluate", "--sets", "math500", "--limit", "500"]

[access]
weights = "code"
data = ["practice problems with answers (GSM8K and MATH training sets)"]
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=50)
    a = ap.parse_args()
    src = next(p for p in SOURCES if os.path.exists(p))
    v3 = open(src).read()
    print("source", src, hashlib.sha256(v3.encode()).hexdigest())
    if a.passes != 50:
        raise SystemExit("only the 50-pass prompt is derived so far (the 10/25 prompts differ in more than the number)")
    unhinted = v3
    for old, new in UNHINT:
        assert unhinted.count(old) == 1, old
        unhinted = unhinted.replace(old, new)
    for word in ("repet", "repeat"):
        assert word not in unhinted.lower(), word
    os.makedirs(os.path.join(OUT, "prompts"), exist_ok=True)
    for variant, text in (("hinted", v3), ("unhinted", unhinted)):
        pf = f"prompts/{variant}-p{a.passes}.txt"
        open(os.path.join(OUT, pf), "w").write(text)
        open(os.path.join(OUT, f"{variant}-p{a.passes}.toml"), "w").write(toml(variant, a.passes, pf))
        print(variant, len(text), "chars", hashlib.sha256(text.encode()).hexdigest()[:16])
    open(os.path.join(OUT, "appendix.txt"), "w").write(APPENDIX + "\n")
    base_prelude = open(os.path.join(H, "tasks", "exp8", "prelude.py")).read()
    open(os.path.join(OUT, "prelude.py"), "w").write(base_prelude.rstrip() + "\n" + PRELUDE_EXTRA)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
