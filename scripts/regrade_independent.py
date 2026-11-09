"""Independent second grader for saved MATH-500 generations (credibility check; no GPU, stdlib only).
  python3 -I scripts/regrade_independent.py <eval/qwen_math500/math500.jsonl> [...]
The scorer grades the last \\boxed{} after </think> with math-verify. This script re-extracts the last \\boxed{} from
the full saved generation (`.raw.json` next to the jsonl, aligned with `items.json`; the text after the last
</think>) with its own parser and compares it with the gold answer
after the normalisation of the original MATH benchmark (Hendrycks et al. 2021, `is_equiv` / `strip_string`).
String normalisation is stricter than math-verify (no symbolic equivalence), so it can only under-count; we report
agreement, both one-sided disagreement counts and the accuracy each grader gives."""
import json
import sys


def last_boxed(s):
    if not s:
        return None
    i = max(s.rfind("\\boxed"), s.rfind("\\fbox"))
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
    return None


# --- Hendrycks et al. (2021) MATH normalisation, re-implemented ---
def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    for substr in substrs[1:]:
        new_str += "\\frac"
        if substr and substr[0] == "{":
            new_str += substr
        else:
            if len(substr) < 2:
                return string
            a, b = substr[0], substr[1]
            if b != "{":
                new_str += "{" + a + "}{" + b + "}" + substr[2:]
            else:
                new_str += "{" + a + "}" + b + substr[2:]
    return new_str


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a, b = int(a), int(b)
        assert string == f"{a}/{b}"
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except (ValueError, AssertionError):
        return string


def _remove_right_units(string):
    if "\\text{ " in string:
        return string.split("\\text{ ")[0]
    return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new += "\\sqrt" + split
    return new


def strip_string(string):
    string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace("\\%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def is_equiv(a, b):
    if a is None or b is None:
        return False
    try:
        return strip_string(a) == strip_string(b)
    except Exception:
        return a == b


def main():
    for path in sys.argv[1:]:
        import os
        rows = [json.loads(l) for l in open(path)]
        d = os.path.dirname(path)
        items = json.load(open(os.path.join(d, "items.json")))
        gens = json.load(open(os.path.join(d, ".raw.json")))
        text = {(it["key"], it["sample"]): g["text"] for it, g in zip(items, gens)}
        both = mv_only = hd_only = neither = 0
        examples = []
        for r in rows:
            t = text[(r["key"], r["sample"])]
            ans = t.split("</think>")[-1] if "</think>" in t else None   # no </think>: cut off inside thinking
            pred = last_boxed(ans) if ans is not None else None
            hd = is_equiv(pred, r["gold"])
            mv = bool(r.get("correct"))
            if mv and hd:
                both += 1
            elif mv:
                mv_only += 1
                examples.append(("math-verify only", r["key"], r["gold"], pred))
            elif hd:
                hd_only += 1
                examples.append(("hendrycks only", r["key"], r["gold"], pred))
            else:
                neither += 1
        n = len(rows)
        print(json.dumps({"file": path, "n": n, "math_verify_acc": round((both + mv_only) / n, 4),
                          "hendrycks_acc": round((both + hd_only) / n, 4), "agree": round((both + neither) / n, 4),
                          "math_verify_only": mv_only, "hendrycks_only": hd_only}))
        for e in examples[:12]:
            print("   ", e)


if __name__ == "__main__":
    main()
