"""Hold-out scorer (python/adapters/holdout_eval.py): MMLU-Pro answer extraction and prompt, and the graders against the
authors' published first-sample texts (no GPU). Run with the qwen venv (the IFEval checkers need nltk, langdetect,
immutabledict, absl-py): assets/venvs/qwen/bin/python -I tests/test_holdout.py
The published-text tests are skipped unless `holdout_eval.py prepare` has been run and the authors' logs are present."""
import glob
import gzip
import importlib.util
import json
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("holdout_eval", os.path.join(ROOT, "python/adapters/holdout_eval.py"))
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)

DATA = os.path.join(ROOT, "assets", "exp8", "holdout")
AUTHORS = next((p for p in (os.path.join(ROOT, "assets", "lm_understand-logs", "exp8_math", "eval"),
                            os.path.join(os.path.dirname(ROOT), "external", "lm_understand-logs", "exp8_math", "eval"))
                if os.path.isdir(p)), None)
PREPARED = all(os.path.exists(os.path.join(DATA, f"{s}.jsonl")) for s in H.SETS)
try:
    import immutabledict, langdetect, nltk  # noqa: F401,E401
    CHECKER_DEPS = True
except ImportError:
    CHECKER_DEPS = False


class Extraction(unittest.TestCase):
    def test_official_chain(self):
        self.assertEqual(H.mc_extract_official("so the answer is (C). or maybe the answer is D"), "C")   # first match
        self.assertEqual(H.mc_extract_official("blah\nAnswer: F\nmore"), "F")
        self.assertEqual(H.mc_extract_official("I pick B over A here"), "A")                               # last A-J letter
        self.assertIsNone(H.mc_extract_official("no letter here"))

    def test_reasoning_is_cut_at_the_last_think_end(self):
        self.assertEqual(H.mc_extract("the answer is (A)\n</think>\nthe answer is (B)"), "B")
        self.assertEqual(H.mc_extract("still thinking ... the answer is (G)"), "G")      # no </think>: the whole text

    def test_ifeval_answer_is_after_think_end(self):
        self.assertEqual(H.ifeval_response("reasoning</think>\n\n Hello "), "Hello")
        self.assertEqual(H.ifeval_response("cut inside the reasoning"), "")

    def test_mmlupro_prompt(self):
        p = H.mc_prompt("law", "Q?", ["x", "y"])
        self.assertTrue(p.startswith("The following are multiple choice questions (with answers) about law. Think step by step"))
        self.assertTrue(p.endswith('letter choice.\n\n\n\nQuestion:\nQ?\nOptions:\nA. x\nB. y\nAnswer: Let\'s think step by step.'))

    def test_ids_match_the_authors_counts(self):
        ids = json.load(open(H.IDS))
        self.assertEqual([len(ids[s]) for s in H.SETS], [1000, 198, 541])
        self.assertEqual(ids["mmlupro1k"][0], "mmlupro_71")


@unittest.skipUnless(PREPARED and AUTHORS, "run `holdout_eval.py prepare` first; needs the authors' logs")
class PublishedTexts(unittest.TestCase):
    def published(self, ck, s, n):
        rows = {r["id"]: r for r in H.load_rows(DATA, s)}
        out = []
        for d in sorted(glob.glob(os.path.join(AUTHORS, ck, "parts", f"{s}_s*"))):
            per = {p["id"]: p["correct"][0] for p in json.load(open(os.path.join(d, f"{s}_per_problem.json")))}
            for line in gzip.open(os.path.join(d, f"{s}_first_samples.jsonl.gz"), "rt", encoding="utf-8"):
                x = json.loads(line)
                out.append((rows[x["id"]], x, per[x["id"]]))
                if len(out) >= n:
                    return out
        return out

    def test_mmlupro_matches_authors(self):
        for ck in ("unedited", "p50c"):
            for row, x, theirs in self.published(ck, "mmlupro1k", 40):
                self.assertEqual(x["answer"], row["answer"], x["id"])
                self.assertEqual(int(H.grade_one(row, x["text"], "stop")["correct"]), theirs, (ck, x["id"]))

    @unittest.skipUnless(CHECKER_DEPS, "needs nltk, langdetect, immutabledict, absl-py (the qwen venv)")
    def test_ifeval_matches_authors(self):
        for ck in ("unedited", "p10c"):
            for row, x, theirs in self.published(ck, "ifeval", 40):
                g = H.grade_one(row, x["text"], "stop", os.path.join(DATA, "nltk_data"))
                self.assertEqual(int(g["correct"]), theirs, (ck, x["id"]))

    def test_gpqa_items(self):
        rows = H.load_rows(DATA, "gpqa")
        self.assertEqual(len(rows), 198)
        for r in rows:
            self.assertIn(r["answer"], "ABCD")
            self.assertIn("\nD. ", r["prompt"])
        self.assertEqual([r["id"] for r in rows], json.load(open(H.IDS))["gpqa"])


if __name__ == "__main__":
    unittest.main()
