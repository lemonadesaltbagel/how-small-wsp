"""qwen_pass.read_texts_file: JSON list of strings under /work only. Run: python3 -I tests/test_texts_file.py"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python", "adapters"))
import qwen_pass as Q  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as other:
        os.environ["OPENWISH_WORK"] = work
        json.dump(["<|im_start|>user\nhi", "b"], open(os.path.join(work, "t.json"), "w"))
        assert Q.read_texts_file("/work/t.json") == (["<|im_start|>user\nhi", "b"], None)
        os.makedirs(os.path.join(work, "sub"))
        json.dump(["x"], open(os.path.join(work, "sub", "u.json"), "w"))
        assert Q.read_texts_file("/work/sub/u.json")[0] == ["x"]
        json.dump(["secret"], open(os.path.join(other, "s.json"), "w"))
        os.symlink(os.path.join(other, "s.json"), os.path.join(work, "link.json"))
        for bad in ("/work/link.json", "/work/../etc/passwd", "/etc/passwd", "t.json", "/work/missing.json", 7):
            texts, err = Q.read_texts_file(bad)
            assert texts is None and err, (bad, texts, err)
        open(os.path.join(work, "bad.json"), "w").write('{"a": 1}')
        assert Q.read_texts_file("/work/bad.json")[1].endswith("JSON list of strings")
    print("ok: texts_file")


if __name__ == "__main__":
    main()
