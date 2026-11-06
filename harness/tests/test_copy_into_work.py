"""qwen_pass.copy_into_work never follows a symlink the agent planted in /work. Run: python3 -I tests/test_copy_into_work.py"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python", "adapters"))
import qwen_pass as Q  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as host, tempfile.TemporaryDirectory() as out:
        victim = os.path.join(host, "victim.txt")
        open(victim, "w").write("host file")
        os.symlink(victim, os.path.join(work, "gen_1.jsonl"))          # planted file link
        os.makedirs(os.path.join(host, "dir"))
        os.symlink(os.path.join(host, "dir"), os.path.join(work, "gen_1_acts"))  # planted folder link
        open(os.path.join(out, "gen.jsonl"), "w").write("results")
        os.makedirs(os.path.join(out, "gen_acts"))
        open(os.path.join(out, "gen_acts", "index.json"), "w").write("{}")
        Q.copy_into_work(os.path.join(out, "gen.jsonl"), work, "gen_1.jsonl")
        Q.copy_into_work(os.path.join(out, "gen_acts"), work, "gen_1_acts")
        assert open(victim).read() == "host file"
        assert os.listdir(os.path.join(host, "dir")) == []
        assert not os.path.islink(os.path.join(work, "gen_1.jsonl"))
        assert open(os.path.join(work, "gen_1.jsonl")).read() == "results"
        assert not os.path.islink(os.path.join(work, "gen_1_acts"))
        assert os.listdir(os.path.join(work, "gen_1_acts")) == ["index.json"]
        Q.copy_into_work(os.path.join(out, "gen.jsonl"), work, "gen_1.jsonl")  # replacing a regular file works too
    print("ok: copy_into_work follows no symlink")


if __name__ == "__main__":
    main()
