"""Base model on all 1,319 GSM8K test problems x 2 samples (analysis plan, secondary outcome), on the pool host in
OPENWISH_QWEN_REMOTE. Writes harness/runs/exp8_conformance/base_gsm8k_full/ (the 500-problem run stays in base_gsm8k).
  cd harness && OPENWISH_QWEN_REMOTE=<remote.json> assets/venvs/sandbox/bin/python -I ../scripts/base_gsm8k_full.py"""
import json
import os
import sys

H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")
sys.path.insert(0, os.path.join(H, "python", "adapters"))
import qwen_pass as Q  # noqa: E402

empty = os.path.join(H, "assets", "exp8", "published", "_none")
os.makedirs(empty, exist_ok=True)
out = os.path.join(H, "runs", "exp8_conformance", "base_gsm8k_full")
res = Q.run_worker(["evaluate", "--model", "{model}", "--edits", "{edits}", "--set", "gsm8k", "--data", "{data}",
                    "--limit", "1319", "--samples", "2", "--out", "{out}"], {"edits": empty, "tag": "base"}, out)
res["config"] = "base, gsm8k 1319 x 2"
os.makedirs(out, exist_ok=True)
json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1)
print(json.dumps({k: res.get(k) for k in ("config", "ok", "score", "accuracy_se", "share_cut_at_max_tokens", "mean_tokens", "problems", "error")}), flush=True)
