"""Check that a generation pass split across GPU hosts (qwen_pass.py hosts list) returns what one host returns.
Runs ON the harness node: python3 -I scripts/test_split_pass.py <adapters dir> <host.json> [<host2.json> ...]
Pass 1: the first host alone. Pass 2: all hosts plus {"local": true}. Pass 3: single-model split.
Small request (6 problems x 2 samples, 1,024 tokens, activations of 3 layers)."""
import json, os, shutil, subprocess, sys, tempfile

ad, hosts = os.path.abspath(sys.argv[1]), [json.load(open(p)) for p in sys.argv[2:]]
H = os.path.dirname(os.path.dirname(os.path.abspath(ad)))      # harness/
PY = os.path.join(H, "assets", "venvs", "sandbox", "bin", "python")
probs = json.load(open(os.path.join(H, "runs/phase0/opus55-hinted-p50-s1/requests/forward_1.json")))["problems"][:6]
T = tempfile.mkdtemp(prefix="splittest_")


def run(name, req, cfg):
    d = os.path.join(T, name)
    for sub in ("run", "out", "work"):
        os.makedirs(os.path.join(d, sub))
    json.dump(req, open(os.path.join(d, "req.json"), "w"))
    json.dump(cfg, open(os.path.join(d, "remote.json"), "w"))
    env = dict(os.environ, OPENWISH_REQ=os.path.join(d, "req.json"), OPENWISH_NET=os.path.join(H, "assets/exp8/base_net"),
               OPENWISH_OUT=os.path.join(d, "out"), OPENWISH_RUN_DIR=os.path.join(d, "run"), OPENWISH_WORK=os.path.join(d, "work"),
               OPENWISH_PASS="1", OPENWISH_QWEN_REMOTE=os.path.join(d, "remote.json"))
    p = subprocess.run([PY, "-I", os.path.join(ad, "qwen_pass.py")], env=env, capture_output=True, text=True, cwd=H)
    lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
    res = json.loads(lines[-1]) if lines else {"ok": False, "error": p.stderr[-2000:]}
    print(f"== {name}: ok={res.get('ok')} hosts={res.get('gpu_hosts')} {str(res.get('error', ''))[:300]}")
    return res, os.path.join(d, "out")


def recs(out, f):
    return [json.loads(l) for l in open(os.path.join(out, f))]


def compare(a, b, outa, outb, files):
    ok = True
    for k in ("type", "model"):
        ok &= a.get(k) == b.get(k)
    for f in files:
        ra, rb = recs(outa, f), recs(outb, f)
        same_keys = [(x["key"], x["sample"]) for x in ra] == [(x["key"], x["sample"]) for x in rb]
        same_text = sum(x["text"] == y["text"] for x, y in zip(ra, rb))
        print(f"   {f}: {len(ra)} vs {len(rb)} records, same (key, sample) order {same_keys}, identical texts {same_text}/{len(ra)}")
        ok &= same_keys and len(ra) == len(rb)
    ia, ib = a.get("activations"), b.get("activations")
    print(f"   activations equal shapes: {ia == ib}  {json.dumps(ib)[:300]}")
    ok &= ia == ib
    print(f"   summaries: {json.dumps({k: a[k] for k in a if k in ('summary', 'paired')})[:300]}\n              {json.dumps({k: b[k] for k in b if k in ('summary', 'paired')})[:300]}")
    return ok


req = {"type": "generate", "model": "both", "problems": probs, "n": 2, "max_tokens": 1024, "activations": {"layers": [0, 12, 24], "mlp": True}}
bad, _ = run("reserved_key", dict(req, _parts=[]), hosts[0])
assert bad.get("ok") is False and "reserved" in bad.get("error", ""), bad
r1, o1 = run("single_host", req, hosts[0])
r2, o2 = run("split_both", req, {"hosts": hosts + [{"local": True, "weight": 0.9}]})
all_ok = r1.get("ok") and r2.get("ok") and compare(r1, r2, o1, o2, ["gen_original.jsonl", "gen_edited.jsonl"])
req3 = dict(req, model="original", activations=None)
req3.pop("activations")
r3, o3 = run("single_host_original", req3, hosts[0])
r4, o4 = run("split_original", req3, {"hosts": hosts + [{"local": True, "weight": 0.9}]})
all_ok = all_ok and r3.get("ok") and r4.get("ok") and compare(r3, r4, o3, o4, ["gen.jsonl"])
print("RESULT", "PASS" if all_ok else "FAIL", T)
