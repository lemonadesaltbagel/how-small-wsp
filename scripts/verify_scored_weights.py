"""Check that the weights each scoring loaded equal base + the declared edit, at the precision the scorer runs (numpy only).
  <python with numpy> -I scripts/verify_scored_weights.py [--only SUBSTR] > notes/verify_scored_weights.jsonl
The scorer (qwen_pass.py --evaluate / holdout) stages the tensors named in the run's edits/ledger.jsonl, copied from
OPENWISH_FINAL_NET, and the worker (qwen_worker.checkpoint) writes them over the base model.safetensors cast to each
tensor's stored dtype (bf16, a few f32). The checkpoint itself is not kept, so this rebuilds it:
  loaded   = base safetensors, with every tensor listed in the score record's edited_tensors replaced by
             cast(final/net/<name>.npy)  (what the worker loaded, given the record)
  expected = base safetensors, with every tensor of final/editset replaced by cast(edit.npz[name])  (the declared edit)
and compares the two bit for bit (sha256 over all tensors in name order). It also checks the base file's sha256, that
base_net (the float32 copy the editors saw) equals the safetensors upcast, that final/net == base_net + edit set over
all tensors (as verify_edits.py), and counts the entries whose stored bits differ from the base (the bf16 count)."""
import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
H = os.path.join(ROOT, "harness")
A = os.path.join(H, "assets", "exp8")
BASE_ST = os.path.join(A, "qwen35-text", "model.safetensors")
BASE_SHA_PREFIX = "68194c7b"  # notes/phase0_log.md 2026-11-06 (Qwen/Qwen3.5-0.8B @ 2fc0636, text checkpoint)

# Scoring records not kept as eval/*.json: the summary JSON printed in these logs (run, label, log).
LOG_RECORDS = [
    ("runs/phase0/qwen0p8b-hinted-p50-strict2-s1", "rescore (logs/rescore_q08b_s1.log)", "logs/rescore_q08b_s1.log"),
    ("runs_laptop/phase0/qwen0p8b-hinted-p50-strict2-s2", "rescore (logs/rescore_q08b_s2.log)", "logs/rescore_q08b_s2.log"),
    ("runs/phase0/qwen9b-hinted-p50-strict2-s1", "rescore (logs/rescore_q9b_s1.log)", "logs/rescore_q9b_s1.log"),
    ("runs_laptop/phase0/deepseekflash-hinted-p50-strict2-s1", "4-sample rescore", "logs/rescore_dsh1.log"),
    ("runs_laptop/phase0/deepseekflash-unhinted-p50-strict2-s2", "4-sample rescore", "logs/rescore_dsu2.log"),
    ("runs_laptop/phase0/deepseekflash-hinted-p50-strict2-s2", "4-sample rescore", "logs/rescore_dsh2_4s.log"),
    ("runs/phase0/opus55-hinted-p50-strict-s1", "4-sample rescore", "logs/rescore4_opus_s1.log"),
]
EXTRA_RUNS = ["runs_laptop/phase0/deepseekflash-unhinted-p50-strict2-s2",  # counted, not yet in the ledger
              "runs_laptop/phase0/deepseekflash-hinted-p50-strict2-s2"]


def read_safetensors(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)
    out = {}
    for k, v in hdr.items():
        a, b = v["data_offsets"]
        dt = {"BF16": np.uint16, "F32": np.uint32}[v["dtype"]]
        out[k] = (v["dtype"], tuple(v["shape"]), mm[a:b].view(dt).reshape(v["shape"]))
    return out


def to_stored(x, dtype):
    """float32 -> the stored bits, as torch's .to(dtype) does (bf16: round to nearest even; NaN kept quiet)."""
    b = np.ascontiguousarray(x, dtype="<f4").view(np.uint32)
    if dtype == "F32":
        return b
    b64 = b.astype(np.uint64)
    r = ((b64 + 0x7FFF + ((b64 >> 16) & 1)) >> 16).astype(np.uint16)
    nan = np.isnan(np.ascontiguousarray(x, dtype="<f4"))
    if nan.any():
        r[nan] = ((b[nan] >> 16) | 0x0040).astype(np.uint16)
    return r


def upcast(bits, dtype):
    return (bits.astype(np.uint32) << 16).view(np.float32) if dtype == "BF16" else bits.view(np.float32)


def fname(k):
    return k.replace("/", "__") + ".npy"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def model_hash(st, repl):
    h = hashlib.sha256()
    for k in sorted(st):
        h.update(k.encode())
        h.update(np.ascontiguousarray(repl.get(k, st[k][2])).tobytes())
    return h.hexdigest()


def records(run):
    """(label, edited_tensors) for every scoring of this run we hold a record of."""
    out = []
    for f, lab in (("score.json", "score"), ("holdout_summary.json", "hold-out")):
        p = os.path.join(H, run, "eval", f)
        if os.path.exists(p):
            d = json.load(open(p))
            if "edited_tensors" in d:
                out.append((lab, d["edited_tensors"]))
            else:
                out.append((lab, None))
    for r, lab, log in LOG_RECORDS:
        if r == run:
            txt = open(os.path.join(ROOT, log)).read()
            body = txt[txt.index("{"):txt.rindex("}") + 1]
            try:
                d = json.loads(body)
            except ValueError:  # one JSON object per line: take the last
                d = json.loads([l for l in txt.splitlines() if l.startswith("{")][-1])
            out.append((lab, d["edited_tensors"]))
    return out


def check_run(st, base_net, run, edits_src=None, recs=None):
    """edits_src: dict name -> float32 array of the declared edit (default: the run's final/editset)."""
    problems, notes = [], []
    rd = os.path.join(H, run)
    if edits_src is None:
        m = json.load(open(os.path.join(rd, "final", "editset", "manifest.json")))
        ed = np.load(os.path.join(rd, "final", "editset", "edit.npz"))
        edits_src = {k: ed[k] for k in m["tensors"]}
        net = os.path.join(rd, "final", "net")
    else:
        m, net = None, None
    expected = {k: to_stored(v, st[k][0]) for k, v in edits_src.items()}
    for k in expected:
        if expected[k].shape != st[k][1]:
            problems.append(f"{k}: edit shape {expected[k].shape} != base {st[k][1]}")
    h_exp = model_hash(st, expected)
    changed_bits = {k: int((expected[k] != st[k][2]).sum()) for k in expected}
    changed_f32 = {k: int((~((np.asarray(v, np.float32) == upcast(st[k][2], st[k][0])) |
                             (np.isnan(v) & np.isnan(upcast(st[k][2], st[k][0]))))).sum()) for k, v in edits_src.items()}
    # final/net == base_net + edit set over every tensor (float32, as verify_edits.py)
    if net is not None:
        for f in sorted(os.listdir(net)):
            if not f.endswith(".npy"):
                continue
            k = f[:-4].replace("__", "/")
            got = np.load(os.path.join(net, f), mmap_mode="r")
            want = edits_src[k] if k in edits_src else np.load(os.path.join(base_net, f), mmap_mode="r")
            a, b = np.asarray(got, np.float32), np.asarray(want, np.float32)
            if a.shape != b.shape or not ((a == b) | (np.isnan(a) & np.isnan(b))).all():
                problems.append(f"{k}: final/net differs from base + edit set")
    scored = []
    for lab, names in (recs if recs is not None else records(run)):
        if names is None:
            scored.append({"record": lab, "ok": None, "note": "no edited_tensors in the record"})
            continue
        loaded = {}
        for k in names:
            src = edits_src[k] if net is None else np.load(os.path.join(net, fname(k)))
            loaded[k] = to_stored(src, st[k][0])
        h_load = model_hash(st, loaded)
        missing = sorted(k for k in expected if changed_bits[k] and k not in names)
        extra = sorted(k for k in names if k not in expected)
        ok = h_load == h_exp
        if not ok:
            problems.append(f"{lab}: loaded model hash != base + declared edit (declared but not loaded: {missing}; "
                            f"loaded but not declared: {extra})")
        scored.append({"record": lab, "ok": ok, "n_loaded": len(names), "sha_loaded": h_load[:16],
                       **({"declared_not_loaded_bf16_changed": missing} if missing else {}),
                       **({"loaded_not_declared": extra} if extra else {})})
    if any(changed_f32[k] and not changed_bits[k] for k in changed_f32):
        notes.append("declared tensors whose change vanishes after the bf16 cast: "
                     + ", ".join(k for k in changed_f32 if changed_f32[k] and not changed_bits[k]))
    numel = sum(int(np.prod(v[1])) for v in st.values())
    cb, cf = sum(changed_bits.values()), sum(changed_f32.values())
    out = {"run": run, "declared_tensors": len(edits_src), "sha_expected": h_exp[:16],
           "changed_bf16_bits": cb, "changed_f32": cf, "manifest_changed_total": m.get("changed_total") if m else None,
           "numel_total": numel, "share_bf16_bits": cb / numel, "records": scored,
           "ok": not problems and all(s["ok"] is not False for s in scored) and bool(scored),
           "problems": problems, **({"notes": notes} if notes else {})}
    print(json.dumps(out), flush=True)
    return out["ok"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    st = read_safetensors(BASE_ST)
    base_net = os.path.join(A, "base_net")
    sha = sha_file(BASE_ST)
    bad = [k for k in st if not np.array_equal(
        np.load(os.path.join(base_net, fname(k)), mmap_mode="r").view(np.uint32), upcast(st[k][2], st[k][0]).view(np.uint32))]
    print(json.dumps({"base": BASE_ST, "sha256": sha, "sha_ok": sha.startswith(BASE_SHA_PREFIX), "tensors": len(st),
                      "base_net_equals_upcast": not bad, "base_net_mismatch": bad[:5]}), flush=True)
    ok = sha.startswith(BASE_SHA_PREFIX) and not bad
    led = json.load(open(os.path.join(ROOT, "notes", "episode_ledger.json")))
    runs = [r["run"] for r in led if r.get("counts") is True or "strict2" in r["run"]] + EXTRA_RUNS
    for run in runs:
        if a.only in run:
            ok &= check_run(st, base_net, run)
    if a.only in "astra-p50c":
        pub = os.path.join(A, "published", "p50c")
        eds = {f[:-4]: np.load(os.path.join(pub, f)) for f in sorted(os.listdir(pub)) if f.endswith(".npy")}
        # conformance scoring staged exactly these files (summary.json config = p50c); the published hashes were
        # checked on 2026-11-06 (both sha256 match), so loaded == declared by construction; recorded for completeness
        ok &= check_run(st, base_net, "exp8_conformance/p50c_math500 (Astra p50c)", edits_src=eds,
                        recs=[("conformance score", sorted(eds))])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
