"""Build the data release: one .tar.gz per episode (trajectory, prompts, tool calls, reviews, audits, scored outputs and
the submitted edit), plus one per auxiliary folder, with a per-file SHA-256 list.

Left out: full model copies (final/net, edits/current), intermediate edit tensors (edits/blobs, replayable from the base
model and edits/ledger.jsonl only for the submitted state), sandbox scratch (work/) and smoke tests. GPQA hold-out
records quote the questions, which GPQA asks not to post in plain text: they go into one archive encrypted with the
GPQA dataset's own public password (openssl enc -d -aes-256-cbc -pbkdf2 -pass pass:deserted-untie-orchid).

  python3 -I scripts/make_release.py build  OUT_DIR   # on a node (compresses ~3 GB)
  python3 -I scripts/make_release.py verify OUT_DIR [EXTRACTED_HARNESS_DIR]
Stdlib only. Paths inside the archives are relative to harness/ (e.g. runs/phase0/<run>/tool_calls.jsonl)."""
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = os.path.join(ROOT, "harness")
EPISODE_ROOTS = ["runs/phase0", "runs_laptop/phase0"]
AUX = ["runs/exp8_conformance", "runs/external_audit", "runs/holdout", "runs/note6_reaudit", "runs_laptop/calib"]
SKIP_DIRS = {("final", "net"), ("edits", "current"), ("edits", "blobs")}
GPQA_ASSET, GPQA_PASS = "gpqa_records.tar.gz.enc", "deserted-untie-orchid"
OPENSSL = ["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-salt", "-pass", "pass:" + GPQA_PASS]


def is_gpqa(rel):
    return any("gpqa" in p for p in rel.split("/")[:-1])


def keep(rel, gpqa=False):
    parts = rel.split("/")
    if "work" in parts or is_gpqa(rel) != gpqa:
        return False
    return not any(parts[i:i + 2] == list(d) for d in SKIP_DIRS for i in range(len(parts) - 1))


def files_under(rel_dir, gpqa=False):
    out = []
    for dp, dn, fn in os.walk(os.path.join(H, rel_dir)):
        dn.sort()
        for f in sorted(fn):
            rel = os.path.relpath(os.path.join(dp, f), H)
            if keep(rel, gpqa):
                out.append(rel)
    return out


def groups():
    """(asset name, [harness-relative files]); a run's sibling <run>.meta.json goes with the run."""
    for root in EPISODE_ROOTS:
        base = os.path.join(H, root)
        for run in sorted(os.listdir(base)):
            if not os.path.isdir(os.path.join(base, run)):
                continue
            fs = files_under(os.path.join(root, run))
            meta = os.path.join(root, run + ".meta.json")
            if os.path.exists(os.path.join(H, meta)):
                fs.append(meta)
            yield f"{root.split('/')[0]}__phase0__{run}.tar.gz", fs
    for d in AUX:
        yield d.replace("/", "__") + ".tar.gz", files_under(d)
    yield GPQA_ASSET, [f for r in EPISODE_ROOTS + AUX for f in files_under(r, gpqa=True)]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(out):
    os.makedirs(out, exist_ok=True)
    manifest, lines = [], []
    for name, fs in groups():
        tar = os.path.join(out, name[:-4] if name == GPQA_ASSET else name)
        with tarfile.open(tar, "w:gz", compresslevel=6) as t:
            for rel in fs:
                t.add(os.path.join(H, rel), arcname=rel, recursive=False)
                lines.append(f"{sha256(os.path.join(H, rel))}  {rel}")
        if name == GPQA_ASSET:
            subprocess.run(OPENSSL + ["-in", tar, "-out", os.path.join(out, name)], check=True)
            os.remove(tar)
        manifest.append({"asset": name, "files": len(fs), "bytes_uncompressed": sum(os.path.getsize(os.path.join(H, r)) for r in fs),
                         "bytes": os.path.getsize(os.path.join(out, name)), "sha256": sha256(os.path.join(out, name))})
        print(f"{name}: {len(fs)} files, {manifest[-1]['bytes'] / 1e6:.1f} MB", flush=True)
    json.dump(manifest, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    open(os.path.join(out, "FILES.sha256"), "w").write("\n".join(lines) + "\n")
    print(f"{len(manifest)} assets, {sum(m['bytes'] for m in manifest) / 1e9:.2f} GB, {len(lines)} files")


def verify(out, extracted=None):
    """Asset hashes and member lists against the manifest; with EXTRACTED, every file's hash against FILES.sha256."""
    manifest = json.load(open(os.path.join(out, "MANIFEST.json")))
    want = dict(reversed(l.split("  ", 1)) for l in open(os.path.join(out, "FILES.sha256")).read().splitlines())
    bad = 0
    for m in manifest:
        p = os.path.join(out, m["asset"])
        if sha256(p) != m["sha256"]:
            print("asset hash mismatch", m["asset"]); bad += 1
        if m["asset"] == GPQA_ASSET:
            with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
                subprocess.run(OPENSSL + ["-d", "-in", p, "-out", tmp.name], check=True)
                with tarfile.open(tmp.name) as t:
                    names = [x.name for x in t.getmembers() if x.isfile()]
        else:
            with tarfile.open(p) as t:
                names = [x.name for x in t.getmembers() if x.isfile()]
        if len(names) != m["files"] or any(n not in want for n in names):
            print("member list mismatch", m["asset"]); bad += 1
    if extracted:
        for rel, h in want.items():
            p = os.path.join(extracted, rel)
            if not os.path.exists(p) or sha256(p) != h:
                print("file mismatch", rel); bad += 1
    print(f"verified {len(manifest)} assets, {len(want)} files: {'OK' if not bad else f'{bad} problems'}")
    return bad


if __name__ == "__main__":
    cmd, out = sys.argv[1], sys.argv[2]
    if cmd == "build":
        build(out)
    else:
        sys.exit(1 if verify(out, sys.argv[3] if len(sys.argv) > 3 else None) else 0)
