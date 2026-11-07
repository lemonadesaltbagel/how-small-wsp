"""Experiment 8 forward-pass adapter (harness side). It finds the edited tensors from the run's ledger, sends them
with the request to a GPU machine, runs python/adapters/qwen_worker.py there and brings the results back.

The GPU machine is this computer, unless assets/exp8/remote.json (or $OPENWISH_QWEN_REMOTE) names a remote one:
  {"host": "root@1.2.3.4", "port": 12345, "key": "~/.ssh/id_rsa", "root": "/root/openwish", "python": "/root/qwen/bin/python"}
Run `qwen_pass.py --setup-remote` once to copy the model, practice and evaluation sets and the worker there.

forward pass (harness):  OPENWISH_REQ, OPENWISH_NET, OPENWISH_OUT, OPENWISH_RUN_DIR, OPENWISH_WORK, OPENWISH_PASS
scorer:                  qwen_pass.py --evaluate --sets math500 --limit 500 [--samples 1]   (OPENWISH_RUN, OPENWISH_FINAL_NET)
hold-out scorer:         qwen_pass.py --evaluate --sets mmlupro1k,gpqa,ifeval [--limit N] [--samples K]
                         (python/adapters/holdout_eval.py; defaults: all items, the authors' 1 / 4 / 2 samples;
                         data from `holdout_eval.py prepare` in assets/exp8/holdout; docs/holdout.md)
Prints one JSON line."""
import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A = os.path.join(ROOT, "assets", "exp8")
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_worker.py")
LOCAL_PY = os.path.join(ROOT, "assets", "venvs", "qwen", "bin", "python")
HOLDOUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "holdout_eval.py")
HOLDOUT_SETS = ("mmlupro1k", "gpqa", "ifeval")


def hosts_cfg():
    """The GPU hosts of remote.json: one host object, or {"hosts": [host, ...]} to split generation passes across
    several GPUs. A host is {"host", "port", "key", "root", "python", "gpu_util"?, "weight"?} or {"local": true,
    "weight"?} (this machine's GPU, used only when its memory allows). "weight" = the host's share (use its memory
    bandwidth in TB/s; generation is bandwidth bound). None: no remote.json, everything runs locally."""
    p = os.environ.get("OPENWISH_QWEN_REMOTE") or os.path.join(A, "remote.json")
    if not os.path.exists(p):
        return None
    c = json.load(open(p))
    hosts = c["hosts"] if isinstance(c, dict) and "hosts" in c else [c]
    seen = set()
    for h in hosts:
        if not h.get("local"):
            h["key"] = os.path.expanduser(h.get("key", "~/.ssh/id_rsa"))
            k = (h["host"], int(h["port"]))
            if k in seen:   # each pod runs one worker at a time (the orphan cleanup below ends any other)
                raise ValueError(f"host {k} listed twice in {p}")
            seen.add(k)
    return hosts


def remote_cfg():
    """The host for everything that is not split: the first remote host (None = local)."""
    hosts = hosts_cfg()
    if hosts is None:
        return None
    remote = [h for h in hosts if not h.get("local")]
    return remote[0] if remote else None


def ssh_base(c):
    return ["ssh", "-i", c["key"], "-p", str(c["port"]), "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=20", c["host"]]


def rsync(c, src, dst):
    e = f"ssh -i {shlex.quote(c['key'])} -p {c['port']} -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
    subprocess.run(["rsync", "-az", "--partial", "-e", e, src, dst], check=True, capture_output=True, text=True)


def edited_names(run_dir):
    names = set()
    lp = os.path.join(run_dir, "edits", "ledger.jsonl")
    if os.path.exists(lp):
        for line in open(lp):
            v = json.loads(line)
            if v.get("kind") == "edit":
                names.add(v["op"]["tensor"])
    return sorted(names)


def stage_edits(net_dir, names, dst):
    os.makedirs(dst, exist_ok=True)
    for n in names:
        shutil.copyfile(os.path.join(net_dir, n.replace("/", "__") + ".npy"), os.path.join(dst, n + ".npy"))
    h = hashlib.sha256()
    for n in names:
        h.update(n.encode())
        h.update(open(os.path.join(dst, n + ".npy"), "rb").read())
    return h.hexdigest()[:16]


def run_worker(args_list, stage, out_dir, worker=WORKER, host="default", guard=False):
    """Run the worker with `args_list` (paths as placeholders {model} {edits} {out} {practice} {data} {holdout} {req});
    returns its JSON. `worker` is qwen_worker.py or holdout_eval.py (which runs qwen_worker.py's `_one` beside it).
    `host`: a host of hosts_cfg() ({"local": true} = this machine), default the first remote host.
    `guard`: for a local run, stop it when this machine runs short of memory (see local_memory_ok)."""
    c = remote_cfg() if host == "default" else (None if host.get("local") else host)
    if c is None:
        local = {"model": os.path.join(A, "qwen35-text"), "practice": os.path.join(A, "practice"), "data": os.path.join(A, "eval"),
                 "holdout": os.path.join(A, "holdout"), "edits": stage["edits"], "out": out_dir, "req": stage.get("req", "")}
        cmd = [LOCAL_PY, worker] + [a.format(**local) for a in args_list] + ["--work", os.path.join(A, "work")]
        p = run_guarded(cmd) if guard else subprocess.run(cmd, capture_output=True, text=True)
        tail = p.stderr[-3000:]
    else:
        r = c["root"]
        tag = os.path.basename(out_dir.rstrip("/")) + "_" + stage["tag"]
        rem = {"model": f"{r}/model", "practice": f"{r}/practice", "data": f"{r}/eval", "holdout": f"{r}/holdout",
               "edits": f"{r}/edits/{stage['tag']}", "out": f"{r}/out/{tag}", "req": f"{r}/req/{tag}.json"}
        subprocess.run(ssh_base(c) + [f"mkdir -p {r}/edits/{stage['tag']} {r}/req {r}/out && rm -rf {r}/out/{tag}"], check=True, capture_output=True)
        rsync(c, stage["edits"] + "/", f"{c['host']}:{rem['edits']}/")
        if stage.get("req"):
            rsync(c, stage["req"], f"{c['host']}:{rem['req']}")
        rsync(c, WORKER, f"{c['host']}:{r}/qwen_worker.py")
        if worker != WORKER:      # the hold-out scorer, its item ids and the vendored IFEval checkers, beside qwen_worker.py
            for f in (worker, os.path.join(os.path.dirname(worker), "holdout_ids.json")):
                rsync(c, f, f"{c['host']}:{r}/{os.path.basename(f)}")
            rsync(c, os.path.join(os.path.dirname(worker), "instruction_following_eval") + "/", f"{c['host']}:{r}/instruction_following_eval/")
        argv = [c["python"], f"{r}/{os.path.basename(worker)}"] + [a.format(**rem) for a in args_list] + ["--work", f"{r}/work"]
        # the remote venv's bin must be on PATH: vLLM's kernel builds call tools installed there (ninja)
        env = f"export PATH={shlex.quote(os.path.dirname(c['python']))}:$PATH VLLM_USE_FLASHINFER_SAMPLER=0; "
        # a worker orphaned by a dropped SSH connection keeps the GPU (no pty, so no SIGHUP): end it first. One run per
        # pod. The anchored patterns match the worker processes, never this shell's own command line.
        # vLLM engine processes left behind get parent 1; an editor server's engine (same pod) keeps its parent.
        env += (f"pkill -f '^{c['python']} {r}/qwen_worker.py' ; pkill -f '^{c['python']} {r}/holdout_eval.py' ; sleep 3 ; "
                "for p in $(pgrep -f '^VLLM::EngineCore'); do [ \"$(ps -o ppid= -p $p | tr -d ' ')\" = 1 ] && kill $p ; done ; sleep 2 ; ")
        if c.get("gpu_util"):  # a GPU shared with another server (e.g. the editor on the same pod)
            env += f"export QWEN_GPU_UTIL={float(c['gpu_util'])}; "
        p = subprocess.run(ssh_base(c) + [env + " ".join(shlex.quote(x) for x in argv)], capture_output=True, text=True)
        tail = p.stderr[-3000:]
        os.makedirs(out_dir, exist_ok=True)
        try:
            rsync(c, f"{c['host']}:{rem['out']}/", out_dir + "/")
            # the results now live on this machine: free the pod's disk (teacher-forced outputs reach GBs)
            subprocess.run(ssh_base(c) + [f"rm -rf {rem['out']} {rem['req']}"], capture_output=True, timeout=120)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
    lines = [l for l in p.stdout.splitlines() if l.strip().startswith("{")]
    if not lines:
        return {"ok": False, "error": f"the GPU worker failed (exit {p.returncode}): {tail}"}
    return json.loads(lines[-1])


# ------------------------------------------------------------------ generation split across several GPUs
POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
LOCAL_LOCK = os.path.join(A, "local_gpu.lock")
# Measured on the desktop (2026-11-08): a local vLLM generation costs ~13-15 GB of Windows commit for the GPU memory
# alone, and WSL's page cache grows by the model; with too little left Windows pages hard (and runs out of commit,
# which crashes the machine). Start a local share only with this much free; stop it below the guard values.
LOCAL_START = {"wsl_avail_mb": 6000, "commit_free_gb": 24}
LOCAL_GUARD = {"wsl_avail_mb": 2500, "commit_free_gb": 6}


def wsl_avail_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 0


def win_commit_free_gb():
    """Windows commit headroom seen from WSL (None when not on WSL or unreadable)."""
    if not os.path.exists(POWERSHELL):
        return None
    q = "$c=(Get-Counter '\\Memory\\Committed Bytes','\\Memory\\Commit Limit').CounterSamples; [math]::Round(($c[1].CookedValue-$c[0].CookedValue)/1GB,1)"
    try:
        p = subprocess.run([POWERSHELL, "-NoProfile", "-Command", q], capture_output=True, text=True, timeout=60)
        return float(p.stdout.strip().replace(",", "."))
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def local_memory_ok(limits):
    if wsl_avail_mb() < limits["wsl_avail_mb"]:
        return False
    c = win_commit_free_gb()
    return c is None or c >= limits["commit_free_gb"]


def run_guarded(cmd):
    """subprocess.run for a local worker, killed (whole process group) when memory falls below LOCAL_GUARD."""
    import signal
    import threading
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    stop = threading.Event()
    killed = []

    def watch():
        while not stop.wait(20):
            if not local_memory_ok(LOCAL_GUARD):
                killed.append(True)
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                return
    t = threading.Thread(target=watch, daemon=True)
    t.start()
    out, err = proc.communicate()
    stop.set()
    if killed:
        err += "\nstopped by the local memory guard"
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def plan_split(n_items, names, weights):
    """Contiguous item ranges proportional to `weights`; each host runs every model of the pass on its range, so the
    original and edited answers to a problem in a "both" pass come from the same GPU (a clean pairing).
    Returns one list of parts [{"model", "start", "end"}] per host (empty when a host gets nothing)."""
    wsum = float(sum(weights))
    bounds, acc = [0], 0.0
    for w in weights:
        acc += w
        bounds.append(round(acc / wsum * n_items))
    return [[{"model": name, "start": a, "end": b} for name in names] if a < b else []
            for a, b in zip(bounds[:-1], bounds[1:])]


def merge_parts(host_outs, names, n_items, out):
    """Join the parts of all hosts into what one unsplit generate pass writes to `out`; returns the summary."""
    import numpy as np
    sys.path.insert(0, os.path.dirname(WORKER))
    from qwen_worker import paired, summarise
    os.makedirs(out, exist_ok=True)
    shutil.copyfile(os.path.join(host_outs[0], "items.json"), os.path.join(out, "items.json"))
    results, acts_info = {}, {}
    for name in names:
        found = []
        for ho in host_outs:
            pdir = os.path.join(ho, "parts")
            for f in os.listdir(pdir) if os.path.isdir(pdir) else []:
                if f.startswith(name + "_") and f.endswith(".jsonl"):
                    found.append((int(f[len(name) + 1:-6]), pdir, f))
        found.sort()
        recs, nxt, adirs = [], 0, []
        for start, pdir, f in found:
            if start != nxt:
                raise RuntimeError(f"split pass: items of the {name} model not contiguous at {start} (expected {nxt})")
            rows = [json.loads(l) for l in open(os.path.join(pdir, f))]
            recs += rows
            nxt = start + len(rows)
            ad = os.path.join(pdir, f"acts_{name}_{start}")
            if os.path.isdir(ad):
                adirs.append(ad)
        if nxt != n_items:
            raise RuntimeError(f"split pass: {nxt} of {n_items} generations of the {name} model came back")
        results[name] = recs
        fname = "gen.jsonl" if len(names) == 1 else f"gen_{name}.jsonl"
        with open(os.path.join(out, fname), "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        if adirs:
            dst = os.path.join(out, "gen_acts") if len(names) == 1 else os.path.join(out, "gen_acts", name)
            os.makedirs(dst, exist_ok=True)
            idx = [json.load(open(os.path.join(d, "index.json"))) for d in adirs]
            files = {}
            for k in idx[0]["files"]:
                arr = np.concatenate([np.load(os.path.join(d, k)) for d in adirs], axis=0)
                np.save(os.path.join(dst, k), arr)
                files[k] = list(arr.shape)
            rows = [dict(r, row=i) for i, r in enumerate(r for ix in idx for r in ix["rows"])]
            index = dict(idx[0], files=files, rows=rows,
                         answer_tokens_averaged=[t for ix in idx for t in ix["answer_tokens_averaged"]])
            json.dump(index, open(os.path.join(dst, "index.json"), "w"), indent=1)
            acts_info[name] = {"files": files, "layers": index["layers"], "mlp_layers": index["mlp_layers"]}
    which = "both" if len(names) > 1 else names[0]
    summary = {"ok": True, "type": "generate", "model": which,
               **({"summary": summarise(results[names[0]])} if len(names) == 1 else
                  {"original": summarise(results["original"]), "edited": summarise(results["edited"]),
                   "paired": paired(results["original"], results["edited"])})}
    summary["files"] = sorted(f for f in os.listdir(out) if not f.startswith("."))
    if acts_info:
        summary["activations"] = acts_info if len(names) > 1 else acts_info[names[0]]
    return summary


def split_generate(req, hosts, stage, out, tmp):
    """A generation pass spread over `hosts`. Failed shares are run again on a host that succeeded."""
    import fcntl
    import threading
    which = req.get("model", "edited")
    names = ["original", "edited"] if which == "both" else [which]
    n_items = (len(req.get("problems") or []) + len(req.get("prompts") or [])) * int(req.get("n", 1))
    lock = None
    use = []
    for h in hosts:
        if h.get("local"):
            fh = open(LOCAL_LOCK, "w")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)   # one local share at a time (episodes on this machine)
            except OSError:
                fh.close()
                continue
            if not local_memory_ok(LOCAL_START):
                fh.close()
                continue
            lock = fh
        use.append(h)
    if not use:
        return {"ok": False, "error": "no GPU host available"}
    plans = plan_split(n_items, names, [float(h.get("weight", 1.0)) for h in use])
    args = ["generate", "--model", "{model}", "--edits", "{edits}", "--req", "{req}", "--out", "{out}", "--practice", "{practice}"]
    res, outs = {}, {}

    def go(i, parts, host):
        rp = os.path.join(tmp, f"req_host{i}.json")
        json.dump(dict(req, _parts=parts), open(rp, "w"))
        outs[i] = os.path.join(tmp, f"host{i}")
        res[i] = run_worker(args, dict(stage, req=rp), outs[i], host=host, guard=bool(host.get("local")))

    threads = [threading.Thread(target=go, args=(i, parts, use[i])) for i, parts in enumerate(plans) if parts]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if lock:
        lock.close()
    failed = [i for i in res if not res[i].get("ok")]
    if any(r.get("ok") is False and "error" in r and "generation failed" not in r["error"] and "GPU worker failed" not in r["error"]
           for r in res.values()):
        return next(r for r in res.values() if not r.get("ok"))   # the request itself is wrong: same error everywhere
    good = [i for i in res if res[i].get("ok") and not use[i].get("local")]
    for k, i in enumerate(failed):
        if not good:
            return {"ok": False, "error": f"split pass: every host failed; first error: {res[i].get('error', '')[:2000]}"}
        j = good[0]
        rp = os.path.join(tmp, f"req_retry{k}.json")
        json.dump(dict(req, _parts=plans[i]), open(rp, "w"))
        outs[i] = os.path.join(tmp, f"retry{k}")
        r = run_worker(args, dict(stage, req=rp), outs[i], host=use[j])
        if not r.get("ok"):
            return r
    summary = merge_parts([outs[i] for i in sorted(outs)], names, n_items, out)
    summary["gpu_hosts"] = len(threads)
    return summary


TEXTS_FILE_MAX_BYTES = 64_000_000


def read_texts_file(guest_path):
    """(texts, error) for a "texts_file" request: a JSON list of strings at a path under /work (the agent's view),
    read from the run's work folder. Symlinks and paths outside /work are refused."""
    work = os.environ.get("OPENWISH_WORK")
    if not work or not isinstance(guest_path, str):
        return None, '"texts_file" must be a path under /work'
    rel = os.path.normpath(guest_path)
    if not rel.startswith("/work/"):
        return None, '"texts_file" must be a path under /work'
    host = os.path.join(work, rel[len("/work/"):])
    real_work = os.path.realpath(work)
    if os.path.islink(host) or not os.path.realpath(host).startswith(real_work + os.sep):
        return None, '"texts_file" must be a regular file under /work (no symlinks)'
    if not os.path.isfile(host):
        return None, f"no file {guest_path}"
    if os.path.getsize(host) > TEXTS_FILE_MAX_BYTES:
        return None, f"{guest_path} is larger than {TEXTS_FILE_MAX_BYTES:,} bytes"
    try:
        texts = json.load(open(host, encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        return None, f"{guest_path} is not valid JSON ({e})"
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        return None, f"{guest_path} must hold a JSON list of strings"
    return texts, None


def forward():
    req_path = os.environ["OPENWISH_REQ"]
    try:
        req = json.load(open(req_path))
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"the ```forward block must hold JSON ({e})"}
    kind = req.get("type")
    if isinstance(req, dict) and any(str(k).startswith("_") for k in req):
        return {"ok": False, "error": "request keys starting with \"_\" are reserved"}
    if kind not in ("generate", "teacher_forced"):
        return {"ok": False, "error": 'the request needs "type": "generate" or "teacher_forced"'}
    out = os.environ["OPENWISH_OUT"]
    with tempfile.TemporaryDirectory() as tmp:
        if kind == "teacher_forced" and "texts_file" in req:
            # texts written from Python to a JSON file under /work (texts with the template's special tokens)
            texts, err = read_texts_file(req.pop("texts_file"))
            if err:
                return {"ok": False, "error": err}
            if "texts" in req or "chats" in req:
                return {"ok": False, "error": 'give one of "texts", "texts_file" or "chats"'}
            req["texts"] = texts
            req_path = os.path.join(tmp, "req.json")
            json.dump(req, open(req_path, "w"))
        stage = {"edits": os.path.join(tmp, "edits"), "req": req_path}
        stage["tag"] = stage_edits(os.environ["OPENWISH_NET"], edited_names(os.environ["OPENWISH_RUN_DIR"]), stage["edits"])
        args = [kind, "--model", "{model}", "--edits", "{edits}", "--req", "{req}", "--out", "{out}", "--practice", "{practice}"]
        hosts = hosts_cfg()
        if kind == "generate" and hosts and len(hosts) > 1:
            res = split_generate(req, hosts, stage, out, tmp)
        else:
            res = run_worker(args, stage, out)
    if res.get("ok") and kind == "generate":
        # also where the published prompt says they are: /work/gen_<k>.jsonl (gen_<k>_original/_edited with "both")
        k = os.environ.get("OPENWISH_PASS", "0")
        work = os.environ.get("OPENWISH_WORK")
        for f in ("gen.jsonl", "gen_original.jsonl", "gen_edited.jsonl", "gen_acts"):
            src = os.path.join(out, f)
            if work and os.path.exists(src):
                copy_into_work(src, work, f"gen_{k}" + f[3:])  # the agent controls /work: never follow its symlinks
        if work and os.path.isdir(os.path.join(out, "gen_acts")):
            res.setdefault("activations", {})
            if isinstance(res["activations"], dict):
                res["activations"]["dir"] = f"/work/gen_{k}_acts/"
    return res


def copy_into_work(src, work, name):
    """Copy the file or folder `src` to `work`/`name`, replacing whatever the agent left at that name. The agent
    controls /work, so the copy follows no symlink: the old entry goes first (rmtree does not follow links), and
    each new entry gets created with O_EXCL and O_NOFOLLOW below a folder opened by descriptor."""
    dst = os.path.join(work, name)
    if os.path.isdir(dst) and not os.path.islink(dst):
        shutil.rmtree(dst)
    elif os.path.lexists(dst):
        os.unlink(dst)
    parent = os.open(work, os.O_RDONLY | os.O_DIRECTORY)
    try:
        _copy_at(src, parent, name)
    finally:
        os.close(parent)


def _copy_at(src, dir_fd, name):
    if os.path.isdir(src):
        os.mkdir(name, 0o755, dir_fd=dir_fd)
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dir_fd)
        try:
            for entry in sorted(os.listdir(src)):
                _copy_at(os.path.join(src, entry), fd, entry)
        finally:
            os.close(fd)
        return
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=dir_fd)
    with open(src, "rb") as f, os.fdopen(fd, "wb") as g:
        shutil.copyfileobj(f, g)


def evaluate(a):
    run = os.environ.get("OPENWISH_RUN", "")
    net = os.environ.get("OPENWISH_FINAL_NET", "")
    names = edited_names(run) if run else []
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        stage = {"edits": os.path.join(tmp, "edits")}
        os.makedirs(stage["edits"], exist_ok=True)  # an empty folder for an unedited run (the remote rsync needs one)
        stage["tag"] = stage_edits(net, names, stage["edits"]) if names else "base"
        for s in a.sets.split(","):
            out = os.path.join(run or tmp, "eval", f"qwen_{s}")
            if s in HOLDOUT_SETS:
                args = ["evaluate", "--model", "{model}", "--edits", "{edits}", "--set", s, "--data", "{holdout}", "--out", "{out}"]
                args += (["--limit", str(a.limit)] if a.limit is not None else []) + (["--samples", str(a.samples)] if a.samples else [])
                results[s] = run_worker(args, stage, out, worker=HOLDOUT)
                continue
            args = ["evaluate", "--model", "{model}", "--edits", "{edits}", "--set", s, "--data", "{data}",
                    "--limit", str(500 if a.limit is None else a.limit), "--samples", str(a.samples or 1), "--out", "{out}"]
            results[s] = run_worker(args, stage, out)
    ok = all(r.get("ok") for r in results.values())
    first = next(iter(results.values()))
    return {"ok": ok, "score": first.get("score", 0.0), "edited_tensors": names, **results}


def setup_remote():
    c = remote_cfg()
    if c is None:
        return {"ok": False, "error": "no assets/exp8/remote.json"}
    r = c["root"]
    subprocess.run(ssh_base(c) + [f"mkdir -p {r}/model {r}/practice {r}/eval"], check=True)
    for src, dst in ((os.path.join(A, "qwen35-text") + "/", f"{r}/model/"), (os.path.join(A, "practice") + "/", f"{r}/practice/"),
                     (os.path.join(A, "eval") + "/", f"{r}/eval/"), (WORKER, f"{r}/qwen_worker.py")):
        rsync(c, src, f"{c['host']}:{dst}")
    hd = os.path.join(A, "holdout")
    if os.path.exists(os.path.join(hd, "manifest.json")):      # the prepared hold-out sets and NLTK data (not raw/)
        subprocess.run(ssh_base(c) + [f"mkdir -p {r}/holdout && chmod 755 {r}/holdout"], check=True)
        for f in ("manifest.json", "mmlupro1k.jsonl", "gpqa.jsonl", "ifeval.jsonl", "nltk_data"):
            rsync(c, os.path.join(hd, f) + ("/" if f == "nltk_data" else ""), f"{c['host']}:{r}/holdout/{f}" + ("/" if f == "nltk_data" else ""))
    return {"ok": True, "remote": c["host"], "root": r}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--setup-remote", action="store_true")
    ap.add_argument("--sets", default="math500")
    ap.add_argument("--limit", type=int, default=None, help="default: 500 for math500 / gsm8k, all items for the hold-out sets")
    ap.add_argument("--samples", type=int, default=None, help="default: 1 for math500 / gsm8k, the authors' counts for the hold-out sets")
    a = ap.parse_args()
    try:
        res = setup_remote() if a.setup_remote else evaluate(a) if a.evaluate else forward()
    except Exception as e:  # noqa: BLE001 - reported to the editor, not charged
        res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(json.dumps(res))
