"""One JSON line: memory inside WSL by source. Each process (own user) is assigned to a project by its working
directory (~/Research/<project>) or that of its nearest ancestor, else its command name; PSS from /proc/<pid>/smaps_rollup.
Also the page cache, MemAvailable and which processes hold the GPU (nvidia-smi under WSL cannot give per-process
memory; the total comes from the Windows counters in win_mem_sampler.sh).
usage: python3 -I scripts/mem_by_project.py"""
import json, os, re, subprocess, time

def ppid(pid):
    try:
        return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1]
    except (OSError, IndexError):
        return "0"


def project(pid):
    """The project of the process, else of its nearest ancestor (sandboxed kernels run in another cwd)."""
    p, own = str(pid), None
    for _ in range(12):
        if p in ("0", "1"):
            break
        r = project_of(p)
        if r is None:
            return own
        if not r.startswith("other"):
            return r
        own = own or r
        p = ppid(p)
    return own


def project_of(pid):
    try:
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return None
    m = re.match(r"(/Users/[^/]+/Research/[^/]+|/home/[^/]+/Research/[^/]+)", cwd)
    if m:
        return os.path.basename(m.group(1))
    try:
        return "other:" + open(f"/proc/{pid}/comm").read().strip()
    except OSError:
        return "other"

def pss_mb(pid):
    try:
        for l in open(f"/proc/{pid}/smaps_rollup"):
            if l.startswith("Pss:"):
                return int(l.split()[1]) / 1024
    except OSError:
        pass
    return 0.0

groups, gpu = {}, {}
for d in os.listdir("/proc"):
    if d.isdigit():
        p = project(d)
        if p:
            groups[p] = groups.get(p, 0) + pss_mb(d)
try:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
    for pid in out.split():
        pid = pid.strip(",")
        gpu.setdefault(project(pid) or "?", []).append(int(pid))
except Exception:
    pass
mi = {l.split(":")[0]: int(l.split()[1]) // 1024 for l in open("/proc/meminfo")}
print(json.dumps({"time": time.strftime("%H:%M:%S"), "mem_available_mb": mi["MemAvailable"], "page_cache_mb": mi["Cached"],
                  "pss_mb_by_project": {k: round(v) for k, v in sorted(groups.items(), key=lambda kv: -kv[1]) if v >= 50},
                  "gpu_processes_by_project": gpu}))
