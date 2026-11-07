"""Throughput of every successful forward pass of the given runs, with the target GPU it ran on.
usage: python3 -I scripts/pass_throughput.py <run dir> [...]
GPU per pass: the run's .meta.json start entry (host) in effect when the pass's turn folder was written."""
import datetime, glob, json, os, sys

HOST_GPU = {  # from notes/compute_log.md and logs/podsetup_*.log
    "149.36.1.159": "5090", "213.192.6.107": "5090", "217.138.104.123": "5090", "213.173.105.177": "5090",
    "103.196.86.173": "PRO6000", "69.8.146.87": "PRO6000", "205.196.144.34": "PRO6000", "103.196.86.113": "PRO6000",
    "216.81.245.80": "6000Ada", "47.47.180.50": "4090", "69.145.85.73": "4090", "64.247.206.219": "L40S"}


def segments(run):
    out = []
    for e in json.load(open(run.rstrip("/") + ".meta.json")):
        t = datetime.datetime.fromisoformat(e["started"]).timestamp()
        g = e["target_gpu"]
        out.append((t, "local" if g == "local" else HOST_GPU.get(g["host"].split("@")[-1], g["host"])))
    return sorted(out)


def main():
    print("run\tturn\tgpu\ttype\tmin\tgens\tmean_tok\tMtok\tktok/s\tacts")
    for run in sys.argv[1:]:
        seg = segments(run)
        for l in open(os.path.join(run, "tool_calls.jsonl")):
            r = json.loads(l)
            if not str(r.get("kind", "")).startswith("forward") or not r.get("ok"):
                continue
            td = os.path.join(run, "turns", f"t{int(r['turn']):03d}")
            t = os.path.getmtime(td) if os.path.exists(td) else 0
            gpu = [g for s, g in seg if s <= t + 60]
            gpu = gpu[-1] if gpu else "?"
            s = r.get("summary") or {}
            req = json.loads(r["request"]) if r["request"].strip().startswith("{") else {}
            typ = s.get("type") or req.get("type") or "?"
            inner = s.get("summary") or {}
            gens, mt = inner.get("generations"), inner.get("mean_tokens")
            if typ == "generate" and gens and mt:
                mtok = gens * mt / 1e6
                print(f"{os.path.basename(run)}\t{r['turn']}\t{gpu}\t{typ}\t{r['seconds']/60:.1f}\t{gens}\t{mt:.0f}\t{mtok:.1f}\t{mtok*1e3/r['seconds']:.2f}\t{'y' if req.get('activations') else 'n'}")
            else:
                print(f"{os.path.basename(run)}\t{r['turn']}\t{gpu}\t{typ}\t{r['seconds']/60:.1f}")


if __name__ == "__main__":
    main()
