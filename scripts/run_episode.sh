#!/usr/bin/env bash
# Run one exp8 episode on the harness machine (WSL), in the foreground (start it inside tmux).
#   run_episode.sh <task.toml (relative to harness/)> <run dir (relative to harness/)> <remote.json or "local"> [openwish run flags...]
# Records the git commit, the task hash and the target GPU in <run dir>.meta.json before starting; resumes the run
# if <run dir> already exists.
set -uo pipefail
cd "$(dirname "$0")/../harness"
task=$1; out=$2; remote=$3; shift 3
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if [ "$remote" != "local" ]; then export OPENWISH_QWEN_REMOTE="$(cd .. && pwd)/$remote"; fi
mkdir -p "$(dirname "$out")"
python3 - "$task" "$out" "$remote" "$@" <<'EOF'
import hashlib, json, subprocess, sys, datetime, os
task, out, remote, *flags = sys.argv[1:]
meta = {"started": datetime.datetime.now().astimezone().isoformat(), "task": task,
        "task_sha256": hashlib.sha256(open(task, "rb").read()).hexdigest(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "git_dirty": bool(subprocess.run(["git", "status", "--porcelain", "--", "."], capture_output=True, text=True).stdout.strip()),
        "target_gpu": "local" if remote == "local" else json.load(open(os.path.join("..", remote))),
        "flags": flags}
p = out.rstrip("/") + ".meta.json"
runs = json.load(open(p)) if os.path.exists(p) else []
runs.append(meta)
json.dump(runs, open(p, "w"), indent=1)
EOF
if [ -d "$out" ]; then
  exec target/release/openwish run --resume "$out" "$@"
else
  exec target/release/openwish run "$task" --out "$out" "$@"
fi
