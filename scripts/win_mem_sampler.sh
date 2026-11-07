#!/usr/bin/env bash
# Run ON a WSL node beside a local GPU job: every 15 s, one TSV row of Windows memory counters (commit, physical,
# page file, hard page-ins, GPU dedicated/shared memory, vmmem working set) plus WSL MemAvailable. Also a guard:
# per-project attribution goes to <out>_sources.jsonl (mem_by_project.py). Guard:
# kills process group <pgid> if commit free < MIN_COMMIT_GB or Windows available RAM < MIN_AVAIL_MB.
#   win_mem_sampler.sh <out.tsv> [pgid]
set -uo pipefail
out=$1; pg=${2:-}
MIN_COMMIT_GB=${MIN_COMMIT_GB:-8}; MIN_AVAIL_MB=${MIN_AVAIL_MB:-1500}
PS=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
read -r -d '' Q <<'PSEOF'
$c = Get-Counter -Counter '\Memory\Committed Bytes','\Memory\Commit Limit','\Memory\Available MBytes','\Paging File(_Total)\% Usage','\Memory\Pages Input/sec','\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage','\Process(vmmem*)\Working Set'
$v = @{}; foreach ($s in $c.CounterSamples) { $k = ($s.Path -split '\\')[-1]; if ($s.Path -match 'gpu adapter') { if (-not $v.ContainsKey($k) -or $s.CookedValue -gt $v[$k]) { $v[$k] = $s.CookedValue } } else { $v[$k] = $s.CookedValue } }
'{0:N1}|{1:N0}|{2:N0}|{3:N0}|{4:N2}|{5:N2}|{6:N2}' -f (($v['commit limit'] - $v['committed bytes'])/1GB), $v['available mbytes'], $v['% usage'], $v['pages input/sec'], ($v['dedicated usage']/1GB), ($v['shared usage']/1GB), ($v['working set']/1GB)
PSEOF
echo -e "time\tcommit_free_gb\twin_avail_mb\tpagefile_pct\tpages_in_per_s\tgpu_dedicated_gb\tgpu_shared_gb\tvmmem_ws_gb\twsl_avail_mb" > "$out"
while true; do
  row=$("$PS" -NoProfile -Command "$Q" 2>/dev/null | tr -d '\r' | tr -d ',' | tr '|' '\t')
  wa=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  echo -e "$(date +%H:%M:%S)\t$row\t$wa" >> "$out"
  python3 -I "$(dirname "$0")/mem_by_project.py" >> "${out%.tsv}_sources.jsonl" 2>/dev/null   # attribution by project
  cf=$(echo "$row" | cut -f1 | cut -d. -f1); av=$(echo "$row" | cut -f2)
  if [ -n "$pg" ] && [ -n "$cf" ] && { [ "$cf" -lt "$MIN_COMMIT_GB" ] || [ "${av:-99999}" -lt "$MIN_AVAIL_MB" ]; }; then
    echo "$(date +%H:%M:%S) GUARD: commit free ${cf} GB / avail ${av} MB; killing pgid $pg" >> "$out"; kill -TERM -"$pg"; exit
  fi
  [ -n "$pg" ] && ! kill -0 -"$pg" 2>/dev/null && exit
  sleep 15
done
