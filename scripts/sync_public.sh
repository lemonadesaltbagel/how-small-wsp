#!/usr/bin/env bash
# Push this repository's main, minus the private planning files (and private/: plans of experiments not yet reported), to the public repository. The filter is the same on
# every run, so earlier commits keep their hashes and the push is a fast-forward.
#   bash scripts/sync_public.sh            (needs gh / git credentials for both repositories)
set -euo pipefail
PRIVATE=https://github.com/lemonadesaltbagel/wsp-scale-ladder.git
PUBLIC=https://github.com/lemonadesaltbagel/how-small-wsp.git
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git clone -q --bare "$PRIVATE" "$tmp/repo.git"
cd "$tmp/repo.git"
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --prune-empty --index-filter \
  'git rm -r -q --cached --ignore-unmatch -- tracker_proposals private notes/plan_sim_pilot.md wsp_scale_ladder_research_brief.md "WSP 尺度阶梯项目进度追踪.md" "Weight-Space Programming 的尺度阶梯与自指闭环：项目初步构思.md" "预印本大纲_校对意见.md" "预印本大纲：How Small Can a Weight-Space Programmer Be.md"' \
  -- main >/dev/null 2>&1
n=$(git rev-list --objects main | grep -c -E '^private/|tracker_proposals|plan_sim_pilot|research_brief|进度追踪|初步构思|预印本大纲' || true)
[ "$n" = 0 ] || { echo "private paths still present ($n objects); not pushing" >&2; exit 1; }
git push "$PUBLIC" main
git log -1 --format='pushed %h %s' main
