#!/usr/bin/env bash
# Make the public repository public, on the day the preprint is submitted: sync main, move tag v1.0 (and the release
# with its assets) to the synced head, switch visibility, then check anonymous access to the page, a raw file and an
# asset. Outward-facing: run only when the author says so.
#   bash scripts/publish_release.sh
set -euo pipefail
R=lemonadesaltbagel/how-small-wsp
cd "$(dirname "$0")/.."
bash scripts/sync_public.sh
gh api -X DELETE "repos/$R/git/refs/tags/v1.0"           # the release turns into a draft, assets kept
gh release edit v1.0 -R "$R" --draft=false --target main >/dev/null
echo "v1.0 -> $(gh api "repos/$R/git/refs/tags/v1.0" --jq '.object.sha[0:7]')"
gh repo edit "$R" --visibility public --accept-visibility-change-consequences
for u in "https://github.com/$R" "https://raw.githubusercontent.com/$R/main/README.md" \
         "https://github.com/$R/releases/download/v1.0/MANIFEST.json"; do
  printf '%s %s\n' "$(curl -s -o /dev/null -L -w '%{http_code}' "$u")" "$u"
done
