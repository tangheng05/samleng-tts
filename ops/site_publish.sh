#!/bin/zsh
# Refresh the project page's live status and upload it to the public gist (LaunchAgent, every 5 minutes).
#   ops/site_publish.sh [run name]     gist id in ~/samleng-ops/gist_id, needs `gh auth login`
export PATH=$HOME/.local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin
export LC_ALL=en_US.UTF-8
cd ~/code/samleng-tts || exit 1
out=outputs/site/status.json
uv run python scripts/site_status.py --run ${1:-distill_v2_clear} --out $out || exit 1
gh api -X PATCH gists/$(<~/samleng-ops/gist_id) -F "files[status.json][content]=@$out" --silent
