#!/usr/bin/env bash
# apollo: lab machine, no scheduler. Detaches with nohup so the campaign survives
# the SSH session, and writes one log per launch.
#
# Edit VENV to point at the environment you installed the package into. Nothing
# here needs sudo, and nothing here requires a GPU.
set -euo pipefail

VENV="${VENV:-$HOME/anneal_control/.venv}"
REPO="${REPO:-$HOME/anneal_control}"
OUTPUT="${1:-}"
shift || true

if [ -z "$OUTPUT" ]; then
    cat >&2 <<'USAGE'
usage: launch_apollo.sh OUTPUT_DIR [run_campaign.sh options]

  launch_apollo.sh ~/runs/campaign_v1
  launch_apollo.sh ~/runs/campaign_v1 --dry-run
  launch_apollo.sh ~/runs/campaign_v1 --frontier-config configs/frontier_smoke.json

Re-running the same OUTPUT_DIR resumes; it never restarts from scratch.
USAGE
    exit 2
fi

[ -x "$VENV/bin/python" ] || { echo "error: no interpreter at $VENV/bin/python; set VENV=" >&2; exit 1; }
[ -d "$REPO" ] || { echo "error: repository not found at $REPO; set REPO=" >&2; exit 1; }

mkdir -p "$OUTPUT"
LOG="$OUTPUT/campaign_$(date +%Y%m%d_%H%M%S).log"

cd "$REPO"
echo "launching detached campaign on $(hostname)"
echo "  output : $OUTPUT"
echo "  log    : $LOG"
PYTHON="$VENV/bin/python" nohup bash scripts/run_campaign.sh --output "$OUTPUT" "$@" \
    > "$LOG" 2>&1 &
echo "  pid    : $!"
echo
echo "follow:   tail -f $LOG"
echo "progress: jq -c 'select(.event==\"unit_end\")' $OUTPUT/*/telemetry.jsonl | tail"
echo "stop:     kill $!   (then re-launch the same OUTPUT_DIR to resume)"
