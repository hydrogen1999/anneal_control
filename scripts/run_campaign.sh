#!/usr/bin/env bash
# Portable G2+G3 campaign driver. Called by launch_apollo.sh and launch_goose.slurm,
# and safe to run directly. Every stage is individually resumable, so re-running
# this script after an interruption continues rather than restarting.
#
# It never assumes CUDA. `annealctrl doctor` is recorded first and its verdict is
# reported, but no stage here requires a GPU: the control-family search and the
# paired interventions are CPU propagation work.
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
usage: run_campaign.sh --output DIR [options]

  --output DIR            required; campaign root (created if absent)
  --data-config PATH      dataset config        (default configs/data_research.json)
  --frontier-config PATH  G2 frontier config    (default configs/frontier_research.json)
  --intervention-config PATH  G3 plan config    (default configs/intervention_research.json)
  --splits "a b"          frontier splits       (default "train validation")
  --venue NAME            figure geometry       (default neurips)
  --dry-run               plan every stage and execute none
  --skip-data             reuse an existing DIR/data
  --no-figures            skip figure rendering

Test-split control search is ONLINE ADAPTATION and is deliberately not part of
this script. Run it explicitly with --allow-test-adaptation when you mean to,
and report the result as adapted rather than zero-shot.
USAGE
    exit 2
}

OUTPUT=""
DATA_CONFIG="configs/data_research.json"
FRONTIER_CONFIG="configs/frontier_research.json"
INTERVENTION_CONFIG="configs/intervention_research.json"
SPLITS="train validation"
VENUE="neurips"
DRY_RUN=0
SKIP_DATA=0
FIGURES=1

while [ $# -gt 0 ]; do
    case "$1" in
        --output) OUTPUT="${2:-}"; shift 2 ;;
        --data-config) DATA_CONFIG="${2:-}"; shift 2 ;;
        --frontier-config) FRONTIER_CONFIG="${2:-}"; shift 2 ;;
        --intervention-config) INTERVENTION_CONFIG="${2:-}"; shift 2 ;;
        --splits) SPLITS="${2:-}"; shift 2 ;;
        --venue) VENUE="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --skip-data) SKIP_DATA=1; shift ;;
        --no-figures) FIGURES=0; shift ;;
        -h|--help) usage ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done

# An unset or empty output root would scatter a multi-hour campaign into the
# working directory, so refuse rather than guess.
[ -n "$OUTPUT" ] || { echo "error: --output is required" >&2; usage; }

# Single-threaded BLAS: the propagation is already the bottleneck, and oversubscribed
# threads make wall times incomparable between runs and hosts.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

PYTHON="${PYTHON:-python}"
DRY=""
[ "$DRY_RUN" -eq 1 ] && DRY="--dry-run"

mkdir -p "$OUTPUT"
echo "[campaign] host=$(hostname) output=$OUTPUT dry_run=$DRY_RUN"
echo "[campaign] recording capabilities (no GPU is required by any stage)"
"$PYTHON" -m annealctrl doctor | tee "$OUTPUT/doctor.json"

if [ "$SKIP_DATA" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
    if [ -d "$OUTPUT/data" ]; then
        echo "[campaign] reusing existing dataset at $OUTPUT/data (resume)"
        "$PYTHON" -m annealctrl generate --config "$DATA_CONFIG" --output "$OUTPUT/data" --resume
    else
        echo "[campaign] generating dataset"
        "$PYTHON" -m annealctrl generate --config "$DATA_CONFIG" --output "$OUTPUT/data"
    fi
fi

if [ "$DRY_RUN" -eq 1 ] && [ ! -f "$OUTPUT/data/manifest.json" ]; then
    # A frontier plan counts actual records, so it needs the dataset. Rather than
    # generate one during a dry run, say so and continue to the stages that can
    # be planned without it.
    echo "[campaign] G2 control-sweep plan skipped: no dataset at $OUTPUT/data."
    echo "[campaign]   generate it first, or dry-run against an existing dataset with --skip-data."
    SPLITS=""
fi

for SPLIT in $SPLITS; do
    echo "[campaign] G2 control-sweep: $SPLIT"
    RESUME=""
    [ -d "$OUTPUT/frontier_$SPLIT" ] && RESUME="--resume"
    # shellcheck disable=SC2086
    "$PYTHON" -m annealctrl control-sweep \
        --data "$OUTPUT/data" --config "$FRONTIER_CONFIG" \
        --output "$OUTPUT/frontier_$SPLIT" --split "$SPLIT" $RESUME $DRY
done

if [ "$DRY_RUN" -eq 0 ]; then
    FIGURE_FLAG=""
    [ "$FIGURES" -eq 1 ] && FIGURE_FLAG="--figures --venue $VENUE"
    for SPLIT in $SPLITS; do
        echo "[campaign] G2 frontier-report: $SPLIT"
        # shellcheck disable=SC2086
        "$PYTHON" -m annealctrl frontier-report --sweep "$OUTPUT/frontier_$SPLIT" $FIGURE_FLAG
    done

    FIT_SWEEPS=""
    for SPLIT in $SPLITS; do
        case "$SPLIT" in test) ;; *) FIT_SWEEPS="$FIT_SWEEPS $OUTPUT/frontier_$SPLIT" ;; esac
    done
    if [ -n "$FIT_SWEEPS" ] && [ ! -f "$OUTPUT/screen.json" ]; then
        # Threshold is fitted on train/validation parents only; screen refuses test.
        LAST=""
        for S in $FIT_SWEEPS; do LAST="$S"; done
        echo "[campaign] G2 screening (fit on train/validation only)"
        # shellcheck disable=SC2086
        "$PYTHON" -m annealctrl screen --fit-sweep $FIT_SWEEPS --apply-sweep "$LAST" \
            --output "$OUTPUT/screen.json" || echo "[campaign] screening skipped: $?"
    fi
fi

echo "[campaign] G3 intervention-sweep"
RESUME=""
[ -d "$OUTPUT/interventions" ] && RESUME="--resume"
# shellcheck disable=SC2086
"$PYTHON" -m annealctrl intervention-sweep \
    --config "$INTERVENTION_CONFIG" --output "$OUTPUT/interventions" $RESUME $DRY

if [ "$DRY_RUN" -eq 0 ]; then
    FIGURE_FLAG=""
    [ "$FIGURES" -eq 1 ] && FIGURE_FLAG="--figures --venue $VENUE"
    echo "[campaign] G3 intervention-report"
    # shellcheck disable=SC2086
    "$PYTHON" -m annealctrl intervention-report --sweep "$OUTPUT/interventions" $FIGURE_FLAG
fi

echo "[campaign] done. Reports:"
for SPLIT in $SPLITS; do echo "  $OUTPUT/frontier_$SPLIT/report/FRONTIER.md"; done
echo "  $OUTPUT/interventions/report/INTERVENTIONS.md"
echo "[campaign] telemetry: $OUTPUT/*/telemetry.jsonl  (jq -c 'select(.event==\"unit_end\")')"
