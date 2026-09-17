#!/usr/bin/env bash
# Run one sweep as N concurrent shards on a machine with no scheduler (apollo).
#
# The sweep driver is single-process by design, so parallelism comes from
# sharding: each shard writes its own directory and the reporter merges them.
# Merging shards computed under different settings is refused, so a half-finished
# re-parameterised campaign cannot be silently stitched together.
#
# This is CPU work. CuPy propagation is restricted to one worker, and at the
# physical sizes this project uses it measured only ~1.08x per worker, so N CPU
# shards beat one GPU process by roughly N. Use the GPU for sizes where it has
# been profiled to win, not by default.
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
usage: launch_shards.sh --kind intervention|frontier --output DIR --shards N [options]

  --kind KIND        intervention | frontier
  --output DIR       required; shard directories are created under it
  --shards N         number of concurrent processes
  --config PATH      sweep config (required)
  --data DIR         dataset directory (frontier only)
  --split NAME       frontier split (default validation)
  --nice N           nice level for every shard (default 10; shared machine)
  --allow-test-adaptation
                     required when --split test. Searching controls on the test
                     split consults true outcomes per instance, which is online
                     adaptation and a different cost class from any amortised
                     method. Every row is labelled online_adaptation=true.
  --dry-run          print the commands without running them

Re-running the same --output resumes each shard. Merge afterwards with, e.g.
  python -m annealctrl intervention-report --sweep DIR/shard_* --figures
USAGE
    exit 2
}

KIND=""; OUTPUT=""; SHARDS=""; CONFIG=""; DATA=""; SPLIT="validation"; NICE=10; DRY=0
ADAPT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --kind) KIND="${2:-}"; shift 2 ;;
        --output) OUTPUT="${2:-}"; shift 2 ;;
        --shards) SHARDS="${2:-}"; shift 2 ;;
        --config) CONFIG="${2:-}"; shift 2 ;;
        --data) DATA="${2:-}"; shift 2 ;;
        --split) SPLIT="${2:-}"; shift 2 ;;
        --nice) NICE="${2:-}"; shift 2 ;;
        --allow-test-adaptation) ADAPT="--allow-test-adaptation"; shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) usage ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done

[ -n "$KIND" ] && [ -n "$OUTPUT" ] && [ -n "$SHARDS" ] && [ -n "$CONFIG" ] || usage
case "$KIND" in intervention|frontier) ;; *) echo "unknown --kind $KIND" >&2; usage ;; esac
[ "$KIND" = "frontier" ] && [ -z "$DATA" ] && { echo "error: --data is required for --kind frontier" >&2; exit 2; }
case "$SHARDS" in ''|*[!0-9]*) echo "error: --shards must be a positive integer" >&2; exit 2 ;; esac
# Mirror the CLI's own guard here rather than letting N shards discover it
# separately: a test-split search is an explicit act, and a launcher that makes
# it implicit defeats the point of the flag existing.
if [ "$KIND" = "frontier" ] && [ "$SPLIT" = "test" ] && [ -z "$ADAPT" ]; then
    echo "error: --split test requires --allow-test-adaptation." >&2
    echo "       Searching controls on held-out records is online adaptation, not a free" >&2
    echo "       ceiling; it must be a deliberate choice and is reported as such." >&2
    exit 2
fi
[ "$SHARDS" -ge 1 ] || { echo "error: --shards must be >= 1" >&2; exit 2; }

CORES="$(nproc 2>/dev/null || echo 1)"
if [ "$SHARDS" -gt "$CORES" ]; then
    echo "warning: $SHARDS shards on $CORES cores will oversubscribe and make timings incomparable" >&2
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
PYTHON="${PYTHON:-python}"

# Check the interpreter BEFORE launching N detached shards. Without this, a
# PYTHON that cannot import the package starts N processes that each die
# instantly into their own log, the row count stays at zero, and the failure
# looks like slow progress. That cost an hour of wall clock once.
if ! "$PYTHON" -c "import annealctrl" 2>/dev/null; then
    echo "error: '$PYTHON' cannot import annealctrl." >&2
    echo "       Set PYTHON to the interpreter of the environment the package is installed in," >&2
    echo "       e.g. PYTHON=\$HOME/anneal_control/.venv/bin/python $0 ..." >&2
    exit 1
fi

mkdir -p "$OUTPUT"

echo "launching $SHARDS shards of the $KIND sweep"
echo "  output : $OUTPUT/shard_*"
echo "  cores  : $CORES (load $(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo '?'))"

PIDS=""
for INDEX in $(seq 0 $((SHARDS - 1))); do
    DIR="$OUTPUT/shard_$INDEX"
    RESUME=""
    [ -d "$DIR" ] && RESUME="--resume"
    if [ "$KIND" = "intervention" ]; then
        set -- -m annealctrl intervention-sweep --config "$CONFIG" --output "$DIR" \
               --shard "$INDEX" --shard-count "$SHARDS" $RESUME
    else
        set -- -m annealctrl control-sweep --data "$DATA" --config "$CONFIG" --output "$DIR" \
               --split "$SPLIT" --shard "$INDEX" --shard-count "$SHARDS" $RESUME $ADAPT
    fi
    if [ "$DRY" -eq 1 ]; then
        echo "  would run: nice -n $NICE $PYTHON $*"
        continue
    fi
    nice -n "$NICE" "$PYTHON" "$@" > "$OUTPUT/shard_$INDEX.log" 2>&1 &
    PIDS="$PIDS $!"
done

[ "$DRY" -eq 1 ] && exit 0

echo "  pids   :$PIDS"
FAILED=0
for PID in $PIDS; do
    wait "$PID" || FAILED=$((FAILED + 1))
done

if [ "$FAILED" -gt 0 ]; then
    echo "$FAILED shard(s) exited nonzero; inspect $OUTPUT/shard_*.log and re-run to resume" >&2
    exit 1
fi

echo "all $SHARDS shards finished. Merge with:"
if [ "$KIND" = "intervention" ]; then
    echo "  $PYTHON -m annealctrl intervention-report --sweep $OUTPUT/shard_* --figures"
else
    echo "  $PYTHON -m annealctrl frontier-report --sweep $OUTPUT/shard_* --figures"
fi
