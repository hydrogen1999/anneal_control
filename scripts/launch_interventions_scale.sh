#!/usr/bin/env bash
# G3 interventions at 14 physical qubits, from an isolated source tree.
#
# SRC must not be a tree that a live sweep is hashing. The Pegasus frontier sweep
# runs against annealctrl_deploy and resumes by content fingerprint, so writing an
# updated module there aborts it with source_drift -- the bug that killed the
# training campaign once already. Point SRC at a copy instead.
#
# One pair costs about 650 s at 14 physical qubits (2^14 amplitudes against 2^8
# in the 8-qubit research config), so 228 pairs is roughly 41 hours of single
# threaded work and a few hours across the default eight shards.
set -uo pipefail
SRC="${SRC:-$HOME/annealctrl_probe/src}"
PY="${PY:-$HOME/annealctrl_deploy/anneal_control/.venv/bin/python}"
CONFIG="${CONFIG:-$HOME/annealctrl_probe/intervention_scale14.json}"
OUT="${OUT:-$HOME/runs/interv14}"
SHARDS="${SHARDS:-8}"
NICE="${NICE:-19}"

export PYTHONPATH="$SRC"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Check the interpreter before fanning out. Without this, an interpreter that
# cannot import the package starts N processes that die instantly into their own
# logs and the row count simply stays at zero.
if ! "$PY" -c 'import annealctrl' 2>/dev/null; then
    echo "error: '$PY' with PYTHONPATH=$SRC cannot import annealctrl" >&2
    exit 1
fi
"$PY" -c 'import annealctrl; print("annealctrl from", annealctrl.__file__)'

mkdir -p "$OUT"
for i in $(seq 0 $((SHARDS - 1))); do
    DIR="$OUT/shard_$i"
    RESUME=""
    [ -d "$DIR" ] && RESUME="--resume"
    nohup nice -n "$NICE" "$PY" -m annealctrl.workflow_cli intervention-sweep \
        --config "$CONFIG" --output "$DIR" --shard "$i" --shard-count "$SHARDS" $RESUME \
        > "$OUT/shard_$i.log" 2>&1 &
    echo "[interv14] shard $i launched"
done
echo "[interv14] $SHARDS shards launched into $OUT"
