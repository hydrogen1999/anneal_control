#!/usr/bin/env bash
# Trainable dataset on real Pegasus connectivity, 10-14 physical qubits.
#
# CPU with many workers, not the GPU. CuPy is capped at one parent worker and
# measured 7.41 labels/s at 12 qubits against NumPy's 2.62, so eight CPU workers
# reach roughly 21/s and beat one GPU worker by about 3x on this workload. That
# came from reports/backend_2026-09-17/crossover.json, not from an assumption --
# the earlier assumption pointed the other way.
#
# SRC defaults to an isolated tree. A live sweep resumes by content fingerprint,
# so writing into the tree it hashes aborts it with source_drift.
set -uo pipefail
SRC="${SRC:-$HOME/annealctrl_probe/src}"
PY="${PY:-$HOME/annealctrl_deploy/anneal_control/.venv/bin/python}"
CONFIG="${CONFIG:-$HOME/annealctrl_probe/data_pegasus_trainable.json}"
OUT="${OUT:-$HOME/runs/pegasus_trainable_data}"
WORKERS="${WORKERS:-8}"

export PYTHONPATH="$SRC"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$PY" -c 'import annealctrl; print("annealctrl from", annealctrl.__file__)' || exit 1

# Note the entrypoint: `generate` lives on the `annealctrl` module, not on
# `annealctrl.workflow_cli`, which carries the measurement subcommands.
nice -n 19 "$PY" -m annealctrl generate \
    --config "$CONFIG" --output "$OUT" --backend numpy --workers "$WORKERS"
echo "[gen] done: $OUT"
