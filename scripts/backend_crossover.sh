#!/usr/bin/env bash
# Measure NumPy against CuPy across physical sizes, one JSON artifact per size.
#
# CPU timing is only valid on an unloaded machine. A busy host starves the NumPy
# arm and inflates the GPU ratio -- that is, it biases the result in favour of
# the conclusion "the GPU wins", which is exactly the direction that must not be
# assumed. The guard below refuses to run above a load threshold rather than
# producing a number that looks measured and is not.
set -euo pipefail
ANNEAL_ROOT="${ANNEAL_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="${OUT:-$ANNEAL_ROOT/runs/backend_crossover}"
ANNEAL_PYTHON="${ANNEAL_PYTHON:-$ANNEAL_ROOT/.venv/bin/python}"
MAX_LOAD="${MAX_LOAD:-4.0}"
SIZES="${SIZES:-10 12 14 16}"
REPEATS="${REPEATS:-4}"
WARMUPS="${WARMUPS:-1}"
ORDER_SEED="${ORDER_SEED:-0}"
PARITY_TOLERANCE="${PARITY_TOLERANCE:-1e-8}"
# Fix the CPU thread budget before importing numerical libraries. Explicit
# operator settings remain intact and are recorded in every arm's host status.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

LOAD=$(awk '{print $1}' /proc/loadavg)
if awk -v l="$LOAD" -v m="$MAX_LOAD" 'BEGIN{exit !(l>m)}'; then
  echo "refusing to measure: load average $LOAD exceeds $MAX_LOAD." >&2
  echo "A starved NumPy arm inflates the GPU ratio. Wait for the host to settle," >&2
  echo "or set MAX_LOAD deliberately and record that you did." >&2
  exit 1
fi

mkdir -p "$OUT"
for q in $SIZES; do
  echo "[crossover] ${q} physical qubits; each arm rechecks load"
  "$ANNEAL_PYTHON" -m annealctrl.profiling \
      --config "$ANNEAL_ROOT/configs/backend_profile_${q}q.json" \
      --output "$OUT/backend_profile_${q}q" \
      --backends numpy cupy --repeats "$REPEATS" --warmups "$WARMUPS" \
      --order-seed "$ORDER_SEED" --max-load "$MAX_LOAD" \
      --parity-tolerance "$PARITY_TOLERANCE"
done
echo "[crossover] artifacts in $OUT"
