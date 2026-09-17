#!/usr/bin/env bash
# Measure NumPy against CuPy across physical sizes, one JSON artifact per size.
#
# CPU timing is only valid on an unloaded machine. A busy host starves the NumPy
# arm and inflates the GPU ratio -- that is, it biases the result in favour of
# the conclusion "the GPU wins", which is exactly the direction that must not be
# assumed. The guard below refuses to run above a load threshold rather than
# producing a number that looks measured and is not.
set -euo pipefail
ROOT="${ROOT:-$HOME/annealctrl_deploy/anneal_control}"
OUT="${OUT:-$HOME/runs/backend_crossover}"
MAX_LOAD="${MAX_LOAD:-4.0}"
SIZES="${SIZES:-10 12 14 16}"

LOAD=$(awk '{print $1}' /proc/loadavg)
if awk -v l="$LOAD" -v m="$MAX_LOAD" 'BEGIN{exit !(l>m)}'; then
  echo "refusing to measure: load average $LOAD exceeds $MAX_LOAD." >&2
  echo "A starved NumPy arm inflates the GPU ratio. Wait for the host to settle," >&2
  echo "or set MAX_LOAD deliberately and record that you did." >&2
  exit 1
fi

mkdir -p "$OUT"
for q in $SIZES; do
  echo "[crossover] ${q} physical qubits (load ${LOAD})"
  "$ROOT/.venv/bin/python" -m annealctrl.workflow_cli profile-generation \
      --config "$ROOT/configs/backend_profile_${q}q.json" \
      --output "$OUT/backend_profile_${q}q.json" \
      --backends numpy cupy --workers 1
done
echo "[crossover] artifacts in $OUT"
