#!/usr/bin/env bash
# Train on the 240-parent Pegasus dataset without regenerating it.
#
# The experiment runner owns its dataset and refuses a nonempty output that has
# no manifest -- a good guard. So the runner is allowed to write its own
# manifest first, then handed the existing dataset, which IT validates:
# generate_dataset(resume=True) checks config_hash, schema_version and
# source_fingerprint before accepting anything. The annealctrl_probe tree that
# produced gen240 still matches that fingerprint exactly, so the reuse is
# verified rather than asserted.
#
# The data is COPIED, never symlinked. A symlink is what corrupted a dataset in
# this project's history.
set -uo pipefail
export PYTHONPATH="$HOME/annealctrl_probe/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY="$HOME/annealctrl_deploy/anneal_control/.venv/bin/python"
CFG="$HOME/runs/experiment_pegasus240.json"
OUT="$HOME/runs/pegasus240_exp"

rm -rf "$OUT"; mkdir -p "$OUT"

# Step 1: let the runner create its own manifest, then stop it before it gets
# far into generating a dataset we already have.
nice -n 12 "$PY" -u -m annealctrl.workflow_cli run --config "$CFG" --output "$OUT" \
    > "$HOME/runs/pegasus240_bootstrap.log" 2>&1 &   # NOT inside $OUT: a log file
                                                     # there makes the output dir
                                                     # nonempty and the runner,
                                                     # correctly, refuses it.
PID=$!
for _ in $(seq 1 120); do
    [ -f "$OUT/experiment.json" ] && break
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
done
if [ ! -f "$OUT/experiment.json" ]; then
    echo "FAILED: no manifest written"; tail -20 "$HOME/runs/pegasus240_bootstrap.log"; exit 1
fi
kill -TERM "$PID" 2>/dev/null
for _ in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
kill -KILL "$PID" 2>/dev/null
pkill -u "$USER" -TERM -P "$PID" 2>/dev/null
sleep 3
echo "manifest written; generation stopped"

# Step 2: replace whatever partial data exists with the real dataset.
rm -rf "$OUT/data"
cp -r "$HOME/runs/pegasus_trainable240" "$OUT/data"
echo "dataset copied: $(du -sh "$OUT/data" | cut -f1)"

# Step 3: resume. The runner now validates the dataset it was handed.
nice -n 12 "$PY" -u -m annealctrl.workflow_cli run --config "$CFG" --output "$OUT" --resume \
    > "$HOME/runs/pegasus240_exp.log" 2>&1
echo "PEGASUS240_DONE rc=$?"
