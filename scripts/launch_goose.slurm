#!/bin/bash
# goose: SLURM. Submit with
#     sbatch --export=ALL,OUTPUT=$HOME/runs/campaign_v1 scripts/launch_goose.slurm
#
# REQUIRED EDITS before the first submission: set --partition and --account to
# values your allocation actually has. Left blank on purpose so a copied file
# cannot silently land on someone else's queue.
#
#SBATCH --job-name=annealctrl-campaign
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=annealctrl-campaign-%j.log
##SBATCH --partition=EDIT_ME
##SBATCH --account=EDIT_ME
#
# No --gres=gpu line: the G2 frontier and G3 intervention stages are CPU
# propagation work. Add one only if you have profiled a GPU stage as faster.
set -euo pipefail

VENV="${VENV:-$HOME/anneal_control/.venv}"
REPO="${REPO:-$HOME/anneal_control}"

if [ -z "${OUTPUT:-}" ]; then
    echo "error: OUTPUT is required." >&2
    echo "  sbatch --export=ALL,OUTPUT=\$HOME/runs/campaign_v1 scripts/launch_goose.slurm" >&2
    exit 2
fi
[ -x "$VENV/bin/python" ] || { echo "error: no interpreter at $VENV/bin/python" >&2; exit 1; }
[ -d "$REPO" ] || { echo "error: repository not found at $REPO" >&2; exit 1; }

# One BLAS thread per process; SLURM's cpus-per-task is for the parent workers,
# not for oversubscribed linear algebra.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
# Only relevant if a CUDA stage is added later; harmless otherwise.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "$REPO"
mkdir -p "$OUTPUT"
echo "job ${SLURM_JOB_ID:-local} on $(hostname); output=$OUTPUT"

PYTHON="$VENV/bin/python" bash scripts/run_campaign.sh --output "$OUTPUT" "${CAMPAIGN_ARGS:-}"

echo "done. Resubmit the same OUTPUT to resume after a time limit."
