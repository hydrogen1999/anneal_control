#!/bin/bash
# Template only: edit partition/account/GPU resources for your cluster.
#SBATCH --job-name=annealctrl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=annealctrl-%j.log
set -euo pipefail
# Activate your already-installed project environment before submission,
# or add the cluster-approved activation line here. No sudo needed.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m annealctrl doctor --require-gpu
python -m annealctrl run --config configs/experiment_research.json --output runs/research --resume
