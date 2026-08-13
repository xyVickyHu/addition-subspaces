#!/bin/bash
#SBATCH --job-name=fv-step1-train
#SBATCH --gres=gpu:A100:1
#SBATCH --time=3:00:00
#SBATCH --output=log/slurm/step1_train_%j.out
#
# Train-only Step-1 leg for sweep cells that retrain a matrix: runs ONLY the
# matrix substep so the cell's calib and full scan legs can reuse the trained
# matrix by path (matrix trains exactly once per cell).
#
#   sbatch -p YOUR_PARTITION scripts/step1_train_only.sbatch <config.yaml>
set -euo pipefail

CONFIG="${1:?usage: sbatch scripts/step1_train_only.sbatch <config.yaml>}"

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"

# Point HF_HOME at a cache that already contains the pinned model
# (see examples/slurm/run_pipeline.sbatch):
# export HF_HOME=/path/to/huggingface_cache
# export TRANSFORMERS_CACHE="$HF_HOME"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN || true
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_SILENT=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[step1_train_only] config=$CONFIG root=$ROOT HF_HOME=${HF_HOME:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

/usr/bin/time -v .venv/bin/python -m subspaces.runners.step1 train --config "$CONFIG"
