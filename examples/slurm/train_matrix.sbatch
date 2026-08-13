#!/bin/bash
#SBATCH --job-name=fv-train-matrix
#SBATCH --partition=YOUR_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
##SBATCH --account=YOUR_ACCOUNT

# Phase 1 — train the head-coefficient matrix from scratch with the paper's
# canonical settings (Llama-3-8B-Instruct, add-k, lambda = 0.05, coefficients
# clipped to [0,1], 50 epochs, seed 42). Edit the #SBATCH header above for
# your cluster, then submit FROM THE REPOSITORY ROOT:
#
#   sbatch examples/slurm/train_matrix.sbatch [extra subspaces.runners.train_matrix args...]
#
# Checkpoints land in log/<version>/checkpoints/. The --version string below
# is the canonical name that subspaces.runners.run_pipeline's Stage 1 auto-discovers,
# so a subsequent pipeline run reuses this matrix without --reuse-matrix.

set -euo pipefail

# Resolve the repository root. Under Slurm the batch script is copied to a
# spool directory, so ${BASH_SOURCE[0]} cannot be trusted there; prefer the
# submit directory (hence: submit from the repo root). The script-relative
# fallback (examples/slurm/ -> repo root) covers direct local execution.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

# Activate the project environment (see README: python -m venv .venv && pip install -e ".[dev]"):
# source .venv/bin/activate

# --- HuggingFace cache pinning (offline compute nodes) ------------------------
# If your compute nodes have no internet access, pre-download
# meta-llama/Meta-Llama-3-8B-Instruct on a login node, point HF_HOME at the
# cache, and force offline mode. Pin the legacy TRANSFORMERS_CACHE to the SAME
# directory as HF_HOME: model snapshots live there in a models-at-root layout
# and transformers honors TRANSFORMERS_CACHE first, so an inherited
# login-profile value otherwise wins and offline loads fail with
# LocalEntryNotFoundError (config.json).
# export HF_HOME=/path/to/huggingface_cache
# export TRANSFORMERS_CACHE=/path/to/huggingface_cache
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

export TOKENIZERS_PARALLELISM=false
# train_matrix imports wandb; without WANDB_MODE=offline it tries to log in
# and can hang a batch job. Set WANDB_ENTITY to log under a specific entity.
export WANDB_MODE=offline
export WANDB_SILENT=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader || true)"
echo "CWD: $(pwd)"
echo "Python: $(which python)"
echo "Start: $(date -Is)"

# -u: unbuffered stdout/stderr so a crash mid-run does not swallow the traceback.
python -u -m subspaces.runners.train_matrix \
  --version '[0,1]_matrix_add_meta-llama-3-8b-instruct_lambda0.05' \
  --task_dir number_add \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --layer_name blocks.10.hook_resid_mid \
  --lambdaL1 0.05 --epoch_num 50 --n_shot 5 --n_example 100 \
  --train_ratio 0.8 --bs 128 --zero_one_interval \
  "$@"

echo "End: $(date -Is)"
