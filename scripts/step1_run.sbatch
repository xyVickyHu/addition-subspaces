#!/bin/bash
#SBATCH --job-name=fv-step1
#SBATCH --gres=gpu:A100:1
#SBATCH --time=12:00:00
#SBATCH --output=log/slurm/step1_%j.out
#
# Thin Slurm launcher for the composed Step-1 run.
#
#   sbatch scripts/step1_run.sbatch configs/step1_number_add_llama3_tinysmoke.yaml
#
# Submit FROM THE REPO ROOT. Gate ladder: tiny smoke ->
# reduced-sample runtime calibration -> report command/runtime/VRAM ->
# approval -> full 100/task scan. Before submission, the resolved identity (pinned
# revision, checkpoint sha, fingerprints) must be recorded via
# `python -m subspaces.runners.step1 extract-selected --config <cfg>` (CPU).
set -euo pipefail

CONFIG="${1:?usage: sbatch scripts/step1_run.sbatch <config.yaml> [extra args]}"
shift || true

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"

# HuggingFace cache pinning: model loading is offline-only; HF_HOME must point
# at a cache that already contains the pinned revision (see
# examples/slurm/run_pipeline.sbatch), e.g.:
# export HF_HOME=/path/to/huggingface_cache
# export TRANSFORMERS_CACHE="$HF_HOME"
# offline loads need no token; keep it out of job logs/tracebacks
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN || true
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_SILENT=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[step1_run] config=$CONFIG root=$ROOT HF_HOME=${HF_HOME:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

/usr/bin/time -v .venv/bin/python -m subspaces.runners.step1 run --config "$CONFIG" "$@"
