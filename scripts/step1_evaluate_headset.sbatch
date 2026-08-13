#!/bin/bash
#SBATCH --job-name=fv-headset
#SBATCH --gres=gpu:A100:1
#SBATCH --time=2:00:00
#SBATCH --output=log/slurm/headset_%j.out
#
# Thin single-pair evaluate-headset launcher (selector-variant final
# metrics; everything upstream reuse-hits by identity).
#
#   sbatch -p YOUR_PARTITION scripts/step1_evaluate_headset.sbatch <config.yaml> <main_heads.json>
set -euo pipefail
CONFIG="${1:?usage: sbatch step1_evaluate_headset.sbatch <config.yaml> <main_heads.json>}"
MAIN="${2:?usage: sbatch step1_evaluate_headset.sbatch <config.yaml> <main_heads.json>}"
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
echo "[headset] config=$CONFIG main=$MAIN"
/usr/bin/time -v .venv/bin/python -m subspaces.runners.step1 evaluate-headset --config "$CONFIG" --main "$MAIN"
