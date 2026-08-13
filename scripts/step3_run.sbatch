#!/bin/bash
#SBATCH --job-name=fv-step3
#SBATCH --gres=gpu:A100:1
#SBATCH --time=1:00:00
#SBATCH --output=log/slurm/step3_%j.out
#
# Thin step-3 token-group decomposition launcher (one forward pass per
# prompt; directions from the context's z cache; heads from a Step-1
# heads artifact).
#
#   sbatch -p YOUR_PARTITION scripts/step3_run.sbatch <context.yaml> <heads.json> [extra step3 args...]
set -euo pipefail
CONTEXT="${1:?usage: sbatch step3_run.sbatch <context.yaml> <heads.json> [args...]}"
HEADS="${2:?usage: sbatch step3_run.sbatch <context.yaml> <heads.json> [args...]}"
shift 2
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
echo "[step3] context=$CONTEXT heads=$HEADS extra=$*"
/usr/bin/time -v .venv/bin/python -m subspaces.runners.step3 \
  --context "$CONTEXT" --heads-artifact "$HEADS" --head-set main "$@"
