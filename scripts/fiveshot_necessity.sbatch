#!/bin/bash
#SBATCH --job-name=fv-5shot-necessity
#SBATCH --partition=YOUR_PARTITION  # set to your cluster GPU partition
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --output=log/slurm/fiveshot_necessity_%j.out
#
# Five-shot necessity check (paper §3.2 camera-ready paragraph): mean-ablate
# the main trio / random selected 3-subsets during the clean five-shot
# forward pass of an existing headset_eval cell. Usage:
#   sbatch scripts/fiveshot_necessity.sbatch <headset_eval-manifest> [extra args]
# Submit FROM THE REPO ROOT.
set -euo pipefail

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

MANIFEST="$1"
shift
echo "[5shot-necessity] root=$ROOT manifest=$MANIFEST extra=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

/usr/bin/time -v .venv/bin/python scripts/fiveshot_necessity.py "$MANIFEST" "$@"
echo "[5shot-necessity] done"
