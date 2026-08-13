#!/bin/bash
#SBATCH --job-name=fv-step1-smoke
#SBATCH --gres=gpu:A100:1
#SBATCH --time=1:30:00
#SBATCH --output=log/slurm/step1_smoke_%j.out
#
# Gate A: tiny functional GPU smoke.
# Runs the gpu-marked Lane-A equivalence tests, then the composed tiny run.
# Submit FROM THE REPO ROOT:  sbatch -p YOUR_PARTITION scripts/step1_smoke.sbatch
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"

# Point HF_HOME at a cache that already contains the pinned model
# (see examples/slurm/run_pipeline.sbatch):
# export HF_HOME=/path/to/huggingface_cache
# export TRANSFORMERS_CACHE="$HF_HOME"
# offline loads need no token; keep it out of job logs/tracebacks
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN || true
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_SILENT=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[smoke] root=$ROOT HF_HOME=${HF_HOME:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

echo "[smoke] === Lane-A GPU equivalence tests ==="
.venv/bin/pytest -m gpu tests/test_gpu_lane_a.py -v

echo "[smoke] === composed tiny run ==="
/usr/bin/time -v .venv/bin/python -m subspaces.runners.step1 run \
  --config configs/step1_number_add_llama3_tinysmoke.yaml

echo "[smoke] === artifacts ==="
find log/runs/addllama_tinysmoke__* -type f | sort
nvidia-smi --query-gpu=memory.used --format=csv || true
echo "[smoke] DONE"
