#!/bin/bash
#SBATCH --job-name=fv-projcausal
#SBATCH --gres=gpu:A100:1
#SBATCH --time=4:00:00
#SBATCH --output=log/slurm/projcausal_%j.out
#
# Appendix F regeneration launcher (thin wrapper):
#
#   F — subspaces.runners.run_projection_causal: onto/out-of mod-vector subspace
#       causal projections per cell of configs/step23_significant_cells.tsv
#       (results land in <significant node>/projection-causal/).
#
#   sbatch -p YOUR_PARTITION scripts/projection_causal.sbatch smoke
#   sbatch -p YOUR_PARTITION scripts/projection_causal.sbatch {add|phi4|qwen} [extra args...]
#
# Submit FROM THE REPO ROOT at a clean commit.
set -euo pipefail

MODE="${1:?usage: projection_causal.sbatch <smoke|add|phi4|qwen> [extra args...]}"
shift || true

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Point HF_HOME at a cache that already contains the pinned model
# (see examples/slurm/run_pipeline.sbatch):
# export HF_HOME=/path/to/huggingface_cache
# export TRANSFORMERS_CACHE="$HF_HOME"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN || true
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_SILENT=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[projcausal] root=$ROOT mode=$MODE extra=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

case "$MODE" in
  smoke)
    # Tiny llama functional smoke: 1 head x 1 arm x 2 tasks x 10 queries.
    /usr/bin/time -v .venv/bin/python -m subspaces.runners.run_projection_causal \
      --cell add --heads 15:2 --arms onto_unit --tasks 2 --test-limit 10 "$@"
    ;;
  add|phi4|qwen)
    /usr/bin/time -v .venv/bin/python -m subspaces.runners.run_projection_causal \
      --cell "$MODE" "$@"
    ;;
  *)
    echo "[projcausal] ERROR: unknown mode '$MODE'" >&2
    exit 1
    ;;
esac
echo "[projcausal] done (mode=$MODE)"
