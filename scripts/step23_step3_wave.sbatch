#!/bin/bash
#SBATCH --job-name=fv-step3wave
#SBATCH --gres=gpu:A100:1
#SBATCH --time=4:00:00
#SBATCH --output=log/slurm/step3wave_%j.out
#
# Step-3 token-group wave (2026-08-04; grouping v2): one token-group
# decomposition per cell of
# configs/step23_significant_cells.tsv over the cell's significant head set
# (⊇ the unified mains — main-head distributions are read from the same
# artifact), context configs/contexts/step3_significant_<slug>.yaml.
#
#   sbatch -p YOUR_PARTITION scripts/step23_step3_wave.sbatch [odd|even|all|<slug>] [extra step3 args...]
#
# Submit FROM THE REPO ROOT at a clean commit.
set -euo pipefail

SELECT="${1:-all}"
shift || true
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

echo "[step3wave] root=$ROOT select=$SELECT extra=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

INDEX=0
N_DONE=0
while IFS=$'\t' read -r SLUG CONFIG SIGNIFICANT UNIFIED; do
  [[ -z "$SLUG" || "$SLUG" == \#* ]] && continue
  INDEX=$((INDEX + 1))
  case "$SELECT" in
    all) ;;
    odd)  [[ $((INDEX % 2)) -eq 1 ]] || continue ;;
    even) [[ $((INDEX % 2)) -eq 0 ]] || continue ;;
    *)    [[ "$SLUG" == "$SELECT" ]] || continue ;;
  esac
  CONTEXT="configs/contexts/step3_significant_${SLUG}.yaml"
  echo "[step3wave] ==== $SLUG  context=$CONTEXT"
  echo "[step3wave]      heads=$SIGNIFICANT"
  /usr/bin/time -v .venv/bin/python -m subspaces.runners.step3 \
    --context "$CONTEXT" --heads-artifact "$SIGNIFICANT" --head-set main "$@"
  N_DONE=$((N_DONE + 1))
done < configs/step23_significant_cells.tsv
echo "[step3wave] all $N_DONE cells complete (select=$SELECT)"
