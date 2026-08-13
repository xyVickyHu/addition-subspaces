#!/bin/bash
#SBATCH --job-name=fv-projwave
#SBATCH --gres=gpu:A100:1
#SBATCH --time=4:00:00
#SBATCH --output=log/slurm/projwave_%j.out
#
# Step-2b projected-intervention wave (2026-08-04): for every cell of
# configs/step23_recpos_cells.tsv, two impl-v4 headset evaluations with the
# projected arm (proj_meanab; --subspace = the recpos node's step-2 artifact
# over the recpos superset): (1) the unified floor-0 MAIN set, (2) the
# recovery-positive (paired_bh q=.05) set. Set-independent baselines must
# reproduce the cells' existing impl-v4 values bit-exactly (quote rule:
# floor twins are set-identical).
#
#   sbatch -p YOUR_PARTITION scripts/step23_projected_wave.sbatch [odd|even|all|<slug>]
#
# Submit FROM THE REPO ROOT at a clean commit.
set -euo pipefail

SELECT="${1:-all}"
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

echo "[projwave] root=$ROOT select=$SELECT"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

INDEX=0
N_DONE=0
while IFS=$'\t' read -r SLUG CONFIG RECPOS UNIFIED; do
  [[ -z "$SLUG" || "$SLUG" == \#* ]] && continue
  INDEX=$((INDEX + 1))
  case "$SELECT" in
    all) ;;
    odd)  [[ $((INDEX % 2)) -eq 1 ]] || continue ;;
    even) [[ $((INDEX % 2)) -eq 0 ]] || continue ;;
    *)    [[ "$SLUG" == "$SELECT" ]] || continue ;;
  esac
  NODE_DIR="$(dirname "$RECPOS")"
  mapfile -t SUBS < <(ls "$NODE_DIR"/subspace-*.json 2>/dev/null)
  if [[ ${#SUBS[@]} -ne 1 ]]; then
    echo "[projwave] ERROR: expected exactly 1 subspace artifact in $NODE_DIR, found ${#SUBS[@]}" >&2
    exit 1
  fi
  SUBSPACE="${SUBS[0]}"
  for MAIN in "$RECPOS" "$UNIFIED"; do
    echo "[projwave] ==== $SLUG  config=$CONFIG"
    echo "[projwave]      main=$MAIN"
    echo "[projwave]      subspace=$SUBSPACE"
    /usr/bin/time -v .venv/bin/python -m subspaces.runners.step1 evaluate-headset \
      --config "$CONFIG" --main "$MAIN" --subspace "$SUBSPACE"
    N_DONE=$((N_DONE + 1))
  done
done < configs/step23_recpos_cells.tsv
echo "[projwave] all $N_DONE evaluations complete (select=$SELECT)"
