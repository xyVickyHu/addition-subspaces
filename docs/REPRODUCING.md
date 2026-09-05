# Reproducing the paper

This guide covers the four analysis phases, the end-to-end orchestrator and
what it writes, the activation-cache behavior that determines run-to-run
variance, the shipped datasets and task splits, and the Slurm examples.

Reproduction targets (bootstrap-CI numbers; ±0.02–0.04 tolerance, see
[the z-cache section](#the-z-cache-activation-cache)):

| Configuration | Accuracy |
|---|---:|
| Clean 5-shot ICL baseline | 0.8567 |
| 33-head FV (`sum_33`, unit coefficients) | 0.7817 |
| 3 main heads + mean-ablation of the other 30 | 0.7017 |

The paper's corrected-protocol AIE (Todd et al.) baselines — e.g. top-33 =
0.414 on Llama-3 add-k — reproduce via the `configs/step1_*_aie.yaml` cells
with `subspaces.runners.step1`.

## The four phases

Every phase has a dedicated runner module under `subspaces/runners/`; the
`subspaces.runners.run_pipeline` orchestrator chains them for one (model, task) pair.

| Phase | Description | Entry point | Library modules |
|------:|-------------|-------------|-----------------|
| 1 | Head-selection optimization training (sparse coefficient matrix over layer×head), then selected-head extraction | `subspaces.runners.train_matrix` + `subspaces.runners.run_head_select` | `subspaces.utils.activations`, `subspaces.utils.intervene`, `subspaces.utils.data` |
| 2 | Per-head and subset FV evaluation (mean-ablation narrow-down to the main heads) | `subspaces.runners.run_head_eval`, `subspaces.runners.run_head_sum_eval` | `subspaces.utils.heads`, `subspaces.utils.intervene` |
| 3 | PCA decomposition + paper §4 six-dimensional basis (4D periodic units mod 2/5/10 + 2D magnitude) | `subspaces.runners.run_pca_decomposition` | `subspaces.utils.pca`, `subspaces.utils.activations` |
| 4 | Per-demonstration extracted-signal magnitudes/directions (§5.1/5.2 label-token peaking + signal alignment) | `subspaces.runners.run_signal_extraction` | `subspaces.utils.signals` |

Phases 1, 2, and 4 need a GPU (model forward passes). Phase 3 is CPU-only —
it works purely on cached per-task mean head activations.

Runtime dependencies in `pyproject.toml` are unpinned; the versions this
release was validated against are: `torch` 2.8.0, `transformers` 4.56.1,
`transformer-lens` 2.16.1, `scikit-learn` 1.3.0, `numpy` 1.25.0,
`pandas` 1.5.3, `matplotlib` 3.7.1 (Python 3.10). If a future dependency
release breaks something, pin these.

## End-to-end: `subspaces.runners.run_pipeline`

```bash
python -m subspaces.runners.run_pipeline \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --task_dir number_add \
  --reuse-matrix artifacts/matrix_add_0204_clip_lambda0.05
```

The orchestrator's internal stage numbering (selected via `--stages`, default
`1,2,3,4,5`) — note that internal stage 5 is just the experiments-log append,
so the four analysis phases above are stages 1–4:

1. **Matrix training or reuse.** With `--reuse-matrix <dir>` (path resolved
   relative to the repository root) training is skipped. Without it, the stage
   looks for a canonical matrix at
   `log/[0,1]_matrix_add_<model-slug>_lambda<lambdaL1>` and otherwise invokes
   `subspaces.runners.train_matrix` as a subprocess.
2. **Head selection + FV-variant evaluations.** Auto-thresholds the trained
   matrix (`--auto-threshold elbow|fraction|fixed`, default `elbow`, which
   returns the paper's 33 heads), then measures: the clean ceiling, the
   full-selected FV (`sum_33` config, unit coefficients), the raw
   trained-coefficient FV, a per-head recovery (dose-response) scan used by
   the default `--main-rank-by recovery` main-head selector, and the
   main+mean-ablation FV. `--main-heads-fallback 15:2,15:1,13:6` pins the
   paper's exact main-head trio; `--main-rank-by accuracy` (with `--main-k`)
   is the legacy paper-reproduction ranking and needs the matrix dir's
   `head_acc_dict.pth` (shipped in `artifacts/matrix_add_0204_clip_lambda0.05/`,
   regenerable with `subspaces.runners.run_head_eval`); `--main-rank-by coef` ranks
   by |coefficient|.
3. **PCA decomposition** per selected head plus the §4 period/magnitude
   basis fit on the main heads (add-k tasks only).
4. **§5 extracted-signal analysis** on the main heads (add-k only): §5.1/5.2
   label-token peaking, signal alignment, and the label-token aggregation
   analysis.
5. **Experiment logging** — appends a structured summary section (head sets,
   accuracy table, recovery decisions, PCA and signal results, and the exact
   command) to `subspaces/experiments.md`.

Other useful flags: `--test-limit` (eval samples per task, default 10),
`--rerun` (a run directory with an existing `manifest.json` is otherwise
skipped), and `--run-tag <tag>` (keeps parallel runs of the same model+task in
separate run directories).

### What each stage writes to `subspaces/runs/<slug>/`

The slug is `<model-name-lowercased>__<task_dir>` (plus `__<run-tag>` if
given), e.g. `meta-llama_meta-llama-3-8b-instruct__number_add`.

| File | Written by | Contents |
|---|---|---|
| `manifest.json` | end of run | Full run record: arguments, matrix dir, stages run, and the stage summaries. Its presence marks the run complete (skip unless `--rerun`). |
| `stage2_head_eval.json` | stage 2 | Threshold, selected/main head sets, all accuracy variants (clean, `sum_33`, raw-coefficient, main+mean-ablation, mean-FV), recovery curves and decisions. |
| `stage2_recovery_per_example.json` | stage 2 | Per-example 0/1 correctness for every (head, scale) point of the recovery scan. |
| `stage3_pca.json` | stage 3 | Per-head PCA cumulative variance, #PCs to reach 95%, and the §4 mod-vector fits (R² per direction) on the main heads. |
| `stage4_signals.json` | stage 4 | Per-main-head §5.1/5.2 statistics (label-token peaking, alignment, label-token aggregation) with bootstrap CIs. |
| `plots/stage2_recovery_dose_response.png` | stage 2 | Per-head dose-response curves; main heads highlighted. |
| `plots/pca_cumvar_L{L}H{H}.png` | stage 3 | Cumulative-variance curve per selected head. |
| `plots/attn_profile_*.png`, `plots/weighting_gain_*.png` | stage 4 | Attention-profile and label-token weighting-gain plots. |

Stage 5 does not write into the run directory; it appends to
`subspaces/experiments.md` (a stub ships inside the package directory; each run appends to it).

## The z-cache (activation cache)

All stages consume `z_results`: per-task mean attention-head outputs at the
final token, extracted once per (task family, shot count, prompt format) and
cached as

```
z_results_dict_<task_dir>_shot<n_shot><format_tag>.pth
```

- **Location.** The pipeline prefers `<matrix_dir>/savevars/` if a cache file
  already exists there, and otherwise uses `log/<task_dir>/savevars/`. The
  shipped `artifacts/` directory intentionally does not include the cache
  (it is large), so a fresh clone writes it to `log/number_add/savevars/`.
- **Auto-recompute.** If the file is absent, any GPU stage (stage 2, training,
  signal extraction) recomputes it from freshly sampled prompts and saves it.
  The CPU-only stage 3 cannot recompute; run stage 2 first (the default
  `--stages 1,2,3,4,5` does this in order).
- **Run-to-run variance.** Because a regenerated cache is built from a fresh
  random prompt sample, all downstream accuracies shift slightly — this,
  together with GPU nondeterminism, is the source of the ±0.02–0.04 tolerance
  on the reproduction targets.
- **Prompt formats.** The prompt template is selected by the
  `FV_PROMPT_FORMAT` environment variable (see `subspaces/utils/prompt_formats.py`).
  The default `arrow` format (`x->y#`) is the paper's template and keeps the
  legacy untagged cache filename; other formats get a format tag appended so
  caches never collide.

## Training the matrix from scratch

`subspaces.runners.run_pipeline` stage 1 trains with exactly these settings when no
matrix is found; running the trainer directly is equivalent:

```bash
python -m subspaces.runners.train_matrix \
  --version '[0,1]_matrix_add_meta-llama-3-8b-instruct_lambda0.05' \
  --task_dir number_add \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --layer_name blocks.10.hook_resid_mid \
  --lambdaL1 0.05 --epoch_num 50 --n_shot 5 --n_example 100 \
  --train_ratio 0.8 --bs 128 --zero_one_interval
```

This writes per-epoch checkpoints to `log/<version>/checkpoints/`; downstream
stages load the highest-epoch `matrix_epoch*.pth`. Using the canonical
`--version` string above lets `run_pipeline` discover and reuse the matrix
automatically. Training logs to Weights & Biases: set `WANDB_MODE=offline` to
avoid a login (recommended on clusters) and `WANDB_ENTITY` if you want runs
under a specific entity.

## Running phases individually

Useful for partial reruns; all `--log_dir` arguments point at a matrix
directory.

These runners write their outputs into `--log_dir`, so copy the shipped
artifact directory first and work on the copy — the files under `artifacts/`
are immutable reference artifacts:

```bash
mkdir -p log && cp -r artifacts/matrix_add_0204_clip_lambda0.05 log/matrix_add_rerun

# Phase 1 (post-training) — extract selected heads, auto-thresholded
# (--report-main-heads ranks by head_acc_dict.pth when present, else by |coef|)
python -m subspaces.runners.run_head_select \
  --log_dir log/matrix_add_rerun \
  --auto-threshold elbow --report-main-heads 3

# Phase 2 — per-head and subset FV evaluations (GPU). run_head_eval writes
# head_eval.jsonl (per-head accuracies) into --log_dir; run_head_sum_eval
# consumes it, so pass the matching --head_eval_name.
python -m subspaces.runners.run_head_eval \
  --log_dir log/matrix_add_rerun --task_dir_name number_add
python -m subspaces.runners.run_head_sum_eval \
  --log_dir log/matrix_add_rerun --task_dir_name number_add \
  --head_eval_name head_eval.jsonl \
  --top_signal accuracy --subset_signal accuracy

# Phase 3 — PCA + paper §4 six feature directions (CPU; needs the z-cache)
python -m subspaces.runners.run_pca_decomposition \
  --model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --z-results log/number_add/savevars/z_results_dict_number_add_shot5.pth \
  --heads 15:2,15:1,13:6 \
  --out-run meta-llama_meta-llama-3-8b-instruct__number_add \
  --compare-mod-vectors artifacts/matrix_add_0204_clip_lambda0.05/mod_vectors_dict.pth

# Phase 4 — §5 signal extraction on the main heads (GPU)
python -m subspaces.runners.run_signal_extraction \
  --z-results log/number_add/savevars/z_results_dict_number_add_shot5.pth \
  --heads 15:2,15:1,13:6 \
  --out-run meta-llama_meta-llama-3-8b-instruct__number_add
```

`head_acc_dict.pth` (the per-head accuracy cache behind the legacy
`--main-rank-by accuracy` path) ships precomputed with the canonical artifact;
the released runners do not regenerate it — `subspaces.runners.run_head_eval` writes
its per-head accuracies to a JSONL instead (pass `--other_signal mean` for the
paper's mean-ablation setting). For newly trained matrices, use
`run_pipeline`'s default recovery-based main-head selection, or pin heads
explicitly with `--main-heads-fallback`.
`subspaces.runners.verify_reproducibility` is a CPU-only sanity check that the
shipped matrix reproduces the paper's 33-head selected set and the
(13,6)/(15,2)/(15,1) main heads.

## Reproducing the generality appendix (App. B)

The per-cell configs under `configs/` reference run nodes under `log/runs/`
by content hash; those nodes are not shipped, but the hashes are
deterministic, so re-running a cell recreates them at the same paths. The
flow per cell: train the matrix with its `*_train.yaml` config
(`scripts/step1_train_only.sbatch`), run head selection/evaluation with the
cell's `*_siggap_avg3.yaml` and `*_aie.yaml` configs
(`scripts/step1_run.sbatch`, `scripts/step1_aie.sbatch`,
`scripts/step1_evaluate_headset.sbatch`), then build the step-2/step-3
contexts (`scripts/make_step2_context.py`, `scripts/make_step3_context.py`)
and run the subspace/label-share waves over `configs/step23_significant_cells.tsv`
(`scripts/step23_projected_wave.sbatch`, `scripts/step23_step3_wave.sbatch`,
`scripts/step23_label_share_summary.py`). The Appendix-E onto/out-of
projection arms run via `subspaces.runners.run_projection_causal`
(`scripts/projection_causal.sbatch`).

## Datasets

The five task families used in the paper ship pre-built under
`dataset_files/`, each with its fixed train/held-out split committed in
`configs/task_splits/*_paper.yaml` (the splits were fixed once and are never
regenerated; no dataset-generation scripts ship with the release):

| Family | Directory | Split file |
|---|---|---|
| Addition (add-k, k ∈ [1,30], x ∈ [1,100]) | `dataset_files/number_add/` | `number_add_paper.yaml` |
| Subtraction (add-k with k ∈ [-30,0], x ∈ [1,100]) | `dataset_files/number_add[-30,0]_[1,100]/` | `number_add_subtract_paper.yaml` |
| Multiplication (mul-k, k ∈ [1,30], x ∈ [1,100]) | `dataset_files/number_mul/` | `number_mul_paper.yaml` |
| Extractive (word-list + CoNLL-2003 entity tasks) | `dataset_files/extractive/` | `extractive_paper.yaml` |
| Abstractive (the Todd et al. tasks) | `dataset_files/abstractive/` | `abstractive_paper.yaml` |

Every task file is a JSON list of `{"input": "<x>", "output": "<f(x)>"}`
string pairs; a new family is just a directory of such files, selected with
`--task_dir <dirname>`. A tiny smoke-test fixture
(`dataset_files/number_add_tiny/`, split `number_add_tiny.yaml`) backs the
`*_tinysmoke` configs.

## Slurm examples

`examples/slurm/` contains two parameterized batch scripts:

- `run_pipeline.sbatch` — runs the full pipeline against the
  shipped paper matrix. Extra arguments after the script name are forwarded to
  `subspaces.runners.run_pipeline`, e.g.
  `sbatch examples/slurm/run_pipeline.sbatch --main-heads-fallback 15:2,15:1,13:6 --rerun`.
- `train_matrix.sbatch` — trains the coefficient matrix from scratch with the
  paper's canonical settings; extra arguments are forwarded to
  `subspaces.runners.train_matrix`.

Before first use, edit the `#SBATCH` header (partition/account/GPU type for
your cluster) and the conda-activation line. Submit from the repository root —
under Slurm the batch script is copied to a spool directory, so the scripts
resolve the project root from `SLURM_SUBMIT_DIR` (with a script-relative
fallback for local execution). Both scripts document the HuggingFace offline
cache pinning needed on clusters whose compute nodes have no internet access.
