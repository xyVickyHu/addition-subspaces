# Understanding In-context Learning of Addition via Activation Subspaces

[![arXiv](https://img.shields.io/badge/arXiv-2505.05145-b31b1b.svg)](https://arxiv.org/abs/2505.05145)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[Camera-ready PDF](paper.pdf)

Code for the COLM 2026 paper
**[Understanding In-context Learning of Addition via Activation Subspaces](https://arxiv.org/abs/2505.05145)**
by Xinyan Hu, Kayo Yin, Michael I. Jordan, Jacob Steinhardt, and Lijie Chen.

## What the paper shows

To perform in-context learning, a language model must extract signals from its
few-shot demonstrations, aggregate them into a learned prediction rule, and
apply that rule to new queries. The paper dissects this process end-to-end,
centering its in-depth case study on a structured family of tasks whose true
rule is *add an integer k to the input*:

- **A general localization pipeline.** A sparse-optimization method that
  jointly trains a coefficient matrix over all (layer, head) pairs, refined by
  statistical and effect-size criteria, localizes the model's few-shot ability
  to only a few attention heads. On Llama-3-8B-Instruct, just **three
  attention heads** carry most of the addition mechanism. The same pipeline
  generalizes across **five task families** (addition, subtraction,
  multiplication, and the abstractive and extractive suites of Todd et al.):
  one middle-layer head recurs as a main head in every family, further heads
  are shared within the arithmetic families, and the semantic families
  localize to broader head sets. It also carries over to other models, with
  Phi-4-14B mirroring this structure and Qwen2.5-7B localizing on addition
  (paper Appendix B).
- **Subspace Characterization.** For the addition tasks, dimensionality
  reduction and decomposition of those three heads' outputs show the task
  signal lives in a **six-dimensional subspace** per head: four dimensions
  track the unit digit with trigonometric functions at periods 2, 5, and 10,
  and the other two track magnitude with low-frequency components.
- **Signal extraction at the label tokens.** A mathematical identity relating
  each head's "aggregation" subspace to its per-token "extraction" subspaces
  traces the task signal back through the prompt: attention weights and
  extracted-signal norms peak at the demonstration label tokens, which carry
  96–99% of each main head's task signal, and the heads extract the offset
  y_i − x_i from each demonstration separately.

## Installation

```bash
git clone https://github.com/xyVickyHu/addition-subspaces.git
cd addition-subspaces
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python >= 3.10. The model-forward phases need a CUDA GPU; the test
suite (`pytest -m "not gpu"`) and the PCA/decomposition analyses run on CPU.

Model weights are **not** included: the code downloads
`meta-llama/Meta-Llama-3-8B-Instruct` from HuggingFace at runtime. The model is
gated, so request access on its HuggingFace page and authenticate with
`huggingface-cli login` first (or pre-download the weights and point `HF_HOME`
at the cache; the Slurm examples document offline-cluster cache pinning).

## Quick start: reproduce the paper pipeline

The paper's analysis runs as four code phases chained by a single
orchestrator. Using the shipped trained matrix
(`artifacts/matrix_add_0204_clip_lambda0.05`) skips the expensive
head-selection training:

```bash
python -m subspaces.runners.run_pipeline \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --task_dir number_add \
  --reuse-matrix artifacts/matrix_add_0204_clip_lambda0.05
```

This runs, in order: (1) matrix reuse (or training from scratch when no matrix
is given), (2) selected-head extraction plus the function-vector evaluations,
narrowing down to the paper's three main heads (layer 15 head 2, layer 15
head 1, layer 13 head 6) via mean-ablation, (3) per-head PCA and the
six-dimensional period/magnitude basis fit (paper Section 4), (4) the
Section 5 extracted-signal analysis, and (5) an experiments-log append.
Result summaries (JSON) and plots land in
`subspaces/runs/<model>__<task>/`; recomputable activation caches and newly trained
matrices land under `log/`.

The default main-head selector is data-driven (dose-response recovery) and
may pick a slightly different set on a fresh run; append
`--main-heads-fallback 15:2,15:1,13:6` to pin the paper's exact trio (the
third table row below assumes it).

Reproduction targets (bootstrap-CI numbers; expect roughly ±0.02–0.04
run-to-run variance because activation caches are rebuilt from freshly sampled
prompts):

| Configuration | Accuracy |
|---|---:|
| Clean 5-shot ICL baseline | 0.8567 |
| 33-head function vector (unit coefficients) | 0.7817 |
| 3 main heads + mean-ablation of the other 30 | 0.7017 |

The paper's AIE baselines, a paper-faithful re-implementation of Todd et al.'s average indirect effect (e.g. top-33 = 0.414 on Llama-3 add-k), reproduce via the `configs/step1_*_aie.yaml` cells with `subspaces.runners.step1`.

`python -m subspaces.runners.verify_reproducibility` is a CPU-only sanity check that
the shipped matrix reproduces the paper's 33-head selected set and the
three main heads.

**[docs/REPRODUCING.md](docs/REPRODUCING.md) is the full reproduction guide**:
per-phase entry points, training the matrix from scratch, the activation-cache
behavior behind the tolerance above, prompt formats, and the shipped task
datasets and splits.

## Head-set terminology

The code uses the paper's names for the three nested head sets (Section 3):

| set | how it is chosen | Llama-3-8B add-k | in the code |
|---|---|---|---|
| **selected** | sparse-optimization coefficient matrix, cut at its largest gap (`largest_gap`; `elbow` / `fraction` / `fixed` are alternatives) | 33 | `selected:` config section, `extract-selected` subcommand, `selected-<method>-v<V>/selected_heads.json` |
| **significant** | paired McNemar test + Benjamini-Hochberg over the recovery scan (`paired_bh` q=0.05) | 13 | `significant-<selector>-v<V>-<h8>/main_heads.json` (the selector-output contract) |
| **main** | quarter-of-the-best-recovery-gain rule (`unified` v1) | 3 | `main-<selector>-v<V>-<h8>/main_heads.json` |

"Statistically significant" keeps its ordinary meaning (the paired selectors'
test verdicts). The pre-2026-09 code called the selected set "significant"
(`sig`) and the significant set "recovery-positive" (`recpos`); artifacts
written under that spelling (including the shipped legacy stage outputs under
`artifacts/`) are translated on read, and every fingerprint is computed over the
legacy spelling so identities never fork; see the terminology block in
`subspaces/artifacts.py` (`LEGACY_KEY_SPELLING`). Two counts keep distinct names
because `n_selected` (the number of heads a selector chose) predates the rename:
the size of the selected set is `n_selected_set` in headset evaluations, and
the number of heads a scan covered (the selected set unless `scan.head_limit`
truncates it) is `n_scanned` in selector verdicts.

## Repository layout

| Path | Contents |
|---|---|
| `subspaces/` | The Python package. `subspaces/runners/` holds the per-phase entry points and the `run_pipeline` orchestrator; `subspaces/utils/` holds the core primitives (activations, interventions, PCA, signals). `subspaces/step1/`–`subspaces/step3/` are a newer modular re-implementation of the analysis steps; the paper-reproduction path is `subspaces.runners.run_pipeline`. |
| `scripts/` | Slurm launch wrappers and analysis helpers (context generation, label-share summary, five-shot necessity) |
| `configs/` | Experiment configs and model profiles for the modular runners |
| `dataset_files/` | The five task families used in the paper (pre-built, with fixed train/held-out splits) |
| `artifacts/` | Shipped reference artifacts, including the paper's trained head-coefficient matrices (the final-epoch checkpoint window and evaluation records) |
| `tests/` | Test suite; `pytest -m "not gpu"` runs the CPU-only checks |
| `examples/slurm/` | Parameterized Slurm batch scripts for the full pipeline and for matrix training (edit the `#SBATCH` header for your cluster) |
| `docs/REPRODUCING.md` | Full reproduction guide |
| `paper.pdf` | The COLM 2026 camera-ready paper |

## What ships vs. what gets computed

**Shipped with the repository:**

- the paper's trained sparse head-coefficient matrices and their evaluation
  artifacts (`artifacts/`), so reproduction does not require retraining;
- the five paper task families (`dataset_files/`, JSON input/output pairs)
  with their fixed train/held-out splits (`configs/task_splits/`);
- the camera-ready paper (`paper.pdf`).

**Computed at runtime (not shipped):**

- model weights, downloaded from HuggingFace on first use;
- activation caches of per-task mean head outputs (no raw activations are
  shipped; any GPU phase rebuilds the cache automatically on first run);
- all run outputs, plots, and logs.

## Hardware

All GPU phases fit on a single A100-class GPU (the paper runs used one A100
80GB); the PCA/decomposition phase is CPU-only once the activation cache
exists. The Slurm examples request one GPU, 80GB of RAM, and a 12-hour time
limit, which comfortably covers the full pipeline against the shipped matrix.

## Citation

```bibtex
@inproceedings{hu2026understanding,
  title     = {Understanding In-context Learning of Addition via Activation Subspaces},
  author    = {Xinyan Hu and Kayo Yin and Michael I. Jordan and Jacob Steinhardt and Lijie Chen},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026},
}
```

An earlier version is available as [arXiv:2505.05145](https://arxiv.org/abs/2505.05145).

## License

MIT, see [LICENSE](LICENSE).

## Contact

Please open a GitHub issue for questions about the code. For questions about
the paper, contact Xinyan Hu.
