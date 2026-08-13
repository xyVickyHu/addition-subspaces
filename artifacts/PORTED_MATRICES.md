# Ported coefficient matrices

13 historical coefficient-matrix directories ported on **2026-07-26** from the
original (private) research repository's run archive; only existing matrices
were ported, missing ones retrained. **As shipped**: 8 of the 13 remain (one
was withdrawn — see Withdrawals at the bottom — and the four Qwen2.5-7B
non-add ports were pruned from the release as non-paper cells), each pruned
to its trailing three checkpoints. `artifacts/matrix_add_0204_clip_lambda0.05`
(the paper's canonical addition matrix, documented in its own README) ships
alongside the ported set.

For each source directory `[0,1]_<name>` we created `artifacts/<name>` and copied
**only** `checkpoints/` (all `matrix_epoch*.pth`), `args.json`, and `accuracy_dict.pth`
where present. `savevars/` was **deliberately not ported** — it is a recomputable
legacy z cache (hundreds of MB per directory). `output.txt`, plots, and all other
files were likewise not copied. Files were copied with `cp` (independent physical
copies; no hardlinks into the source archive).

Verification performed per directory: (a) checkpoint count matches the source,
(b) sha256 of the final checkpoint matches the source's, (c) `args.json`
byte-identical to the source (and `accuracy_dict.pth` byte-identical where present).
All 13 passed.

Checkpoint directories are pruned to the trailing three epochs each selection
rule consumes (mean_last_k k=3 / latest / pinned final epoch); the recorded
counts below describe the original training runs.

## Llama-3-8B-Instruct (4)

### matrix_sub_meta-llama-3-8b-instruct_lambda0.05
- Source run: `[0,1]_matrix_sub_meta-llama-3-8b-instruct_lambda0.05` (private research archive)
- Copied: 2026-07-26. Checkpoints: 50 (epochs 0..49). Complete.
- Final ckpt `matrix_epoch49.pth` sha256: `0b6ca9d40688f304d903e6afb56541e5bcb28eebcc384be2657ad747f7907de0`
- args: model_name=meta-llama/Meta-Llama-3-8B-Instruct, task_dir=number_add[-30,0]_[1,100], n_shot=5, lambdaL1=0.05, lr=0.01, bs=128, epoch_num=50, train_ratio=0.8, ood_limit=512, anneal=1, layer_name=blocks.10.hook_resid_mid, seed=42
- accuracy_dict.pth: yes

### matrix_mul_meta-llama-3-8b-instruct_lambda0.05
- Source run: `[0,1]_matrix_mul_meta-llama-3-8b-instruct_lambda0.05` (private research archive)
- Copied: 2026-07-26. Checkpoints: 50 (epochs 0..49). Complete.
- Final ckpt `matrix_epoch49.pth` sha256: `96ae3b8b5ebcc518bc022a6f49083841021e932a4252281722aa127a0176f595`
- args: model_name=meta-llama/Meta-Llama-3-8B-Instruct, task_dir=number_mul, n_shot=5, lambdaL1=0.05, lr=0.01, bs=128, epoch_num=50, train_ratio=0.8, ood_limit=512, anneal=1, layer_name=blocks.10.hook_resid_mid, seed=42
- accuracy_dict.pth: yes

### matrix_extractive_meta-llama-3-8b-instruct_lambda0.05
- Source run: `[0,1]_matrix_extractive_meta-llama-3-8b-instruct_lambda0.05` (private research archive)
- Copied: 2026-07-26. Checkpoints: 33 (epochs 0..32).
- **PARTIAL**: training stopped at epoch 32 of a planned 50; this is nonetheless the
  published matrix (partial-but-published).
- Final ckpt `matrix_epoch32.pth` sha256: `9661bf5aaf795e964f9ffbde17d9ad7cb53673f294f970d4f16aea25d4ff4f14`
- args: model_name=meta-llama/Meta-Llama-3-8B-Instruct, task_dir=extractive, n_shot=5, lambdaL1=0.05, lr=0.01, bs=128, epoch_num=50, train_ratio=0.8, ood_limit=512, anneal=1, layer_name=blocks.10.hook_resid_mid, seed=42
- accuracy_dict.pth: absent in source (training did not complete)

### matrix_abstractive_meta-llama-3-8b-instruct_lambda0.05
- Source run: `[0,1]_matrix_abstractive_meta-llama-3-8b-instruct_lambda0.05` (private research archive)
- Copied: 2026-07-26. Checkpoints: 50 (epochs 0..49). Complete.
- Final ckpt `matrix_epoch49.pth` sha256: `6fb19756831baf41ccbb560e2c07257ec6399773509b3b3ac1b60040272f5eaf`
- args: model_name=meta-llama/Meta-Llama-3-8B-Instruct, task_dir=abstractive, n_shot=5, lambdaL1=0.05, lr=0.01, bs=128, epoch_num=50, train_ratio=0.8, ood_limit=512, anneal=1, layer_name=blocks.10.hook_resid_mid, seed=42
- accuracy_dict.pth: yes

## Phi-4 (5)

### matrix_add_phi-4_lambda0.05 — WITHDRAWN, does not ship (see Withdrawals)
- Source run: `[0,1]_matrix_add_phi-4_lambda0.05` (private research archive)
- Copied: 2026-07-26. Checkpoints: 50 (epochs 0..49). Complete.
- Final ckpt `matrix_epoch49.pth` sha256: `3ec82cd17613da8ccd2bb5a7bd7245b0e91c3886f4ddb6cc5bab888e38111c8c`
- args: model_name=microsoft/phi-4, task_dir=number_add, n_shot=5, lambdaL1=0.05, lr=0.01, bs=32, epoch_num=50, train_ratio=0.9, ood_limit=256, anneal=1, layer_name=blocks.12.hook_resid_mid, seed=42
- accuracy_dict.pth: yes

### matrix_sub_phi-4_lambda0.05
- Source run: `[0,1]_matrix_sub_phi-4_lambda0.05` (private research archive)
- Copied: 2026-07-26. Checkpoints: 50 (epochs 0..49). Complete.
- Final ckpt `matrix_epoch49.pth` sha256: `19632d3cf99cf88df8932c21e4678aee7b04fa0aadf71b1c9920f5fbcf29c070`
- args: model_name=microsoft/phi-4, task_dir=number_add[-30,0]_[1,100], n_shot=5, lambdaL1=0.05, lr=0.01, bs=32, epoch_num=50, train_ratio=0.9, ood_limit=256, anneal=1, layer_name=blocks.12.hook_resid_mid, seed=42
- accuracy_dict.pth: yes

### matrix_mul_phi-4_lambda0.05
- Source run: `[0,1]_matrix_mul_phi-4_lambda0.05` (private research archive)
- Copied: 2026-07-26. Checkpoints: 50 (epochs 0..49). Complete.
- Final ckpt `matrix_epoch49.pth` sha256: `d7dbd9d35e33e7cba6cd88fea96fa6563cb2db6e42b6d823c4ff2636a9bd8691`
- args: model_name=microsoft/phi-4, task_dir=number_mul, n_shot=5, lambdaL1=0.05, lr=0.01, bs=32, epoch_num=50, train_ratio=0.9, ood_limit=256, anneal=1, layer_name=blocks.12.hook_resid_mid, seed=42
- accuracy_dict.pth: yes

### matrix_abstractive_phi-4_lambda0.05
- Source run: `[0,1]_matrix_abstractive_phi-4_lambda0.05` (private research archive)
- Copied: 2026-07-26. Checkpoints: 50 (epochs 0..49). Complete.
- Final ckpt `matrix_epoch49.pth` sha256: `143057dd0cd74e34a69a46bb7663cd7ef8b4a940e7595d17688313f17e049a06`
- args: model_name=microsoft/phi-4, task_dir=abstractive, n_shot=5, lambdaL1=0.05, lr=0.01, bs=32, epoch_num=50, train_ratio=0.9, ood_limit=256, anneal=1, layer_name=blocks.12.hook_resid_mid, seed=42
- accuracy_dict.pth: yes

### matrix_extractive_phi-4_lambda0.05
- Source run: `[0,1]_matrix_extractive_phi-4_lambda0.05` (private research archive)
- Copied: 2026-07-26. Checkpoints: 49 (epochs 0..48).
- **PARTIAL**: 49 of 50 planned checkpoints; the source `output.txt` ends mid-epoch 49
  (final-epoch checkpoint was never written).
- Final ckpt `matrix_epoch48.pth` sha256: `3ab2b81eeb967a5ae94ece913b47784002e84d31dcfae5c98cf6870067fafb08`
- args: model_name=microsoft/phi-4, task_dir=extractive, n_shot=5, lambdaL1=0.05, lr=0.01, bs=32, epoch_num=50, train_ratio=0.9, ood_limit=256, anneal=1, layer_name=blocks.12.hook_resid_mid, seed=42
- accuracy_dict.pth: absent in source (training did not complete)

## Totals

- Directories ported at port time: 13; 9 remain documented above
  (4 llama3 + 5 phi-4) after the four qwen2.5-7b non-add ports were
  pruned from the release (non-paper cells)
- Checkpoints copied at port time: 1,432 (8,865,429 bytes, ~8.5 MiB)
- Shipped in this release: 8 directories (4 llama3 + 4 phi-4;
  matrix_add_phi-4 withdrawn), 3 trailing checkpoints each
  (24 total from the ported set)

## Whitespace normalization note (2026-07-26)

The repo's pre-commit `end-of-file-fixer` appended a trailing newline to
every ported `args.json` at commit time — the same normalization the
release whitespace pass applied to the shipped canonical matrix.
Parsed JSON content is identical to
the sources; "byte-identical" above should be read as byte-identical modulo
that single trailing newline. Checkpoint `.pth` files are untouched
(shas above remain exact).

## Withdrawals (2026-07-27)

- `matrix_add_phi-4_lambda0.05` — REMOVED (user decision): trained on a
  different holdout split (4/5 committed eval tasks inside its training
  set). Source checkpoints + savevars were also deleted in the private
  archive (args.json kept as audit record).
- `matrix_all5_clip_lambda0.05` — REMOVED (matrix and results; the raw
  all5 task JSONs do not ship either): the combined-tasks cell moved to a
  proportioned, validity-filtered mixed-task variant (itself not part of
  the paper and not shipped) before any all5 results were produced.
