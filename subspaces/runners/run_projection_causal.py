"""Appendix F — onto / out-of subspace causal projections (GPU runner).

Regenerates the paper's Appendix-E arms from the ADOPTED modern artifacts,
reproducing the recovered legacy semantics of
the legacy evaluation-notebook protocol exactly:

  For a head ``(l, h)`` with per-head optimal coefficient ``c*`` (the recpos
  ``main_heads.json`` ``decisions[l:h].c_star``), a column subset ``cols`` of
  the fitted mod-vector matrix ``M = mod_vectors[l:h]`` (d_model, 6) —
  column order ``MOD_VECTOR_COLS = (mod50, mod2, mod5, mod10c, mod10s,
  mod25)`` so parity = [1], unit = [1,2,3,4], magnitude = [5,0] — and a
  projection variant:

    onto   (arm "(l, h)*c=1-2-3-4")  : B = QR-orthonormalized ``M[:, cols]``
    out of (arm "(l, h)*c!=1-2-3-4") : B = complement of ``M[:, cols]``
                                       INSIDE the 6-D span
                                       (``subspaces.utils.pca.compute_subspace_complement``)

  the per-task intervention vector is

    FV(t) = c* · [ (z_t[l,h] − mean_z[l,h]) B Bᵀ + mean_z[l,h] ]
            + Σ_{(l',h') ∈ significant ∖ {(l,h)}} mean_z[l',h']

  where ``z_t`` comes from the cell's merged (train + held-out) z caches and
  ``mean_z`` is the TRAIN-task mean (25 tasks — the modern recovery-scan
  convention ``mean_fv − mean_z[h]``; the legacy notebook's across-task mean
  may have included held-out tasks, a small documented difference). The FV is
  injected ADDITIVELY (intervention_mode 0) at the cell's inject layer on the
  ZERO-SHOT prompts of every evaluated add-k task via
  ``subspaces.utils.intervene.compute_task_accuracy`` (intervene-only branch,
  ``print_intervened_num`` = batch size so the per-example target/generated
  strings are retained for the Appendix-E error-decomposition plots).

Results are written as ``results-<tag>.pth`` — a list of legacy-schema
"matrix dicts" (``matrix_name``, per-task ``*_intervened_acc`` /
``*_intervened_target_strings`` / ``*_intervened_generated_strings``,
``train_intervened_acc``, ``test_intervened_acc``) in the legacy "matrix
dict" schema — plus a ``manifest-<tag>.json`` recording all resolved inputs.

Examples
--------
CPU dry run (resolves every artifact, builds and checks the projection bases,
never loads the model)::

    .venv/bin/python -m subspaces.runners.run_projection_causal --cell add --dry-run

Smoke (tiny)::

    .venv/bin/python -m subspaces.runners.run_projection_causal \
        --cell add --heads 15:2 --arms onto_unit --tasks 2 --test-limit 10

Full cell (all default heads × {parity,unit,mag} × {onto,out})::

    .venv/bin/python -m subspaces.runners.run_projection_causal --cell add
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

from subspaces.artifacts import ArtifactError
from subspaces.paths import ProjectPaths

RESULT_KIND = "projection_causal"
RESULT_SCHEMA_VERSION = 1
IMPL = {"module": "subspaces.runners.run_projection_causal", "algorithm_version": 1}

CELLS_TSV = "configs/step23_recpos_cells.tsv"
CONTEXT_TEMPLATE = "configs/contexts/step2_recpos_{cell}.yaml"
SUPPORTED_CELLS = ("add", "phi4", "qwen")

# Named column subsets of the mod-vector matrix (MOD_VECTOR_COLS order:
# mod50, mod2, mod5, mod10c, mod10s, mod25). The tuple order is the legacy
# arm-name order (combinations were enumerated over the list [1,2,3,4,5,0]),
# so arm names match the legacy artifact byte-for-byte ("1-2-3-4", "5-0").
SUBSPACE_COLS: dict[str, tuple[int, ...]] = {
    "parity": (1,),
    "unit": (1, 2, 3, 4),
    "mag": (5, 0),
}
VARIANTS = ("onto", "out")
ALL_ARMS = tuple(f"{v}_{s}" for v in VARIANTS for s in SUBSPACE_COLS)

# Paper Appendix-E heads per cell (the recpos MAIN heads carrying the
# published panels; every one has a c_star decision and a fitted mod-vector).
DEFAULT_HEADS: dict[str, tuple[tuple[int, int], ...]] = {
    "add": ((15, 2), (15, 1), (13, 6)),
    "phi4": ((18, 12), (18, 14), (17, 23)),
    "qwen": ((21, 0), (21, 5), (21, 4), (21, 2)),
}


# --------------------------------------------------------------------------- #
# pure helpers (CPU; importing this module stays light)                       #
# --------------------------------------------------------------------------- #


def parse_heads(spec: str) -> list[tuple[int, int]]:
    """Parse 'L:H,L:H,...' into [(L, H), ...]."""
    heads: list[tuple[int, int]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        layer_str, _, head_str = token.partition(":")
        if not head_str:
            raise ValueError(f"head {token!r} is not in L:H form")
        heads.append((int(layer_str), int(head_str)))
    if not heads:
        raise ValueError(f"no heads parsed from {spec!r}")
    return heads


def parse_arms(spec: str) -> list[tuple[str, str]]:
    """Parse --arms into [(variant, subspace), ...]; 'all' = the 6 arms."""
    if spec.strip() == "all":
        tokens = list(ALL_ARMS)
    else:
        tokens = [token.strip() for token in spec.split(",") if token.strip()]
    arms: list[tuple[str, str]] = []
    for token in tokens:
        variant, _, subspace = token.partition("_")
        if variant not in VARIANTS or subspace not in SUBSPACE_COLS:
            raise ValueError(
                f"arm {token!r} not recognized; expected one of {sorted(ALL_ARMS)}"
            )
        arms.append((variant, subspace))
    if not arms:
        raise ValueError(f"no arms parsed from {spec!r}")
    return arms


def arm_name(
    head: tuple[int, int], coefficient: int, subspace: str, variant: str
) -> str:
    """Legacy arm key, e.g. "(15, 2)*6=1-2-3-4" (onto) / "...*6!=1-2-3-4" (out)."""
    comb = "-".join(str(col) for col in SUBSPACE_COLS[subspace])
    op = "=" if variant == "onto" else "!="
    return f"({head[0]}, {head[1]})*{coefficient}{op}{comb}"


def build_projection_basis(
    mod_vectors: np.ndarray, cols: tuple[int, ...] | list[int], variant: str
) -> np.ndarray:
    """Orthonormal basis for one arm, float64 (d_model, r).

    onto: QR of the chosen mod-vector columns. out: the complement of those
    columns INSIDE the 6-D span (``subspaces.utils.pca.compute_subspace_complement``).
    """
    matrix = np.asarray(mod_vectors, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"mod_vectors must be 2-D, got shape {matrix.shape}")
    chosen = list(cols)
    if variant == "onto":
        basis, _ = np.linalg.qr(matrix[:, chosen])
        return basis
    if variant == "out":
        from subspaces.utils.pca import (
            compute_subspace_complement,  # heavy import chain
        )

        return np.asarray(compute_subspace_complement(matrix, chosen), dtype=np.float64)
    raise ValueError(f"unknown variant {variant!r} (expected 'onto' or 'out')")


def build_fv_vectors(
    z_head: dict[str, np.ndarray],
    mean_z_head: np.ndarray,
    others_mean: np.ndarray,
    basis: np.ndarray,
    coefficient: int,
) -> dict[str, np.ndarray]:
    """Per-task intervention vectors for one arm (float64).

    FV(t) = c·[(z_t − μ_h) B Bᵀ + μ_h] + Σ_{sig∖{h}} μ_{h'} — the recovered
    legacy ``eval_head_mod`` formula with the train-only mean convention.
    """
    fvs: dict[str, np.ndarray] = {}
    projector = basis @ basis.T
    for task_name, z_vec in z_head.items():
        centered = np.asarray(z_vec, dtype=np.float64) - mean_z_head
        projected = centered @ projector + mean_z_head
        fvs[task_name] = coefficient * projected + others_mean
    return fvs


def task_sort_key(task_name: str) -> int:
    return int(task_name.rsplit("add", 1)[-1])


# --------------------------------------------------------------------------- #
# cell / artifact resolution (CPU)                                            #
# --------------------------------------------------------------------------- #


def resolve_cell(cell: str, paths: ProjectPaths) -> dict:
    """Resolve one TSV cell to its config, context, recpos node, and periodic
    artifact paths. Pure path/JSON work — no tensors, no model."""
    import yaml

    from subspaces.config import load_step1_config
    from subspaces.context import load_context

    if cell not in SUPPORTED_CELLS:
        raise ArtifactError(
            f"cell {cell!r} is not supported; expected one of {SUPPORTED_CELLS}"
        )
    tsv_path = paths.resolve(CELLS_TSV)
    row = None
    with open(tsv_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if fields[0] == cell:
                row = fields
                break
    if row is None or len(row) < 3:
        raise ArtifactError(f"cell {cell!r} not found in {tsv_path}")
    config_path = paths.resolve(row[1])
    recpos_main_path = paths.resolve(row[2])
    node_dir = recpos_main_path.parent
    cfg = load_step1_config(config_path)

    context_path = paths.resolve(CONTEXT_TEMPLATE.format(cell=cell))
    context = load_context(context_path)
    context.validate_refs(paths)
    context_heads = context.heads_artifact_path()
    if context_heads and paths.resolve(context_heads) != recpos_main_path:
        raise ArtifactError(
            f"context {context_path} pins heads artifact {context_heads}, but the "
            f"cells TSV points at {recpos_main_path}; refusing mismatched sources."
        )
    if context.model.get("name") != cfg.model.name:
        raise ArtifactError(
            f"context model {context.model.get('name')!r} != config model "
            f"{cfg.model.name!r}; refusing a mislabeled cell."
        )
    for name in ("train", "heldout"):
        if name not in context.activations:
            raise ArtifactError(
                f"context {context_path} lacks activations.{name}; the F arms "
                "need the merged train + held-out z caches."
            )

    periodic_jsons = sorted(node_dir.glob("periodic-*.json"))
    if len(periodic_jsons) != 1:
        raise ArtifactError(
            f"expected exactly 1 periodic-*.json in {node_dir}, "
            f"found {len(periodic_jsons)}"
        )
    periodic_path = periodic_jsons[0]
    with open(periodic_path, encoding="utf-8") as fh:
        periodic = json.load(fh)
    npz_path = node_dir / periodic["mod_vectors_file"]
    if not npz_path.is_file():
        raise ArtifactError(f"mod-vector npz missing: {npz_path}")

    with open(recpos_main_path, encoding="utf-8") as fh:
        main_heads = json.load(fh)
    if "decisions" not in main_heads:
        raise ArtifactError(f"{recpos_main_path} has no per-head decisions (c_star)")
    significant_path = node_dir.parent.parent / "significant_heads.json"
    if not significant_path.is_file():
        raise ArtifactError(f"significant_heads.json missing: {significant_path}")
    with open(significant_path, encoding="utf-8") as fh:
        significant = json.load(fh)

    with open(paths.resolve(cfg.task.task_split), encoding="utf-8") as fh:
        split = yaml.safe_load(fh)

    z_cache_dirs = [
        paths.resolve(context.activations[name]["path"])
        for name in ("train", "heldout")
    ]
    return {
        "cell": cell,
        "config_path": config_path,
        "cfg": cfg,
        "context_path": context_path,
        "context": context,
        "recpos_main_path": recpos_main_path,
        "main_heads": main_heads,
        "significant_path": significant_path,
        "significant": significant,
        "node_dir": node_dir,
        "periodic_path": periodic_path,
        "periodic": periodic,
        "npz_path": npz_path,
        "split": split,
        "z_cache_dirs": z_cache_dirs,
    }


def load_cell_tensors(
    resolved: dict, paths: ProjectPaths, heads: list[tuple[int, int]]
) -> dict:
    """Merge the cell's z caches and slice everything the arms need (CPU).

    Returns per-head per-task z vectors (float64), the TRAIN-task per-head
    means, the significant-set mean sum, mod-vector matrices, and c_star
    decisions.
    """
    import torch

    from subspaces.step2.pca import load_and_merge_zcaches

    z_results, info = load_and_merge_zcaches(resolved["z_cache_dirs"], paths)
    cache_model = info["model"] or {}
    for field in ("name", "revision"):
        declared = resolved["context"].model.get(field)
        actual = cache_model.get(field)
        if declared and actual and declared != actual:
            raise ArtifactError(
                f"context.model.{field} ({declared!r}) does not match the z "
                f"caches' extraction identity ({actual!r}); refusing."
            )

    train_tasks = sorted(resolved["split"]["train_tasks"])
    heldout_tasks = sorted(resolved["split"]["eval_tasks"])
    caches_by_fp = {cache["content_fingerprint"]: cache for cache in info["z_caches"]}
    for label, directory, expected_tasks in (
        ("train", resolved["z_cache_dirs"][0], train_tasks),
        ("heldout", resolved["z_cache_dirs"][1], heldout_tasks),
    ):
        cache = caches_by_fp.get(directory.name)
        if cache is None:
            raise ArtifactError(
                f"the context's '{label}' z cache {directory} did not survive "
                "the merge; refusing."
            )
        if sorted(cache["tasks"]) != expected_tasks:
            raise ArtifactError(
                f"the context's '{label}' z cache does not cover exactly the "
                f"committed {label} split ({resolved['cfg'].task.task_split}); "
                "refusing."
            )

    sig_heads = [
        (int(layer_idx), int(head_idx))
        for layer_idx, head_idx, *_ in resolved["significant"]["heads"]
    ]
    decisions = resolved["main_heads"]["decisions"]
    npz = np.load(resolved["npz_path"])
    mod_vectors = {key: np.asarray(npz[key], dtype=np.float64) for key in npz}

    for layer_idx, head_idx in heads:
        key = f"{layer_idx}:{head_idx}"
        if key not in decisions:
            raise ArtifactError(
                f"head {key} has no c_star decision in {resolved['recpos_main_path']}"
            )
        if key not in mod_vectors:
            raise ArtifactError(
                f"head {key} has no fitted mod-vectors in {resolved['npz_path']}"
            )

    needed = sorted(set(heads) | set(sig_heads))
    mean_z_head: dict[tuple[int, int], np.ndarray] = {}
    for layer_idx, head_idx in needed:
        acc = np.zeros(info["dims"]["d_model"], dtype=np.float64)
        for task in train_tasks:
            acc += z_results[task][layer_idx, head_idx].to(torch.float64).numpy()
        mean_z_head[(layer_idx, head_idx)] = acc / len(train_tasks)
    sig_mean_sum = np.sum([mean_z_head[head] for head in sig_heads], axis=0)

    all_tasks = sorted(z_results, key=task_sort_key)
    z_head: dict[tuple[int, int], dict[str, np.ndarray]] = {
        (layer_idx, head_idx): {
            task: z_results[task][layer_idx, head_idx].to(torch.float64).numpy()
            for task in all_tasks
        }
        for layer_idx, head_idx in heads
    }
    return {
        "z_info": info,
        "z": z_head,
        "all_tasks": all_tasks,
        "train_tasks": train_tasks,
        "heldout_tasks": heldout_tasks,
        "sig_heads": sig_heads,
        "mean_z_head": mean_z_head,
        "sig_mean_sum": sig_mean_sum,
        "decisions": decisions,
        "mod_vectors": mod_vectors,
    }


def plan_arms(
    heads: list[tuple[int, int]], arms: list[tuple[str, str]], tensors: dict
) -> list[dict]:
    """Materialize the arm list: name, basis, per-arm metadata (CPU)."""
    planned = []
    for head in heads:
        key = f"{head[0]}:{head[1]}"
        c_star = int(tensors["decisions"][key]["c_star"])
        matrix = tensors["mod_vectors"][key]
        for variant, subspace in arms:
            basis = build_projection_basis(matrix, SUBSPACE_COLS[subspace], variant)
            gram_err = float(np.abs(basis.T @ basis - np.eye(basis.shape[1])).max())
            planned.append(
                {
                    "name": arm_name(head, c_star, subspace, variant),
                    "head": head,
                    "c_star": c_star,
                    "variant": variant,
                    "subspace": subspace,
                    "cols": list(SUBSPACE_COLS[subspace]),
                    "basis": basis,
                    "basis_rank": int(basis.shape[1]),
                    "basis_orthonormality_err": gram_err,
                }
            )
    return planned


# --------------------------------------------------------------------------- #
# GPU evaluation                                                              #
# --------------------------------------------------------------------------- #


def load_model_pinned(model_cfg: dict, device: str | None):
    """Offline model load with the same revision pinning as subspaces.step1.eval_gpu."""
    import torch

    from subspaces.step1.resolve import resolve_model_identity
    from subspaces.utils.model import load_model_no_grad

    name = model_cfg["name"]
    pinned = model_cfg.get("revision")
    if pinned:
        observed = resolve_model_identity(name)
        if observed.get("revision") != pinned:
            raise ArtifactError(
                f"the local cache resolves {name} to revision "
                f"{observed.get('revision')} but this cell pins {pinned}; "
                "refusing to load drifted weights."
            )
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, model_cfg.get("dtype") or "bfloat16")
    model, _n_layers, _n_heads = load_model_no_grad(name, resolved_device, dtype=dtype)
    return model


def evaluate_arm(
    model,
    arm: dict,
    tensors: dict,
    tasks: dict,
    eval_tasks: list[str],
    layer_name: str,
    n_shot: int,
    test_limit: int,
    batch_size: int,
    seed: int,
) -> dict:
    """Run one arm over the evaluated tasks; returns the legacy matrix dict."""
    import torch

    from subspaces.utils.intervene import compute_task_accuracy

    head = arm["head"]
    mean_z_head = tensors["mean_z_head"][head]
    others_mean = tensors["sig_mean_sum"] - mean_z_head
    fvs = build_fv_vectors(
        tensors["z"][head], mean_z_head, others_mean, arm["basis"], arm["c_star"]
    )
    device = model.cfg.device
    fvs_torch = {
        task: torch.from_numpy(vec).to(device=device, dtype=model.cfg.dtype)
        for task, vec in fvs.items()
    }
    matrix_dict: dict[str, object] = {"matrix_name": arm["name"]}
    for task_name in eval_tasks:
        # Re-seed per (arm, task) so every arm evaluates the IDENTICAL query
        # sample (the legacy notebook seeded once per head sweep instead).
        random.seed(seed)
        torch.manual_seed(seed)
        (
            _clean_acc,
            _corrupted_acc,
            intervened_acc,
            _proj_acc,
            _trans_acc,
            _clean_gen,
            _clean_tgt,
            _corr_gen,
            _corr_tgt,
            intervened_targets,
            intervened_generated,
        ) = compute_task_accuracy(
            task_name,
            tasks,
            n_shot,
            test_limit,
            batch_size,
            fvs_torch,
            None,
            None,
            None,
            model,
            layer_name,
            intervene=True,
            clean=False,
            corrupted=False,
            print_intervened_num=batch_size,
        )
        matrix_dict[f"{task_name}_intervened_acc"] = float(intervened_acc)
        matrix_dict[f"{task_name}_intervened_target_strings"] = list(intervened_targets)
        matrix_dict[f"{task_name}_intervened_generated_strings"] = list(
            intervened_generated
        )
        print(
            f"[projcausal] {arm['name']} {task_name} "
            f"intervened_acc={float(intervened_acc):.4f}",
            flush=True,
        )
    for label, subset in (
        ("train_intervened_acc", tensors["train_tasks"]),
        ("test_intervened_acc", tensors["heldout_tasks"]),
    ):
        accs = [
            matrix_dict[f"{task}_intervened_acc"]
            for task in subset
            if f"{task}_intervened_acc" in matrix_dict
        ]
        matrix_dict[label] = float(np.mean(accs)) if accs else float("nan")
    return matrix_dict


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subspaces.runners.run_projection_causal",
        description="Appendix-E onto/out-of subspace causal projections",
    )
    parser.add_argument("--root", default=None)
    parser.add_argument(
        "--cell",
        required=True,
        choices=SUPPORTED_CELLS,
        help="configs/step23_recpos_cells.tsv row (llama add / phi4 / qwen)",
    )
    parser.add_argument(
        "--heads",
        default=None,
        help="'L:H,L:H,...'; default: the cell's paper Appendix-E heads",
    )
    parser.add_argument(
        "--arms",
        default="all",
        help="comma list of {onto,out}_{parity,unit,mag}, or 'all' (default)",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=None,
        help="evaluate only the first N add-k tasks (k order); default all 30",
    )
    parser.add_argument(
        "--test-limit", type=int, default=100, help="queries per task (default 100)"
    )
    parser.add_argument(
        "--bs", type=int, default=100, help="evaluation batch size (default 100)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default=None, help="override model device (default: cuda)"
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output directory (default: <recpos node>/projection-causal)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve artifacts, build/check bases, print the plan; no model",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = ProjectPaths.from_root(args.root)
    try:
        resolved = resolve_cell(args.cell, paths)
        heads = (
            parse_heads(args.heads) if args.heads else list(DEFAULT_HEADS[args.cell])
        )
        arms = parse_arms(args.arms)

        cfg = resolved["cfg"]
        if resolved["context"].task.get("prompt_format") != "arrow":
            raise ArtifactError(
                "only the arrow prompt format is supported (the Appendix-E "
                "protocol); refusing."
            )
        batch_size = min(args.bs, args.test_limit)
        if args.test_limit % batch_size != 0:
            raise ArtifactError(
                f"--test-limit {args.test_limit} must be a multiple of the "
                f"effective batch size {batch_size} (DataLoader drop_last would "
                "silently drop examples)."
            )

        tensors = load_cell_tensors(resolved, paths, heads)
        planned = plan_arms(heads, arms, tensors)
        eval_tasks = tensors["all_tasks"]
        if args.tasks is not None:
            eval_tasks = eval_tasks[: args.tasks]

        out_dir = (
            paths.resolve(args.out_dir)
            if args.out_dir
            else resolved["node_dir"] / "projection-causal"
        )
        identity = {
            "cell": args.cell,
            "heads": [list(head) for head in heads],
            "arms": sorted(f"{variant}_{sub}" for variant, sub in arms),
            "tasks": eval_tasks,
            "test_limit": args.test_limit,
            "batch_size": batch_size,
            "seed": args.seed,
            "c_star": {
                f"{h[0]}:{h[1]}": int(tensors["decisions"][f"{h[0]}:{h[1]}"]["c_star"])
                for h in heads
            },
            "z_caches": sorted(
                cache["content_fingerprint"] for cache in tensors["z_info"]["z_caches"]
            ),
            "mod_vectors_sha256": resolved["periodic"]["mod_vectors_sha256"],
            "impl": IMPL,
        }
        from subspaces.artifacts import content_hash

        tag = content_hash(identity)[:10]
        result_path = Path(out_dir) / f"results-{tag}.pth"
        manifest_path = Path(out_dir) / f"manifest-{tag}.json"

        print(f"[projcausal] cell={args.cell} config={resolved['config_path']}")
        print(f"[projcausal] context={resolved['context_path']}")
        print(f"[projcausal] recpos={resolved['recpos_main_path']}")
        print(f"[projcausal] periodic={resolved['periodic_path']}")
        print(
            f"[projcausal] model={cfg.model.name} rev={cfg.model.revision} "
            f"dtype={cfg.model.dtype} inject={cfg.sites.inject_layer}"
        )
        print(
            f"[projcausal] sig_heads={len(tensors['sig_heads'])} "
            f"tasks={len(eval_tasks)} test_limit={args.test_limit} bs={batch_size} "
            f"seed={args.seed}"
        )
        for arm in planned:
            print(
                f"[projcausal] arm {arm['name']:<28} rank={arm['basis_rank']} "
                f"orthonormality_err={arm['basis_orthonormality_err']:.2e}"
            )
        print(f"[projcausal] out: {result_path}")
        if args.dry_run:
            print("[projcausal] dry run complete (no model load, no writes).")
            return 0

        if result_path.exists() or manifest_path.exists():
            raise ArtifactError(
                f"{result_path} (or its manifest) already exists; results are "
                "immutable — choose another --out-dir or remove the stale file "
                "deliberately."
            )

        import os

        os.environ["FV_PROMPT_FORMAT"] = resolved["context"].task["prompt_format"]

        import torch

        from subspaces.utils.io import load_task_data

        tasks = load_task_data(str(paths.task_dir(cfg.task.dataset_dir)))
        missing = [task for task in eval_tasks if task not in tasks]
        if missing:
            raise ArtifactError(f"dataset lacks evaluated tasks: {missing}")

        model = load_model_pinned(
            {
                "name": cfg.model.name,
                "revision": cfg.model.revision,
                "dtype": cfg.model.dtype,
            },
            args.device,
        )
        results: list[dict] = []
        out_dir.mkdir(parents=True, exist_ok=True)
        for arm in planned:
            results.append(
                evaluate_arm(
                    model,
                    arm,
                    tensors,
                    tasks,
                    eval_tasks,
                    cfg.sites.inject_layer,
                    cfg.task.n_shot,
                    args.test_limit,
                    batch_size,
                    args.seed,
                )
            )
            torch.save(results, result_path)  # checkpoint after every arm

        from subspaces.artifacts import make_manifest, sha256_file, write_json_atomic

        manifest = make_manifest(
            kind=RESULT_KIND,
            schema_version=RESULT_SCHEMA_VERSION,
            paths=paths,
            config=identity,
            inputs={
                "recpos_main_heads": {
                    "path": paths.relativize(resolved["recpos_main_path"])
                },
                "significant_heads": {
                    "path": paths.relativize(resolved["significant_path"])
                },
                "periodic": {
                    "path": paths.relativize(resolved["periodic_path"]),
                    "mod_vectors_sha256": resolved["periodic"]["mod_vectors_sha256"],
                },
                "z_caches": {
                    "content_fingerprints": identity["z_caches"],
                },
            },
            payload={
                "impl": IMPL,
                "model": {
                    "name": cfg.model.name,
                    "revision": cfg.model.revision,
                    "dtype": cfg.model.dtype,
                },
                "inject_layer": cfg.sites.inject_layer,
                "n_shot": cfg.task.n_shot,
                "mean_convention": "train_tasks_only",
                "train_tasks": tensors["train_tasks"],
                "heldout_tasks": tensors["heldout_tasks"],
                "results_file": result_path.name,
                "results_sha256": sha256_file(result_path),
                "arms": [
                    {
                        "name": arm["name"],
                        "head": list(arm["head"]),
                        "c_star": arm["c_star"],
                        "variant": arm["variant"],
                        "subspace": arm["subspace"],
                        "cols": arm["cols"],
                        "basis_rank": arm["basis_rank"],
                        "train_intervened_acc": result["train_intervened_acc"],
                        "test_intervened_acc": result["test_intervened_acc"],
                    }
                    for arm, result in zip(planned, results, strict=True)
                ],
            },
        )
        write_json_atomic(manifest_path, manifest)
        print(f"[projcausal] wrote {result_path}")
        print(f"[projcausal] wrote {manifest_path}")
        return 0
    except (ArtifactError, ValueError, OSError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
