"""Step-1 CLI. Argparse only — computation lives in ``subspaces.step1.*``.

Substep boundaries (each consumes the persisted artifact of the previous one;
artifacts live in the lineage tree ``log/runs/<node>__<id12>/sig-*/scan-*/main-*``
and per-invocation config records in ``log/journal/<run_name>__<jhash12>/``):

    python -m subspaces.runners.step1 train              --config <yaml>
    python -m subspaces.runners.step1 select-significant --config <yaml> [--matrix <dir|matrix_ref.json>]
    python -m subspaces.runners.step1 make-samples       --config <yaml>
    python -m subspaces.runners.step1 scan               --config <yaml> [--significant <json>]
    python -m subspaces.runners.step1 select-main        --scan <head_scan.json>
                                                  --selector <name> [--selector-version N]
                                                  [--param k=v ...] [--pin-heads 'L:H,...']
    python -m subspaces.runners.step1 evaluate-headset   --config <yaml> --main <main_heads.json>
    python -m subspaces.runners.step1 run                --config <yaml>
    python -m subspaces.runners.step1 aie                --config <yaml> [--methods m1,m2] [--k 20,33]

Changing the significant method never retrains the matrix (sibling ``sig-*``
nodes share the matrix node); changing the main selector never reruns the GPU
scan (``select-main`` is pure CPU; sibling ``main-*`` nodes share their scan);
only ``evaluate-headset`` reruns the model for a new selector. The ``aie``
subcommand runs the parallel AIE-baseline chain (subspaces/step1/aie.py) end to end,
resumable — it never touches the sig/scan/main lineage.
"""

from __future__ import annotations

import argparse
import os
import sys

from subspaces.artifacts import ArtifactError
from subspaces.config import ConfigError, MatrixConfig, load_step1_config
from subspaces.head_sets import HeadSpecError
from subspaces.paths import ProjectPaths
from subspaces.step1 import pipeline
from subspaces.step1.samples import HoldoutViolation
from subspaces.step1.selectors import SelectorError
from subspaces.step1.split import TaskSplitError

_CLI_ERRORS = (
    ArtifactError,
    ConfigError,
    HeadSpecError,
    HoldoutViolation,
    NotImplementedError,
    OSError,  # clean message (not a traceback) on e.g. read-only checkouts
    SelectorError,
    TaskSplitError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subspaces.runners.step1", description=__doc__
    )
    parser.add_argument("--root", default=None, help="repo root (default: auto)")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_config(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--config", required=True, help="Step-1 variant YAML")
        return p

    with_config(sub.add_parser("train", help="train the coefficient matrix"))
    p = with_config(
        sub.add_parser("select-significant", help="significant heads from the matrix")
    )
    p.add_argument(
        "--matrix",
        default=None,
        help="matrix dir or matrix_ref.json (overrides config matrix.reuse_path)",
    )
    with_config(sub.add_parser("make-samples", help="materialize sample manifests"))
    p = with_config(sub.add_parser("scan", help="per-head recovery scan (GPU)"))
    p.add_argument(
        "--significant", default=None, help="significant_heads.json artifact"
    )
    p = sub.add_parser("select-main", help="apply a main-head selector (pure CPU)")
    p.add_argument("--scan", required=True, help="head_scan.json")
    p.add_argument("--selector", required=True, help="selector name (e.g. unified)")
    p.add_argument("--selector-version", type=int, default=1)
    p.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="typed selector-parameter override (repeatable); unknown names "
        "are rejected",
    )
    p.add_argument(
        "--pin-heads", default=None, help="explicit 'L:H,...' list (pin selector)"
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="parent for the main-* node dir (default: the scan's directory)",
    )
    p = with_config(
        sub.add_parser("evaluate-headset", help="evaluate selected set (GPU)")
    )
    p.add_argument("--main", required=True, help="main_heads.json artifact")
    p.add_argument(
        "--subspace",
        default=None,
        help="step2_subspace artifact (subspace-<h10>.json) over ⊇ the "
        "selected heads, fitted on this cell's train+held-out z caches: "
        "adds the projected arm (proj_meanab) to the evaluation",
    )
    p = with_config(
        sub.add_parser(
            "aie",
            help="AIE baseline: corrupted samples -> per-head scores -> "
            "top-k Todd-FV eval (GPU; requires an aie: config block)",
        )
    )
    p.add_argument(
        "--methods",
        default=None,
        help="comma-separated subset of aie.methods (a method not in the "
        "config refuses)",
    )
    p.add_argument(
        "--k",
        default=None,
        help="comma-separated subset of aie.k_values (a k not in the config "
        "refuses)",
    )
    p = sub.add_parser(
        "compose", help="compose the heads artifact (validates lineage, CPU)"
    )
    p.add_argument("--significant", required=True, help="significant_heads.json")
    p.add_argument("--main", required=True, help="main_heads.json artifact")
    p.add_argument("--out-dir", default=None, help="default: alongside the main")
    with_config(sub.add_parser("run", help="composed run of all substeps"))
    return parser


def _matrix_override(
    matrix_arg: str, base: MatrixConfig, paths: ProjectPaths
) -> MatrixConfig:
    """Resolve --matrix (a matrix dir or a matrix_ref.json) to a reuse config.

    Only the locator fields change: the loaded config's checkpoint selection
    (checkpoint_select / checkpoint_epoch), override_provenance and node_name
    survive the override instead of being silently discarded."""
    import dataclasses

    from subspaces.artifacts import read_manifest
    from subspaces.step1.matrix import MATRIX_REF_SCHEMA_VERSION

    target = paths.resolve(matrix_arg)
    reuse_path = str(matrix_arg)
    if target.is_file():
        manifest = read_manifest(
            target,
            expect_kind="matrix_ref",
            max_schema_version=MATRIX_REF_SCHEMA_VERSION,
        )
        selected = manifest.get("selected_checkpoint") or {}
        if selected.get("selection_rule") == "mean_last_k":
            raise ArtifactError(
                f"--matrix points at a DERIVED matrix_ref ({target}): its "
                "matrix_dir holds only the materialized mean, no "
                "matrix_epoch<N>.pth checkpoints. Reproduce the derived "
                "selection from the source instead: set matrix.reuse_path to "
                f"the recorded source_matrix_dir "
                f"({selected.get('source_matrix_dir')!r}) and "
                "matrix.checkpoint_select to "
                f"{{rule: {selected.get('selection_rule')!r}, "
                f"k: {selected.get('k')!r}}} in the config."
            )
        reuse_path = manifest["matrix_dir"]
    return dataclasses.replace(base, mode="reuse", reuse_path=reuse_path, train=None)


def _selector_params(args: argparse.Namespace) -> dict:
    """Preset parameters for known selectors + typed --param overrides."""
    from subspaces.head_sets import parse_heads
    from subspaces.step1.selectors import (
        ABOVE_ABLATION_V1_PARAMS,
        PAIRED_BH_V1_PARAMS,
        PAIRED_BY_V1_PARAMS,
        PAIRED_GLOBAL_BONFERRONI_V1_PARAMS,
        PAIRED_PER_HEAD_V1_PARAMS,
        UNIFIED_V1_PARAMS,
        WEAK_RELEASE_PARAMS,
        SelectorError,
    )

    presets: dict[tuple[str, int], dict] = {
        ("unified", 1): dict(UNIFIED_V1_PARAMS),
        ("recovery_weak", 1): dict(WEAK_RELEASE_PARAMS),
        ("pin", 1): {},
        ("largest_gap", 1): {},  # parameter-free; --param rejects everything
        ("above_ablation", 1): dict(ABOVE_ABLATION_V1_PARAMS),
        ("above_ablation_zero", 1): {},  # parameter-free
        ("compare_zero", 1): {},  # parameter-free
        ("paired_bh", 1): dict(PAIRED_BH_V1_PARAMS),
        ("paired_per_head", 1): dict(PAIRED_PER_HEAD_V1_PARAMS),
        ("paired_by", 1): dict(PAIRED_BY_V1_PARAMS),
        ("paired_global_bonferroni", 1): dict(PAIRED_GLOBAL_BONFERRONI_V1_PARAMS),
    }
    key = (args.selector, args.selector_version)
    params = presets.get(key, {})
    for item in args.param:
        name, sep, raw = item.partition("=")
        if not sep:
            raise SelectorError(f"--param expects NAME=VALUE, got {item!r}")
        if (
            presets.get(key) is not None
            and name not in presets[key]
            and key
            != (
                "pin",
                1,
            )
        ):
            known = sorted(presets[key])
            raise SelectorError(
                f"unknown parameter {name!r} for selector {args.selector} "
                f"v{args.selector_version} (known: {known})"
            )
        try:
            params[name] = float(raw)
        except ValueError:
            raise SelectorError(
                f"--param {name}: expected a number, got {raw!r}"
            ) from None
    if args.pin_heads is not None:
        if args.selector != "pin":
            raise SelectorError("--pin-heads is only valid with --selector pin")
        params["heads"] = [list(h) for h in parse_heads(args.pin_heads)]
    return params


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = ProjectPaths.from_root(args.root)
    try:
        if args.command == "compose":
            main_path = paths.resolve(args.main)
            # frozen pre-tree (flat) runs keep main heads under step1/ or a
            # main_heads/ grouping dir; the defaulted out-dir must never
            # write into them (tree main-* nodes keep working by default)
            if args.out_dir is None and main_path.parent.name in (
                "main_heads",
                "step1",
            ):
                raise ArtifactError(
                    f"{main_path} lives in a frozen pre-tree run: compose "
                    "must not write into it by default; pass --out-dir."
                )
            out_dir = paths.resolve(args.out_dir) if args.out_dir else main_path.parent
            pipeline.compose_heads(
                significant_path=paths.resolve(args.significant),
                main_path=main_path,
                paths=paths,
                out_dir=out_dir,
            )
            print(f"[step1 compose] heads: {out_dir / 'heads.json'}")
            return 0
        if args.command == "select-main":
            scan_path = paths.resolve(args.scan)
            parent_dir = paths.resolve(args.out_dir) if args.out_dir else None
            _, main_node = pipeline.apply_selector(
                scan_path,
                selector_name=args.selector,
                selector_version=args.selector_version,
                params=_selector_params(args),
                paths=paths,
                parent_dir=parent_dir,
            )
            print(f"[step1 select-main] node: {main_node}")
            return 0

        cfg = load_step1_config(paths.resolve(args.config))
        # The prompt format is CONFIG-driven: the legacy rendering/training
        # code reads $FV_PROMPT_FORMAT at call time, so adopt the config's
        # format for this process when the env does not set one (empty
        # string counts as unset, matching resolve_runtime). An env var that
        # CONTRADICTS the config still refuses (resolve_runtime), so a job
        # wrapper cannot silently override the recorded config.
        if not os.environ.get("FV_PROMPT_FORMAT"):
            os.environ["FV_PROMPT_FORMAT"] = cfg.task.prompt_format
        if args.command == "select-significant" and args.matrix:
            cfg.matrix = _matrix_override(args.matrix, cfg.matrix, paths)
        if args.command == "aie" and cfg.aie is None:
            # refuse BEFORE ensure_run creates any journal record
            raise ConfigError(
                f"{args.config}: no aie block — the aie subcommand requires "
                "one (see AIEConfig in subspaces/config.py; "
                "cell configs: configs/step1_*_aie.yaml)"
            )
        journal_dir, split, resolved = pipeline.ensure_run(cfg, paths)
        print(f"[step1 {args.command}] journal: {journal_dir.name}")

        if args.command == "run":
            pipeline.run(cfg, paths)
        elif args.command == "train":
            # through the stage so ref-writing/reuse rules apply to training too
            _, matrix_node = pipeline.stage_matrix(cfg, paths, journal_dir)
            print(f"[step1 train] node: {matrix_node}")
        elif args.command == "select-significant":
            matrix_ref, matrix_node = pipeline.stage_matrix(cfg, paths, journal_dir)
            _, sig_node = pipeline.stage_significant(
                cfg, paths, matrix_node, matrix_ref
            )
            print(f"[step1 select-significant] node: {sig_node}")
        elif args.command == "make-samples":
            samples = pipeline.stage_selection_samples(cfg, paths, split)
            for kind, (_, path) in samples.items():
                print(f"[step1 make-samples] {kind}: {path}")
        elif args.command == "scan":
            from subspaces.artifacts import read_manifest
            from subspaces.step1.significant import SIGNIFICANT_SCHEMA_VERSION

            matrix_ref, matrix_node = pipeline.stage_matrix(cfg, paths, journal_dir)
            if args.significant:
                from subspaces.artifacts import semantic_fingerprint

                significant_path = paths.resolve(args.significant)
                significant = read_manifest(
                    significant_path,
                    expect_kind="significant_heads",
                    max_schema_version=SIGNIFICANT_SCHEMA_VERSION,
                )
                sig_matrix_ref = (significant.get("inputs") or {}).get("matrix_ref")
                matrix_fp = semantic_fingerprint(matrix_ref)
                if (
                    not sig_matrix_ref
                    or sig_matrix_ref.get("semantic_fingerprint") != matrix_fp
                ):
                    raise ArtifactError(
                        "--significant artifact does not descend from this "
                        "run's matrix (fingerprint mismatch); refusing to scan "
                        "a foreign head list."
                    )
                significant, sig_node = pipeline.adopt_significant(
                    matrix_node, significant_path, significant, paths
                )
            else:
                significant, sig_node = pipeline.stage_significant(
                    cfg, paths, matrix_node, matrix_ref
                )
            samples = pipeline.stage_selection_samples(cfg, paths, split)
            model_loader = pipeline.make_model_loader(cfg, resolved)
            _, scan_node = pipeline.stage_scan(
                cfg,
                paths,
                sig_node,
                significant,
                samples["activation"],
                samples["scan"],
                resolved,
                model_loader,
            )
            print(f"[step1 scan] node: {scan_node}")
        elif args.command == "aie":
            from subspaces.step1 import aie as aie_mod

            methods = None
            if args.methods is not None:
                methods = [item.strip() for item in args.methods.split(",")]
            k_values = None
            if args.k is not None:
                try:
                    k_values = [int(item.strip()) for item in args.k.split(",")]
                except ValueError:
                    raise ConfigError(
                        f"--k expects comma-separated integers, got {args.k!r}"
                    ) from None
            summary = aie_mod.run_aie(
                cfg,
                paths,
                journal_dir=journal_dir,
                split=split,
                resolved=resolved,
                methods=methods,
                k_values=k_values,
            )
            for name, entry in sorted(summary["evals"].items()):
                print(f"[step1 aie] {name}: {entry['metrics']}")
            print(f"[step1 aie] dir: {journal_dir / 'aie'}")
        elif args.command == "evaluate-headset":
            from subspaces.artifacts import read_manifest
            from subspaces.step1.selectors import MAIN_HEADS_SCHEMA_VERSION

            main_path = paths.resolve(args.main)
            main_manifest = read_manifest(
                main_path,
                expect_kind="main_heads",
                max_schema_version=MAIN_HEADS_SCHEMA_VERSION,
            )
            matrix_ref, matrix_node = pipeline.stage_matrix(cfg, paths, journal_dir)
            significant, sig_node = pipeline.stage_significant(
                cfg, paths, matrix_node, matrix_ref
            )
            samples = pipeline.stage_selection_samples(cfg, paths, split)
            expected_scan = pipeline._scan_expected_identity(
                cfg,
                paths,
                sig_node / "significant_heads.json",
                significant,
                samples["activation"][0],
                samples["scan"][0],
                samples["scan"][1],
                resolved,
            )
            from subspaces.step1 import tree

            scan_node = tree.scan_node_dir(sig_node, cfg, expected_scan)
            model_loader = pipeline.make_model_loader(cfg, resolved)
            pipeline.stage_evaluate_headset(
                cfg,
                paths,
                split,
                matrix_node,
                sig_node,
                scan_node,
                significant,
                matrix_ref,
                main_manifest,
                main_path.parent,
                samples["activation"],
                samples["scan"],
                resolved,
                model_loader,
                subspace_path=(paths.resolve(args.subspace) if args.subspace else None),
            )
        return 0
    except _CLI_ERRORS as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
