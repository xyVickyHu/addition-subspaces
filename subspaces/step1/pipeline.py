"""Substep composition for Step 1, plus the ``heads`` artifact.

Artifacts live in the LINEAGE TREE (``subspaces.step1.tree``, output layout v2):
each substep's output sits under its primary parent, and method/protocol
variants are prefix-named sibling directories —

    log/runs/<node>__<ckptid12>/selected-*/scan-*/main-*/

The composed runner executes the substeps in order, reusing an existing
node artifact only after validating its input/config/impl identity (the
shared refuse-on-mismatch rule). Running the subcommands sequentially must
produce the same artifacts as one composed run (tested). Per-invocation
config records (requested/resolved + node pointers) land in
``log/journal/<run_name>__<jhash12>/``.

The final ``heads`` artifact (kind ``heads``, ``<main-node>/heads.json``)
is the stable input for Steps 2/3: it exposes ``selected_heads``,
``main_heads``, and ``minor_heads`` plus references to the artifacts it was
derived from. Consumers never parse a journal record, and per-example scan
outcomes are NOT exposed here.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from subspaces.artifacts import (
    IDENTITY_KEYS,
    ArtifactError,
    check_manifest_ref,
    file_ref,
    identity_of,
    make_manifest,
    manifest_ref,
    modernize,
    read_manifest,
    reuse_or_refuse,
    semantic_fingerprint,
    write_json_atomic,
)
from subspaces.config import ConfigError, Step1Config, dump_resolved, to_dict
from subspaces.head_sets import HEADS_SCHEMA_VERSION
from subspaces.paths import ProjectPaths
from subspaces.step1 import matrix as matrix_mod
from subspaces.step1 import samples as samples_mod
from subspaces.step1 import selected as selected_mod
from subspaces.step1 import tree
from subspaces.step1.matrix import MATRIX_REF_SCHEMA_VERSION
from subspaces.step1.recovery import HEAD_SCAN_SCHEMA_VERSION
from subspaces.step1.selected import SELECTED_SCHEMA_VERSION
from subspaces.step1.selectors import MAIN_HEADS_SCHEMA_VERSION
from subspaces.step1.split import TaskSplit, load_task_split

# matrix_ref has no upstream artifact inputs; its content identity is the
# selected checkpoint + provenance, so those join the reuse-identity subset.
_MATRIX_IDENTITY_KEYS = (*IDENTITY_KEYS, "selected_checkpoint", "provenance_check")


def load_split(cfg: Step1Config, paths: ProjectPaths) -> TaskSplit | None:
    if cfg.task.task_split is None:
        return None
    split = load_task_split(cfg.task.task_split, paths)
    if split.family != cfg.task.dataset_dir:
        raise ConfigError(
            f"task split family {split.family!r} does not match "
            f"task.dataset_dir {cfg.task.dataset_dir!r}; refusing."
        )
    return split


def ensure_run(
    cfg: Step1Config, paths: ProjectPaths
) -> tuple[Path, TaskSplit | None, dict]:
    """Finalize the run identity, THEN create the JOURNAL entry.

    The run ID hashes normalized requested semantics plus resolved content IDs
    (checkpoint SHA, dataset/split fingerprints, effective model revision,
    algorithm versions). Locators and cache-availability/status fields are
    excluded, so relocating inputs never forks a run, and late cache
    resolution under a pinned model revision continues the same run.

    Under the lineage tree the journal is BOOKKEEPING (one config = one
    reproducible pass, findable by run_name, guarded against identity
    drift); artifacts dedup at their tree nodes regardless of how many
    journal entries touch them.
    """
    from subspaces.step1.resolve import (
        resolve_runtime,
        run_id_for,
        write_resolved_config,
    )

    split = load_split(cfg, paths)
    resolved = resolve_runtime(cfg, paths, split)
    journal_dir = paths.journal_dir(run_id_for(cfg, resolved))
    journal_dir.mkdir(parents=True, exist_ok=True)
    requested_path = journal_dir / "requested_config.yaml"
    if not requested_path.exists():
        dump_resolved(cfg, requested_path)
    write_resolved_config(cfg, resolved, journal_dir)
    return journal_dir, split, resolved


def _verify_matrix_ref_checkpoint(manifest: dict, paths: ProjectPaths) -> None:
    """The referenced checkpoint must exist with its recorded sha256."""
    from subspaces.artifacts import sha256_file

    selected = manifest["selected_checkpoint"]
    checkpoint = (
        paths.resolve(manifest["matrix_dir"]) / "checkpoints" / selected["file"]
    )
    # derived checkpoints record the byte sha as "file_sha256" (volatile,
    # load-verification only); pinned/latest checkpoints record "sha256".
    expected_sha = selected.get("sha256") or selected.get("file_sha256")
    actual = sha256_file(checkpoint) if checkpoint.is_file() else None
    if actual != expected_sha:
        raise ArtifactError(
            f"matrix checkpoint {checkpoint} is missing or changed "
            f"(sha {str(actual)[:12]} vs recorded {str(expected_sha)[:12]}); "
            "refusing."
        )


def _matrix_request_semantics(matrix_config: dict) -> dict:
    """The matrix config minus locator/presentation fields — what makes two
    train requests 'the same training'."""
    cleaned = dict(matrix_config)
    cleaned.pop("node_name", None)
    cleaned.pop("reuse_path", None)
    return cleaned


def _find_trained_node(cfg: Step1Config, paths: ProjectPaths) -> dict | None:
    """Adopt an existing trained matrix node for this train request, if any.

    One train config = one trained matrix: a re-training would produce a
    numerically DIFFERENT checkpoint (retraining reproducibility,
    the step-1 methods notes), so an existing node with the same training
    request semantics is adopted instead of silently retraining.
    """
    runs = paths.runs_dir
    if not runs.is_dir():
        return None
    want = _matrix_request_semantics(to_dict(cfg.matrix))
    orphans: list[Path] = []
    for node in sorted(entry for entry in runs.glob("*") if entry.is_dir()):
        ref_path = node / "matrix_ref.json"
        if not ref_path.is_file():
            # trained checkpoints without an identity record: the crash
            # window between the staging rename and the matrix_ref write
            if (node / "matrix" / "checkpoints").is_dir():
                orphans.append(node)
            continue
        try:
            candidate = read_manifest(
                ref_path,
                expect_kind="matrix_ref",
                max_schema_version=MATRIX_REF_SCHEMA_VERSION,
            )
        except ArtifactError:
            continue
        if candidate.get("provenance") != "trained":
            continue
        have = _matrix_request_semantics(
            (candidate.get("config") or {}).get("matrix") or {}
        )
        if have != want:
            continue
        # A matching TRAIN REQUEST is necessary but not sufficient: the
        # matrix block carries only hyperparameters, while the trained
        # numbers also depend on task/model settings outside it — the four
        # format matrices share one matrix block and differ only in
        # task.prompt_format (GPU run: the fmt-fx eval adopted the
        # fmt-ab node and the divergence guard aborted the wave). The node's
        # annotated args.json records those settings; a provenance mismatch
        # means "a different training", not an error.
        try:
            matrix_mod.check_provenance(paths.resolve(candidate["matrix_dir"]), cfg)
        except ArtifactError:
            continue
        _verify_matrix_ref_checkpoint(candidate, paths)
        return candidate
    if orphans:
        raise ArtifactError(
            "no adoptable trained matrix node matches this train request, "
            "but orphan trained node dir(s) exist (matrix/checkpoints/ "
            "without matrix_ref.json):\n"
            + "\n".join(f"  {orphan}" for orphan in orphans)
            + "\nThis is the crash window between the staging rename and the "
            "matrix_ref write; inspect each dir and either restore its "
            "matrix_ref.json from the journal memo or remove the dir — "
            "refusing to silently retrain."
        )
    return None


def _trained_matrix(
    cfg: Step1Config, paths: ProjectPaths, journal_dir: Path, memo_path: Path
) -> dict:
    """Train into a journal-scoped staging dir, then move under the matrix
    node (the node key — the checkpoint sha — is unknowable pre-training).
    The journal memo makes re-invocations find their node without retraining;
    an interrupted training resumes in the same staging dir."""
    if memo_path.exists():
        memo = read_manifest(
            memo_path,
            expect_kind="matrix_ref",
            max_schema_version=MATRIX_REF_SCHEMA_VERSION,
        )
        memo_request = _matrix_request_semantics(
            (memo.get("config") or {}).get("matrix") or {}
        )
        if memo_request != _matrix_request_semantics(to_dict(cfg.matrix)):
            raise ArtifactError(
                f"{memo_path}: journal matrix memo does not match this "
                "config's matrix request; refusing."
            )
        _verify_matrix_ref_checkpoint(memo, paths)
        return memo
    adopted = _find_trained_node(cfg, paths)
    if adopted is not None:
        return adopted
    staging = paths.runs_dir / f".train-staging-{journal_dir.name}"
    manifest = matrix_mod.train_matrix(cfg, paths, staging)
    node_dir = tree.matrix_node_dir(paths, cfg, manifest)
    target = node_dir / "matrix"
    if not target.exists():
        node_dir.mkdir(parents=True, exist_ok=True)
        (staging / "step1" / "matrix").rename(target)
        shutil.rmtree(staging, ignore_errors=True)
    # locator only (volatile): the semantic identity is unchanged by the move
    manifest["matrix_dir"] = paths.relativize(target)
    _verify_matrix_ref_checkpoint(manifest, paths)
    return manifest


def stage_matrix(
    cfg: Step1Config, paths: ProjectPaths, journal_dir: Path
) -> tuple[dict, Path]:
    """Resolve or train the matrix; persist the node's ``matrix_ref.json``
    (+ a journal memo). Reuse only on full identity equality (config, impl,
    selected checkpoint, provenance) via the shared reuse rule — two
    requests that resolve to the same checkpoint CONTENT but different
    request semantics (e.g. pinned vs latest) refuse with a clear message
    instead of sharing a node ambiguously.
    """
    memo_path = journal_dir / "matrix_ref.json"
    if cfg.matrix.mode == "reuse":
        manifest = matrix_mod.resolve_reuse_matrix(cfg, paths)
    else:
        manifest = _trained_matrix(cfg, paths, journal_dir, memo_path)
    # journal-memo drift check FIRST: a mismatch (e.g. a checkpoint changed
    # under this journal) must refuse BEFORE any node dir is created.
    memo = reuse_or_refuse(
        memo_path,
        manifest,
        expect_kind="matrix_ref",
        max_schema_version=MATRIX_REF_SCHEMA_VERSION,
        keys=_MATRIX_IDENTITY_KEYS,
    )
    node_dir = tree.matrix_node_dir(paths, cfg, manifest)
    # node-name divergence guard: the SAME matrix identity already
    # materialized under a differently named node dir would silently fork
    # the whole subtree (and re-run its GPU scans) over a presentation name.
    node_key = tree.checkpoint_key(manifest["selected_checkpoint"])
    if paths.runs_dir.is_dir():
        expected_identity = identity_of(manifest, _MATRIX_IDENTITY_KEYS)
        for sibling in sorted(paths.runs_dir.glob(f"*__{node_key}")):
            if sibling.name == node_dir.name:
                continue
            sibling_ref = sibling / "matrix_ref.json"
            if not sibling_ref.is_file():
                continue
            try:
                sibling_manifest = read_manifest(
                    sibling_ref,
                    expect_kind="matrix_ref",
                    max_schema_version=MATRIX_REF_SCHEMA_VERSION,
                )
            except ArtifactError:
                continue
            if identity_of(sibling_manifest, _MATRIX_IDENTITY_KEYS) == (
                expected_identity
            ):
                raise ArtifactError(
                    f"same matrix identity already materialized under "
                    f"{sibling}; refusing to create the duplicate node "
                    f"{node_dir}. Set matrix.node_name to "
                    f"{sibling.name.rsplit('__', 1)[0]!r} or remove the "
                    "stale node."
                )
    node_dir.mkdir(parents=True, exist_ok=True)
    node_ref = node_dir / "matrix_ref.json"
    existing = reuse_or_refuse(
        node_ref,
        manifest,
        expect_kind="matrix_ref",
        max_schema_version=MATRIX_REF_SCHEMA_VERSION,
        keys=_MATRIX_IDENTITY_KEYS,
    )
    if existing is not None:
        manifest = existing
    else:
        write_json_atomic(node_ref, manifest)
    if memo is None:
        write_json_atomic(memo_path, manifest)
    return manifest, node_dir


def stage_selected(
    cfg: Step1Config, paths: ProjectPaths, matrix_node: Path, matrix_ref: dict
) -> tuple[dict, Path]:
    """Selected-head selection with lineage-checked reuse.

    Method variants are sibling ``selected-*`` nodes under the matrix node. An
    existing ``selected_heads.json`` is reused only when its recorded
    (matrix input, selected config, implementation) identity matches what
    this config would produce — a stale artifact from a different matrix or
    method refuses instead of silently flowing downstream.
    """
    selected_node = tree.selected_node_dir(matrix_node, cfg.selected)
    out_path = selected_node / "selected_heads.json"
    ref_path = matrix_node / "matrix_ref.json"
    expected = {
        "kind": "selected_heads",
        "schema_version": SELECTED_SCHEMA_VERSION,
        "inputs": {"matrix_ref": manifest_ref(ref_path, paths, matrix_ref)},
        # identity slice, not to_dict: only the active method's registered
        # parameters — a future SelectedConfig field cannot fork existing
        # methods' identities (subspaces.step1.selected registry).
        "config": {"selected": selected_mod.selected_identity_config(cfg.selected)},
        "impl": selected_mod.impl_for(cfg.selected),
    }
    existing = reuse_or_refuse(
        out_path,
        expected,
        expect_kind="selected_heads",
        max_schema_version=SELECTED_SCHEMA_VERSION,
    )
    if existing is not None:
        return existing, selected_node
    manifest = selected_mod.extract_selected(matrix_ref, cfg, paths, expected=expected)
    selected_node.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_path, manifest)
    return manifest, selected_node


def adopt_selected(
    matrix_node: Path,
    selected_path: Path,
    selected: dict,
    paths: ProjectPaths,
) -> tuple[dict, Path]:
    """Place an EXTERNAL selected artifact (CLI ``--selected``) at its
    canonical node under this matrix; fingerprint-identical copies reuse."""
    selected_node = matrix_node / tree.selected_dirname_from_manifest(selected)
    out_path = selected_node / "selected_heads.json"
    if out_path.exists():
        existing = modernize(json.loads(out_path.read_text(encoding="utf-8")))
        if semantic_fingerprint(existing) != semantic_fingerprint(selected):
            raise ArtifactError(
                f"{out_path} exists with a different semantic identity than "
                f"the supplied {selected_path}; refusing to adopt."
            )
        return existing, selected_node
    selected_node.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_path, selected)
    return selected, selected_node


def _ensure_samples(
    cfg: Step1Config,
    spec,
    kind: str,
    split: TaskSplit | None,
    paths: ProjectPaths,
) -> tuple[dict, Path]:
    """Content-addressed sample manifest (``log/cache/samples/<fp16>/``).

    Samples are protocol inputs, not matrix descendants — they live in the
    shared cache (like z caches) and scans/evals reference them by
    fingerprint."""
    expected = samples_mod.samples_expected(cfg, spec, kind, split, paths)
    cache_dir = paths.samples_cache_dir / semantic_fingerprint(expected)[:16]
    manifest = samples_mod.generate_samples(cfg, spec, kind, split, paths, cache_dir)
    return manifest, cache_dir / f"{kind}.json"


def stage_selection_samples(
    cfg: Step1Config,
    paths: ProjectPaths,
    split: TaskSplit | None,
) -> dict[str, tuple[dict, Path]]:
    """Materialize the SELECTION-side sample manifests (activation + scan,
    plus aie_corrupted when the config carries an aie block).

    Held-out isolation: ``final_eval`` samples are deliberately NOT produced
    here — under the corrected protocol the five held-out tasks stay untouched
    until ``evaluate-headset`` creates/loads its own evaluation samples after
    selection.
    """
    kinds = [
        ("activation", cfg.samples.activation),
        ("scan", cfg.samples.scan),
    ]
    if cfg.aie is not None:
        kinds.append(("aie_corrupted", cfg.samples.aie_corrupted))
    return {
        kind: _ensure_samples(cfg, spec, kind, split, paths) for kind, spec in kinds
    }


def apply_selector(
    scan_path: str | Path,
    *,
    selector_name: str,
    selector_version: int,
    params: dict,
    paths: ProjectPaths,
    parent_dir: Path | None = None,
) -> tuple[dict, Path]:
    """Pure-CPU main-head selection over a saved scan.

    The output is ``<scan-dir>/main-<selector>-v<V>-<h8>/main_heads.json``
    (selector variants are sibling node dirs; the identity hash covers
    params + scan fingerprint + selector). Immutable: an existing file with
    a different semantic identity refuses.

    Selectors in ``OUTCOME_SELECTORS`` additionally receive the scan's
    per-example outcome vectors: the npz next to ``head_scan.json`` is
    loaded and digest-verified against the manifest, then handed to the
    selector as bytes per key (still pure CPU — no model, no re-scan).
    """
    from subspaces.step1.selectors import OUTCOME_SELECTORS, get_selector

    scan_path = paths.resolve(scan_path)
    # Frozen pre-tree (flat) runs keep their scan directly in <run>/step1/;
    # the default parent (the scan's directory) must never write into them.
    # Tree scan-* nodes keep working by default.
    if parent_dir is None and scan_path.parent.name == "step1":
        raise ArtifactError(
            f"{scan_path} lives in a frozen pre-tree run: path-based "
            "selection must not write into it; pass --out-dir to place the "
            "main-* node elsewhere."
        )
    scan_manifest = read_manifest(
        scan_path,
        expect_kind="head_scan",
        max_schema_version=HEAD_SCAN_SCHEMA_VERSION,
    )
    selector = get_selector(selector_name, selector_version)
    if (selector_name, selector_version) in OUTCOME_SELECTORS:
        arrays = _load_outcomes(scan_path.parent, scan_manifest)
        outcome_bytes = {key: value.tobytes() for key, value in arrays.items()}
        result = selector(scan_manifest, params, outcome_bytes)
    else:
        result = selector(scan_manifest, params)
    manifest = make_manifest(
        kind="main_heads",
        schema_version=MAIN_HEADS_SCHEMA_VERSION,
        paths=paths,
        config={"selector": {"name": selector_name, "version": selector_version}},
        inputs={"head_scan": manifest_ref(scan_path, paths, scan_manifest)},
        payload={
            "impl": {"module": "subspaces.step1.selectors", "algorithm_version": 1},
            "selector_name": result.selector_name,
            "selector_version": result.selector_version,
            "params": result.params,
            "main_heads": [list(h) for h in result.main_heads],
            "minor_heads": [list(h) for h in result.minor_heads],
            "decisions": result.decisions,
            "verdict": result.verdict,
        },
    )
    node_dir = tree.main_node_dir(
        parent_dir or scan_path.parent,
        selector_name,
        selector_version,
        params,
        scan_manifest,
    )
    out_path = node_dir / "main_heads.json"
    if out_path.exists():
        existing = modernize(json.loads(out_path.read_text(encoding="utf-8")))
        if semantic_fingerprint(existing) != semantic_fingerprint(manifest):
            raise ArtifactError(
                f"{out_path} exists with a different semantic identity despite "
                "an identical identity hash; refusing to overwrite."
            )
        return existing, node_dir
    node_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_path, manifest)
    return manifest, node_dir


def compose_heads(
    *,
    selected_path: str | Path,
    main_path: str | Path,
    paths: ProjectPaths,
    out_dir: Path,
) -> dict:
    """Compose the downstream ``heads`` artifact from persisted substep outputs.

    Lineage is enforced: the main-heads artifact must reference a head_scan,
    and that scan must descend from the SUPPLIED selected artifact — an
    incompatible main/selected pairing refuses instead of composing. The
    output is ``<main-node>/heads.json`` (single slot: its identity is fully
    determined by the selected + main pair fixed in that node).
    """
    selected = read_manifest(
        paths.resolve(selected_path),
        expect_kind="selected_heads",
        max_schema_version=SELECTED_SCHEMA_VERSION,
    )
    main = read_manifest(
        paths.resolve(main_path),
        expect_kind="main_heads",
        max_schema_version=MAIN_HEADS_SCHEMA_VERSION,
    )
    scan_ref = (main.get("inputs") or {}).get("head_scan")
    if not scan_ref:
        raise ArtifactError(
            f"{main_path}: main_heads artifact has no head_scan input "
            "reference; refusing to compose an un-lineaged head set."
        )
    scan = check_manifest_ref(
        scan_ref,
        paths,
        expect_kind="head_scan",
        max_schema_version=HEAD_SCAN_SCHEMA_VERSION,
    )
    scan_selected_ref = (scan.get("inputs") or {}).get("selected_heads")
    if not scan_selected_ref:
        raise ArtifactError(
            "head_scan artifact has no selected_heads input reference; "
            "refusing to compose an un-lineaged head set."
        )
    supplied_fp = semantic_fingerprint(selected)
    if scan_selected_ref.get("semantic_fingerprint") != supplied_fp:
        raise ArtifactError(
            "lineage mismatch: the main-heads scan descends from a different "
            "selected_heads artifact "
            f"({str(scan_selected_ref.get('semantic_fingerprint'))[:12]} vs supplied "
            f"{supplied_fp[:12]}); refusing to compose."
        )
    manifest = make_manifest(
        kind="heads",
        schema_version=HEADS_SCHEMA_VERSION,
        paths=paths,
        inputs={
            "selected_heads": manifest_ref(selected_path, paths, selected),
            "main_heads": manifest_ref(main_path, paths, main),
        },
        payload={
            "selected_heads": [
                [layer_idx, head_idx] for layer_idx, head_idx, *_ in selected["heads"]
            ],
            "main_heads": main["main_heads"],
            "minor_heads": main["minor_heads"],
            "selector": {
                "name": main["selector_name"],
                "version": main["selector_version"],
                "params": main["params"],
            },
            "model_dims": selected.get("model_dims"),
        },
    )
    out_path = out_dir / "heads.json"
    if out_path.exists():
        existing = modernize(json.loads(out_path.read_text(encoding="utf-8")))
        if semantic_fingerprint(existing) != semantic_fingerprint(manifest):
            raise ArtifactError(
                f"{out_path} exists with a different semantic identity; "
                "refusing to overwrite."
            )
        return existing
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_path, manifest)
    return manifest


def make_model_loader(cfg: Step1Config, resolved: dict):
    """Process-wide lazy model loader shared across GPU substeps (loads once;
    refuses when the model identity is unresolved)."""
    cache: dict = {}

    def loader():
        if "model" not in cache:
            from subspaces.step1.eval_gpu import load_model, require_resolved_model

            require_resolved_model(resolved)
            cache["model"] = load_model(cfg)
        return cache["model"]

    return loader


def _scan_expected_identity(
    cfg: Step1Config,
    paths: ProjectPaths,
    selected_path: Path,
    selected: dict,
    activation_samples: dict,
    scan_samples: dict,
    scan_samples_path: Path,
    resolved: dict,
) -> dict:
    """The scan artifact identity THIS config would produce — shared by
    stage_scan (reuse rule + node placement) and stage_evaluate_headset
    (foreign-scan guard)."""
    from subspaces.step1 import recovery as recovery_mod
    from subspaces.step1.eval_gpu import effective_batch_size
    from subspaces.step1.zcache import cache_fingerprint, cache_identity

    z_identity = cache_identity(cfg, resolved, activation_samples)
    inputs_refs = {
        "selected_heads": manifest_ref(selected_path, paths, selected),
        "scan_samples": manifest_ref(scan_samples_path, paths, scan_samples),
        "z_cache": {"content_fingerprint": cache_fingerprint(z_identity)},
    }
    return {
        "kind": "head_scan",
        "schema_version": HEAD_SCAN_SCHEMA_VERSION,
        "inputs": inputs_refs,
        "config": {
            "scan": to_dict(cfg.scan),
            "sites": to_dict(cfg.sites),
            "intervention_mode": 0,
            "batch_size": effective_batch_size(cfg),
            "scoring": to_dict(cfg.scoring),
        },
        "impl": recovery_mod.IMPL,
    }


def stage_scan(
    cfg: Step1Config,
    paths: ProjectPaths,
    selected_node: Path,
    selected: dict,
    activation_samples: tuple[dict, Path],
    scan_samples: tuple[dict, Path],
    resolved: dict,
    model_loader,
) -> tuple[dict, Path]:
    """Recovery scan with lineage-checked reuse (GPU on cache/scan miss).

    Scan protocol variants are sibling ``scan-*`` nodes under the
    selected node. An EXTERNAL selected artifact (CLI
    ``--selected``) is first adopted at its canonical node
    (``adopt_selected``), so the scan always hangs off the node it
    descends from.
    """
    from subspaces.step1 import recovery as recovery_mod
    from subspaces.step1.zcache import ensure_zcache

    selected_path = selected_node / "selected_heads.json"
    expected = _scan_expected_identity(
        cfg,
        paths,
        selected_path,
        selected,
        activation_samples[0],
        scan_samples[0],
        scan_samples[1],
        resolved,
    )
    scan_node = tree.scan_node_dir(selected_node, cfg, expected)
    out_path = scan_node / "head_scan.json"
    existing = reuse_or_refuse(
        out_path,
        expected,
        expect_kind="head_scan",
        max_schema_version=HEAD_SCAN_SCHEMA_VERSION,
    )
    if existing is not None:
        _verify_outcomes(scan_node, existing)
        return existing, scan_node
    z_results, fingerprint = ensure_zcache(
        model_loader, activation_samples[0], cfg, paths, resolved
    )
    scan_node.mkdir(parents=True, exist_ok=True)
    manifest = recovery_mod.run_scan(
        selected,
        scan_samples[0],
        {"z": z_results, "content_fingerprint": fingerprint},
        cfg,
        paths,
        scan_node,
        resolved=resolved,
        inputs_refs=expected["inputs"],
        model_loader=model_loader,
    )
    write_json_atomic(out_path, manifest)
    return manifest, scan_node


def _load_outcomes(base_dir: Path, manifest: dict) -> dict:
    """Load an artifact's per-example outcomes npz, digest-verified against
    the manifest (arrays keyed as written; digest covers array CONTENT, not
    npz bytes — zip timestamps are not reproducible)."""
    import numpy as np

    from subspaces.step1.recovery import outcomes_content_sha

    reference = manifest.get("outcomes")
    if not reference or "file" not in reference:
        raise ArtifactError(
            f"{base_dir}: the manifest records no per-example outcomes "
            "reference; the artifact cannot be treated as outcome-complete "
            "(needed for reuse verification and for outcome-based selectors)."
        )
    outcomes_path = base_dir / reference["file"]
    if not outcomes_path.is_file():
        raise ArtifactError(
            f"{outcomes_path} is missing but the manifest references it; "
            "refusing to reuse an incomplete artifact."
        )
    loaded = np.load(outcomes_path)
    arrays = {key: loaded[key] for key in loaded.files}
    actual = outcomes_content_sha(arrays)
    expected = reference["content_sha256"]
    if actual != expected:
        raise ArtifactError(
            f"{outcomes_path}: outcome content digest mismatch "
            f"({actual[:12]} vs recorded {expected[:12]}); refusing."
        )
    return arrays


def _verify_outcomes(base_dir: Path, manifest: dict) -> None:
    """A reused scan/headset artifact must still have its outcomes npz,
    content-intact (digest recorded in the manifest)."""
    _load_outcomes(base_dir, manifest)


def verify_selection_lineage(
    main_manifest: dict, scan_manifest: dict, selected: dict
) -> None:
    """A main-heads artifact must descend from THIS scan, and the scan from
    THIS selected artifact — enforced before any selected-set evaluation."""
    scan_ref = (main_manifest.get("inputs") or {}).get("head_scan")
    if not scan_ref:
        raise ArtifactError(
            "main_heads artifact has no head_scan input reference; refusing "
            "to evaluate an un-lineaged head set."
        )
    scan_fp = semantic_fingerprint(scan_manifest)
    if scan_ref.get("semantic_fingerprint") != scan_fp:
        raise ArtifactError(
            "lineage mismatch: the main-heads artifact descends from a "
            f"different scan ({str(scan_ref.get('semantic_fingerprint'))[:12]} "
            f"vs this run's {scan_fp[:12]}); refusing."
        )
    scan_selected_ref = (scan_manifest.get("inputs") or {}).get("selected_heads")
    selected_fp = semantic_fingerprint(selected)
    if (
        not scan_selected_ref
        or scan_selected_ref.get("semantic_fingerprint") != selected_fp
    ):
        raise ArtifactError(
            "lineage mismatch: the scan does not descend from this run's "
            "selected_heads artifact; refusing."
        )


def stage_select_main(
    cfg: Step1Config, paths: ProjectPaths, scan_node: Path
) -> tuple[dict, Path]:
    """Pure-CPU selection from the persisted scan, per the config's selector.

    Returns (manifest, main node dir); the artifact is
    ``<main-node>/main_heads.json``.
    """
    from subspaces.head_sets import parse_heads

    params = dict(cfg.main_selector.params)
    if cfg.main_selector.name == "pin":
        params["heads"] = [
            list(head) for head in parse_heads(cfg.main_selector.pin_heads)
        ]
    manifest, main_node = apply_selector(
        scan_node / "head_scan.json",
        selector_name=cfg.main_selector.name,
        selector_version=cfg.main_selector.version,
        params=params,
        paths=paths,
    )
    return manifest, main_node


def stage_evaluate_headset(
    cfg: Step1Config,
    paths: ProjectPaths,
    split: TaskSplit | None,
    matrix_node: Path,
    selected_node: Path,
    scan_node: Path,
    selected: dict,
    matrix_ref: dict,
    main_manifest: dict,
    main_node: Path,
    activation_samples: tuple[dict, Path],
    scan_samples: tuple[dict, Path],
    resolved: dict,
    model_loader,
    subspace_path: Path | None = None,
) -> dict:
    """Headset evaluation of a main selection. The ONLY stage that materializes held-out
    manifests/caches (after selection), per the corrected protocol. Output:
    ``<main-node>/eval-<h10>.json`` (+ outcomes npz) — final-eval protocol
    variants are sibling files inside the main node. With ``subspace_path``
    (a step2_subspace artifact over ⊇ the evaluated heads, fitted on exactly
    this evaluation's train+held-out z caches) the projected arm
    ``proj_meanab`` joins the arm set and the subspace becomes an explicit
    identity-bearing input."""
    from subspaces.artifacts import identity_of
    from subspaces.step1 import headset_eval as headset_mod
    from subspaces.step1.eval_gpu import mean_z_over_tasks
    from subspaces.step1.zcache import ensure_zcache

    # Lineage FIRST — before any held-out material is created (a foreign main
    # must be refused for free): (1) the on-disk scan must be THE scan this
    # run's config would produce (trust-on-disk closed: audit experiment 7b/7c);
    # (2) main -> that scan -> this run's selected set.
    scan_manifest = read_manifest(
        scan_node / "head_scan.json",
        expect_kind="head_scan",
        max_schema_version=HEAD_SCAN_SCHEMA_VERSION,
    )
    expected_scan = _scan_expected_identity(
        cfg,
        paths,
        selected_node / "selected_heads.json",
        selected,
        activation_samples[0],
        scan_samples[0],
        scan_samples[1],
        resolved,
    )
    if identity_of(scan_manifest) != identity_of(expected_scan):
        raise ArtifactError(
            "the on-disk head_scan.json was not produced by this run's "
            "configuration (scan identity mismatch); refusing to evaluate "
            "against a foreign or stale scan."
        )
    _verify_outcomes(scan_node, scan_manifest)
    verify_selection_lineage(main_manifest, scan_manifest, selected)

    subspace_manifest = None
    if subspace_path is not None:
        from subspaces.step2.pca import (
            STEP2_SUBSPACE_KIND,
            STEP2_SUBSPACE_SCHEMA_VERSION,
        )

        subspace_manifest = read_manifest(
            subspace_path,
            expect_kind=STEP2_SUBSPACE_KIND,
            max_schema_version=STEP2_SUBSPACE_SCHEMA_VERSION,
        )
        main_set = {
            (int(layer_idx), int(head_idx))
            for layer_idx, head_idx in main_manifest["main_heads"]
        }
        covered = {
            (int(layer_idx), int(head_idx))
            for layer_idx, head_idx in subspace_manifest["config"]["heads"]
        }
        uncovered = sorted(main_set - covered)
        if uncovered:
            raise ArtifactError(
                f"step2_subspace artifact {subspace_path} covers no PCA for "
                f"evaluated head(s) {uncovered}; recompute step 2 over the "
                "full selected set."
            )

    raw_source, raw_checkpoint = matrix_mod.resolve_final_coefficient_checkpoint(
        cfg, matrix_ref, paths
    )
    raw_checkpoint_input = file_ref(raw_checkpoint, paths)
    if raw_checkpoint_input["sha256"] != raw_source["sha256"]:
        raise ArtifactError(
            "raw_coef final checkpoint changed while its input reference was "
            "being built; refusing"
        )
    raw_checkpoint_input.update(
        epoch=raw_source["epoch"],
        selection_rule=raw_source["selection_rule"],
    )

    heldout_samples, heldout_path = _ensure_samples(
        cfg, cfg.samples.heldout_activation, "heldout_activation", split, paths
    )
    final_samples, final_path = _ensure_samples(
        cfg, cfg.samples.final_eval, "final_eval", split, paths
    )
    z_train, fp_train = ensure_zcache(
        model_loader, activation_samples[0], cfg, paths, resolved
    )
    train_mean_z = mean_z_over_tasks(z_train)
    z_heldout, fp_heldout = ensure_zcache(
        model_loader, heldout_samples, cfg, paths, resolved
    )

    inputs_refs = {
        "main_heads": manifest_ref(main_node / "main_heads.json", paths, main_manifest),
        "selected_heads": manifest_ref(
            selected_node / "selected_heads.json", paths, selected
        ),
        "matrix_ref": manifest_ref(matrix_node / "matrix_ref.json", paths, matrix_ref),
        "final_eval_samples": manifest_ref(final_path, paths, final_samples),
        "heldout_activation_samples": manifest_ref(
            heldout_path, paths, heldout_samples
        ),
        "z_cache_train": {"content_fingerprint": fp_train},
        "z_cache_heldout": {"content_fingerprint": fp_heldout},
        "raw_coef_checkpoint": raw_checkpoint_input,
    }
    if subspace_manifest is not None:
        # The projected arm's PCA fit set must be EXACTLY this evaluation's
        # two caches — a subspace fitted on other caches (other protocol,
        # other sampling) must be refused, not silently projected.
        subspace_fps = {
            cache["content_fingerprint"]
            for cache in subspace_manifest["inputs"]["z_caches"]
        }
        if subspace_fps != {fp_train, fp_heldout}:
            raise ArtifactError(
                "step2_subspace artifact was fitted on z caches "
                f"{sorted(subspace_fps)} but this evaluation uses "
                f"{sorted({fp_train, fp_heldout})}; refusing to project "
                "against a foreign fit set."
            )
        inputs_refs["pca_subspace"] = manifest_ref(
            subspace_path, paths, subspace_manifest
        )
    from subspaces.step1.eval_gpu import effective_batch_size

    headset_config = headset_mod.identity_config(
        cfg,
        main_manifest,
        effective_batch_size(cfg),
        projection=(
            headset_mod.projection_identity(subspace_manifest)
            if subspace_manifest is not None
            else None
        ),
    )
    expected = {
        "kind": "headset_eval",
        "schema_version": headset_mod.HEADSET_EVAL_SCHEMA_VERSION,
        "inputs": inputs_refs,
        "config": headset_config,
        "impl": headset_mod.IMPL,
    }
    # Name from the COMPLETE artifact identity, not only its inputs: scorer,
    # raw-coefficient policy, batch size, or implementation changes become
    # immutable sibling evaluations instead of colliding with an old file.
    artifact_stem = f"eval-{identity_of(expected)[:10]}"
    out_path = main_node / f"{artifact_stem}.json"
    existing = reuse_or_refuse(
        out_path,
        expected,
        expect_kind="headset_eval",
        max_schema_version=headset_mod.HEADSET_EVAL_SCHEMA_VERSION,
    )
    if existing is not None:
        _verify_outcomes(main_node, existing)
        return existing
    manifest = headset_mod.evaluate_headset(
        main_manifest,
        selected,
        matrix_ref,
        train_mean_z,
        {"z": z_heldout, "content_fingerprint": fp_heldout},
        final_samples,
        cfg,
        paths,
        main_node,
        resolved=resolved,
        inputs_refs=inputs_refs,
        model_loader=model_loader,
        artifact_stem=artifact_stem,
        subspace=subspace_manifest,
        train_z=z_train,
    )
    write_json_atomic(out_path, manifest)
    return manifest


def run(cfg: Step1Config, paths: ProjectPaths) -> dict:
    """Composed Step-1 run. Equivalent to the subcommands run sequentially."""
    journal_dir, split, resolved = ensure_run(cfg, paths)
    model_loader = make_model_loader(cfg, resolved)

    matrix_ref, matrix_node = stage_matrix(cfg, paths, journal_dir)
    selected, selected_node = stage_selected(cfg, paths, matrix_node, matrix_ref)
    selection_samples = stage_selection_samples(cfg, paths, split)
    _, scan_node = stage_scan(
        cfg,
        paths,
        selected_node,
        selected,
        selection_samples["activation"],
        selection_samples["scan"],
        resolved,
        model_loader,
    )
    main_manifest, main_node = stage_select_main(cfg, paths, scan_node)
    headset = stage_evaluate_headset(
        cfg,
        paths,
        split,
        matrix_node,
        selected_node,
        scan_node,
        selected,
        matrix_ref,
        main_manifest,
        main_node,
        selection_samples["activation"],
        selection_samples["scan"],
        resolved,
        model_loader,
    )
    compose_heads(
        selected_path=selected_node / "selected_heads.json",
        main_path=main_node / "main_heads.json",
        paths=paths,
        out_dir=main_node,
    )

    # The journal accumulates node pointers + selection summaries: selector
    # variants of one config land as SIBLING main-* nodes and stack up here.
    selection_key = main_node.name
    selection_summary = {
        "selector": {
            "name": cfg.main_selector.name,
            "version": cfg.main_selector.version,
            "params": main_manifest["params"],
        },
        "main_heads": main_manifest["main_heads"],
        "minor_heads": main_manifest["minor_heads"],
        "verdict": main_manifest.get("verdict"),
        "metrics": headset["metrics"],
        "per_task": headset["per_task"],
        "main_node": paths.relativize(main_node),
    }
    nodes_path = journal_dir / "nodes.json"
    selections: dict = {}
    if nodes_path.exists():
        selections = modernize(json.loads(nodes_path.read_text(encoding="utf-8"))).get(
            "selections", {}
        )
    selections[selection_key] = selection_summary
    summary = make_manifest(
        kind="step1_run",
        schema_version=2,
        paths=paths,
        config={"protocol": cfg.protocol},
        payload={
            "run_id": journal_dir.name,
            "selections": selections,
            "nodes": {
                "matrix": paths.relativize(matrix_node),
                "selected": paths.relativize(selected_node),
                "scan": paths.relativize(scan_node),
            },
        },
    )
    write_json_atomic(nodes_path, summary)
    # convenience echo of the selection just produced
    summary["latest_selection"] = selection_summary
    summary["metrics"] = headset["metrics"]
    summary["per_task"] = headset["per_task"]
    return summary
