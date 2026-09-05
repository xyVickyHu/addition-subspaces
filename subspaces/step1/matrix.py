"""Substep 1 — train or reuse the coefficient matrix.

Artifact: ``matrix_ref.json`` (kind ``matrix_ref``) inside the run's ``step1/``
directory, pointing at a matrix directory that contains ``checkpoints/
matrix_epoch<N>.pth``. The semantic identity hashes the SELECTED CHECKPOINT FILE,
not the containing directory, so reference matrices may live anywhere (inside the
repo, e.g. ``artifacts/matrix_add_0204_clip_lambda0.05``, or outside it via an
absolute path).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from pathlib import Path

from subspaces.artifacts import (
    ArtifactError,
    make_manifest,
    sha256_file,
    write_json_atomic,
)
from subspaces.config import Step1Config, to_dict
from subspaces.paths import ProjectPaths
from subspaces.step1.split import load_task_split

MATRIX_REF_SCHEMA_VERSION = 1
IMPL = {"module": "subspaces.step1.matrix", "algorithm_version": 1}

_EPOCH_RE = re.compile(r"^matrix_epoch(\d+)\.pth$")

# Historical dataset-dir names that are content-identical to current families
# (number_add_1018 == number_add: same examples in the same order; the byte
# difference is a trailing newline from the release whitespace pass).
PROVENANCE_TASK_ALIASES = {"number_add_1018": "number_add"}

# Archived-default mappings: historical args.json predates the prompt-format
# registry (missing prompt_format means the legacy arrow format) and encodes
# the injection site as a shorthand (layer "mid" is the paper's verified
# blocks.10.hook_resid_mid).
ARCHIVED_DEFAULT_PROMPT_FORMAT = "arrow"
ARCHIVED_LAYER_ALIASES = {"mid": "blocks.10.hook_resid_mid"}

# Fields compared strictly between the archived args.json and the current
# config; a mismatch refuses unless the field is named in
# matrix.override_provenance (recorded in the artifact).
_STRICT_PROVENANCE_FIELDS = (
    "model_name",
    "task_dir",
    "n_shot",
    "prompt_format",
    "inject_layer",
)


def find_checkpoints(matrix_dir: Path) -> list[tuple[int, Path]]:
    """(epoch, path) pairs sorted by epoch; mirrors legacy epoch-number sorting."""
    checkpoint_dir = matrix_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        raise ArtifactError(f"no checkpoints/ directory in matrix dir: {matrix_dir}")
    found = []
    for entry in checkpoint_dir.iterdir():
        match = _EPOCH_RE.match(entry.name)
        if match:
            found.append((int(match.group(1)), entry))
    if not found:
        raise ArtifactError(f"no matrix_epoch<N>.pth checkpoints in {checkpoint_dir}")
    return sorted(found)


def _archived_site(archived: dict) -> str | None:
    """Map the archived layer field to a hook name, or None if unmappable."""
    raw = archived.get("layer_name") or archived.get("layer")
    if raw is None:
        return None
    if isinstance(raw, str) and raw.startswith("blocks."):
        return raw
    return ARCHIVED_LAYER_ALIASES.get(raw)


def check_provenance(matrix_dir: Path, cfg: Step1Config) -> dict:
    """Validate a reused matrix against the current settings via args.json.

    Strictly compared (refuse on mismatch, unless the field is named in
    ``matrix.override_provenance``): model name, task family (through
    ``PROVENANCE_TASK_ALIASES``), n_shot, prompt format (archived default:
    arrow), and injection site (archived ``layer: "mid"`` maps to the verified
    ``blocks.10.hook_resid_mid``; an unmappable archived site also refuses).
    The full archived args.json SHA and recoverable fields are recorded.
    """
    args_path = matrix_dir / "args.json"
    if not args_path.is_file():
        return {
            "status": "unverified",
            "reason": "no args.json in matrix dir",
            "checked": {},
        }
    archived = json.loads(args_path.read_text(encoding="utf-8"))
    overrides = set(cfg.matrix.override_provenance)

    archived_task = archived.get("task_dir")
    archived_site = _archived_site(archived)
    current = {
        "model_name": cfg.model.name,
        "task_dir": cfg.task.dataset_dir,
        "n_shot": cfg.task.n_shot,
        "prompt_format": cfg.task.prompt_format,
        "inject_layer": cfg.sites.inject_layer,
    }
    trained = {
        "model_name": archived.get("model_name"),
        "task_dir": PROVENANCE_TASK_ALIASES.get(archived_task, archived_task),
        "n_shot": archived.get("n_shot"),
        "prompt_format": archived.get("prompt_format", ARCHIVED_DEFAULT_PROMPT_FORMAT),
        "inject_layer": archived_site,
    }
    mismatches = {
        key: {"trained": trained[key], "current": current[key]}
        for key in _STRICT_PROVENANCE_FIELDS
        if trained[key] != current[key] and key not in overrides
    }
    if mismatches:
        raise ArtifactError(
            f"matrix provenance mismatch for {matrix_dir}:\n"
            + "\n".join(
                f"  {key}: trained with {value['trained']!r}, current config "
                f"has {value['current']!r}"
                for key, value in mismatches.items()
            )
            + "\nRefusing to reuse a matrix trained under different settings "
            "(name the field in matrix.override_provenance to override "
            "explicitly for a research run)."
        )

    return {
        "status": "ok" if not overrides else "ok_with_overrides",
        "source": "args.json",
        "args_sha256": sha256_file(args_path),
        "checked": trained,
        "overridden": sorted(overrides),
        "recorded": {
            "task_dir_archived": archived_task,
            "layer_archived": archived.get("layer", archived.get("layer_name")),
            "n_example": archived.get("n_example"),
            "seed": archived.get("seed"),
            "lambdaL1": archived.get("lambdaL1"),
            "zero_one_interval": archived.get("zero_one_interval"),
            "train_ratio": archived.get("train_ratio"),
            "bs": archived.get("bs"),
            "epoch_num": archived.get("epoch_num"),
        },
    }


def _derive_mean_checkpoint(
    cfg: Step1Config,
    paths: ProjectPaths,
    matrix_dir: Path,
    checkpoints: list[tuple[int, Path]],
) -> tuple[dict, Path]:
    """Materialize the mean of the trailing k checkpoints under the matrix
    node's ``derived/`` dir; return (selected_checkpoint payload, effective
    matrix dir for loading).

    Identity is the TENSOR content digest (dtype/shape/bytes) — the bytes of
    a saved .pth are not contractual across torch versions, so the digest
    keys the node and re-materialization can never fork it. The saved file's
    byte sha256 is recorded as ``file_sha256`` (VOLATILE for fingerprints)
    and used for load verification only.

    Motivation (step-1 methods notes, Substep 2): single final checkpoints
    catch boundary heads mid-fall under the L1 pressure; the trailing mean
    restores the training-stable selection (canonical-33 / retrained-38)
    with ~10x cut margins.
    """
    import hashlib

    import torch

    from subspaces.step1 import tree

    select = cfg.matrix.checkpoint_select
    if select.k < 1:  # config validation re-checked: checkpoints[-0:] = ALL
        raise ArtifactError(f"matrix.checkpoint_select.k must be >= 1: {select.k}")
    if len(checkpoints) < select.k:
        raise ArtifactError(
            f"matrix.checkpoint_select.k={select.k} but only "
            f"{len(checkpoints)} checkpoint(s) in {matrix_dir}/checkpoints"
        )
    sources = checkpoints[-select.k :]
    epochs_list = [epoch for epoch, _ in sources]
    if epochs_list != list(range(epochs_list[0], epochs_list[0] + select.k)):
        raise ArtifactError(
            f"matrix.checkpoint_select (mean_last_k, k={select.k}) averages a "
            f"contiguous trailing window of epochs, but the trailing epochs in "
            f"{matrix_dir}/checkpoints are non-consecutive: {epochs_list}. "
            "Refusing (mirrors the train-path non-contiguity refusal) — the "
            "mean over a gapped window would not be the documented trailing "
            "mean."
        )
    tensors = []
    source_records = []
    for epoch, path in sources:
        tensor = torch.load(path, map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            raise ArtifactError(f"{path} did not contain a torch.Tensor")
        tensors.append(tensor.detach().to(torch.float32))
        source_records.append(
            {"file": path.name, "epoch": epoch, "sha256": sha256_file(path)}
        )
    shapes = {tuple(t.shape) for t in tensors}
    if len(shapes) != 1:
        raise ArtifactError(
            f"checkpoint_select sources disagree on shape: {sorted(shapes)}"
        )
    mean = torch.stack(tensors).mean(dim=0)
    digest = hashlib.sha256()
    digest.update(str(mean.dtype).encode())
    digest.update(repr(tuple(mean.shape)).encode())
    digest.update(mean.contiguous().numpy().tobytes())
    content_digest = digest.hexdigest()

    epochs = [epoch for epoch, _ in sources]
    node = paths.runs_dir / f"{tree.node_name(cfg)}__{content_digest[:12]}"
    derived_dir = node / "derived"
    filename = f"mean_last{select.k}_epoch{min(epochs)}-{max(epochs)}.pth"
    out_path = derived_dir / "checkpoints" / filename
    if out_path.is_file():
        # adopt only content-verified: the existing file's tensor must hash
        # to the digest just computed from the sources (tamper/crash window).
        try:
            existing = torch.load(out_path, map_location="cpu")
        except Exception as err:
            raise ArtifactError(
                f"derived checkpoint {out_path} is unreadable (partial/"
                f"corrupt write?): {err}\nDelete it to re-materialize from "
                "the source checkpoints."
            ) from err
        if not isinstance(existing, torch.Tensor):
            raise ArtifactError(
                f"derived checkpoint {out_path} did not contain a "
                "torch.Tensor; delete it to re-materialize from the source "
                "checkpoints."
            )
        have = hashlib.sha256()
        have.update(str(existing.dtype).encode())
        have.update(repr(tuple(existing.shape)).encode())
        have.update(existing.detach().contiguous().numpy().tobytes())
        if have.hexdigest() != content_digest:
            raise ArtifactError(
                f"{out_path} exists but its tensor content digest "
                f"({have.hexdigest()[:12]}) does not match the recomputed "
                f"source mean ({content_digest[:12]}); refusing."
            )
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Concurrency-safe unique temp + atomic replace. The saved BYTES are
        # not contractual (identity is content_digest); a concurrent winner
        # is simply overwritten with a content-identical tensor.
        fd, tmp_name = tempfile.mkstemp(
            dir=out_path.parent, prefix=f".{filename}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "wb") as fh:
                torch.save(mean, fh)
            os.replace(tmp_name, out_path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise
    selected = {
        "file": filename,
        "epoch": None,
        # load-verification only, VOLATILE for fingerprints (subspaces.artifacts):
        # the tensor content_digest below is the contractual identity.
        "file_sha256": sha256_file(out_path),
        "selection_rule": "mean_last_k",
        "k": select.k,
        "content_digest": content_digest,
        "source_matrix_dir": paths.relativize(matrix_dir),
        "source_checkpoints": source_records,
    }
    return selected, derived_dir


def check_split_provenance(
    matrix_dir: Path, cfg: Step1Config, paths: ProjectPaths
) -> None:
    """Refuse a reused matrix whose archived TRAINING split disagrees with
    the configured task split.

    Pure validation — it records nothing (the passing case must not alter
    ``provenance_check``, which is part of node identity). The archived
    ``indist_keys``/``ood_keys`` are the tasks the matrix actually trained
    on; if any configured EVAL task sits inside the archived training set,
    the "held-out" metrics would not be held out from matrix training
    (2026-07-27 finding: the historical phi-4 add matrix trained on a
    different random split, covering 4 of the committed 5 eval tasks).
    Named override: ``matrix.override_provenance: [task_split]`` — the
    override is recorded in the manifest's config slice.
    """
    if "task_split" in set(cfg.matrix.override_provenance):
        return
    args_path = matrix_dir / "args.json"
    if not args_path.is_file():
        return  # recorded as unverified by check_provenance
    archived = json.loads(args_path.read_text(encoding="utf-8"))
    indist, ood = archived.get("indist_keys"), archived.get("ood_keys")
    if not indist or not ood:
        return  # pre-schema archive: nothing to compare against
    if not cfg.task.task_split:
        return
    import yaml

    split = yaml.safe_load(
        paths.resolve(cfg.task.task_split).read_text(encoding="utf-8")
    )
    problems = []
    if set(indist) != set(split["train_tasks"]):
        problems.append(
            f"archived indist_keys ({len(indist)}) != split train_tasks "
            f"({len(split['train_tasks'])})"
        )
    leaked = sorted(set(indist) & set(split["eval_tasks"]))
    if leaked:
        problems.append(
            f"configured EVAL tasks inside the archived TRAINING set: {leaked}"
        )
    if set(ood) != set(split["eval_tasks"]):
        problems.append(
            f"archived ood_keys {sorted(ood)} != split eval_tasks "
            f"{sorted(split['eval_tasks'])}"
        )
    if problems:
        raise ArtifactError(
            f"matrix TRAINING-split provenance mismatch for {matrix_dir}:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
            + "\nThe reused matrix was trained on a different task split; "
            "its evaluation on the configured held-out tasks would not be "
            "held out from matrix training. Name task_split in "
            "matrix.override_provenance to accept this explicitly for a "
            "research run."
        )


def resolve_reuse_matrix(cfg: Step1Config, paths: ProjectPaths) -> dict:
    """Build the ``matrix_ref`` manifest for ``matrix.mode=reuse``."""
    if cfg.matrix.mode != "reuse":
        raise ArtifactError("resolve_reuse_matrix requires matrix.mode=reuse")
    matrix_dir = paths.resolve(cfg.matrix.reuse_path)
    if not matrix_dir.is_dir():
        raise ArtifactError(f"matrix.reuse_path not found: {matrix_dir}")
    # checkpoints first: while a train leg is still writing, "no checkpoints"
    # is the truthful refusal (args.json gets its prompt_format annotation
    # only after training completes — a provenance-first check would blame
    # the format instead).
    checkpoints = find_checkpoints(matrix_dir)
    provenance = check_provenance(matrix_dir, cfg)
    check_split_provenance(matrix_dir, cfg, paths)
    effective_dir = matrix_dir
    if cfg.matrix.checkpoint_select is not None:
        selected, effective_dir = _derive_mean_checkpoint(
            cfg, paths, matrix_dir, checkpoints
        )
    elif cfg.matrix.checkpoint_epoch is not None:
        by_epoch = dict(checkpoints)
        if cfg.matrix.checkpoint_epoch not in by_epoch:
            raise ArtifactError(
                f"matrix.checkpoint_epoch={cfg.matrix.checkpoint_epoch} not "
                f"found in {matrix_dir}/checkpoints (available: "
                f"{sorted(by_epoch)})"
            )
        epoch = cfg.matrix.checkpoint_epoch
        checkpoint = by_epoch[epoch]
        selected = {
            "file": checkpoint.name,
            "epoch": epoch,
            "sha256": sha256_file(checkpoint),
            "selection_rule": "pinned",
        }
    else:
        epoch, checkpoint = checkpoints[-1]
        selected = {
            "file": checkpoint.name,
            "epoch": epoch,
            "sha256": sha256_file(checkpoint),
            "selection_rule": "latest",
        }
    return make_manifest(
        kind="matrix_ref",
        schema_version=MATRIX_REF_SCHEMA_VERSION,
        paths=paths,
        config={"matrix": to_dict(cfg.matrix)},
        payload={
            "impl": IMPL,
            "matrix_dir": paths.relativize(effective_dir),
            "provenance": "reuse",
            "provenance_check": provenance,
            "selected_checkpoint": selected,
        },
    )


def resolve_final_coefficient_checkpoint(
    cfg: Step1Config, matrix_ref: dict, paths: ProjectPaths
) -> tuple[dict, Path]:
    """Resolve the immutable coefficient source for ``raw_coef``.

    Head selection may use a pinned checkpoint or a derived trailing mean.
    ``raw_coef`` is deliberately independent of that choice: it always uses
    every coefficient from the final epoch checkpoint in the source matrix
    directory.  The returned record is suitable for artifact provenance; the
    caller should also retain a content-hashed file reference.
    """
    selected = matrix_ref.get("selected_checkpoint") or {}
    if cfg.matrix.mode == "reuse":
        if not cfg.matrix.reuse_path:
            raise ArtifactError(
                "raw_coef final-checkpoint resolution requires "
                "matrix.reuse_path for a reused matrix"
            )
        source_dir = paths.resolve(cfg.matrix.reuse_path)
    else:
        source_dir = paths.resolve(matrix_ref["matrix_dir"])

    epoch, checkpoint = find_checkpoints(source_dir)[-1]
    record = {
        "selection_rule": "final_checkpoint",
        "file": checkpoint.name,
        "epoch": epoch,
        "sha256": sha256_file(checkpoint),
    }

    # Cross-check against the matrix_ref where it recorded what the final
    # checkpoint WAS at resolution time; a disagreement means the matrix
    # reference or its source directory changed afterwards (e.g. training
    # extended the epoch range) and evaluation would mix incompatible
    # provenance.  A trailing mean records the exact source window (its last
    # member must still be the final checkpoint); `latest` and
    # `trained_final_epoch` record the final checkpoint itself.  A `pinned`
    # selection may legitimately be a non-final epoch, so it carries no
    # record of the final checkpoint to check against — the epoch+sha in the
    # headset input reference keeps it auditable.
    rule = selected.get("selection_rule")
    recorded_final = None
    if rule == "mean_last_k":
        sources = selected.get("source_checkpoints") or []
        if not sources:
            raise ArtifactError(
                "derived matrix_ref lacks source_checkpoints; cannot resolve "
                "the final-checkpoint coefficients for raw_coef"
            )
        recorded_final = max(sources, key=lambda item: int(item["epoch"]))
    elif rule in ("latest", "trained_final_epoch"):
        recorded_final = selected
    if recorded_final is not None:
        mismatches = {
            key: {"recorded": recorded_final.get(key), "actual": record[key]}
            for key in ("file", "epoch", "sha256")
            if recorded_final.get(key) != record[key]
        }
        if mismatches:
            raise ArtifactError(
                "raw_coef final-checkpoint source disagrees with the "
                f"matrix_ref's {rule} provenance: {mismatches}; refusing"
            )
    return record, checkpoint


def _checkpoint_epochs(matrix_dir: Path) -> set[int]:
    checkpoint_dir = matrix_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        return set()
    return {
        int(match.group(1))
        for entry in checkpoint_dir.iterdir()
        if (match := _EPOCH_RE.match(entry.name))
    }


def _trainer_argv(cfg: Step1Config, split, version: str) -> list[str]:
    """Explicit argv for the legacy trainer — nothing inferred, nothing drawn.

    The committed split supplies indist/ood keys VERBATIM (file order = the
    historical key order; the trainer's random-split fallback is unreachable).
    ``--savevar`` stays off: the legacy z cache is keyed without the model —
    the documented collision hazard. ``--run_name``/``--add_mean``/
    ``--nonnegative``/``--corrupted`` keep their canonical defaults.
    """
    train = cfg.matrix.train
    argv = [
        "subspaces.runners.train_matrix",
        "--version",
        version,
        "--task_dir",
        cfg.task.dataset_dir,
        "--indist_keys",
        json.dumps(list(split.train_tasks)),
        "--ood_keys",
        json.dumps(list(split.eval_tasks)),
        "--ood_task_num",
        str(len(split.eval_tasks)),
        "--indist_limit",
        str(train.indist_limit),
        "--train_ratio",
        str(train.train_ratio),
        "--ood_limit",
        str(train.ood_limit),
        "--bs",
        str(train.bs),
        "--lr",
        str(train.lr),
        "--epoch_num",
        str(train.epochs),
        "--test_gap",
        str(train.test_gap),
        "--n_shot",
        str(cfg.task.n_shot),
        "--n_example",
        str(train.n_example),
        "--seed",
        str(train.seed),
        "--lambdaL1",
        str(train.lambda_l1),
        "--anneal",
        str(train.anneal),
        "--model_name",
        cfg.model.name,
        "--layer_name",
        cfg.sites.inject_layer,
    ]
    if train.zero_one_interval:
        argv.append("--zero_one_interval")
    return argv


def _invoke_legacy_trainer(
    cfg: Step1Config, paths: ProjectPaths, matrix_dir: Path, argv: list[str]
) -> None:
    """Run the legacy trainer in-process with explicit argv (zero drift).

    The legacy ``main()`` parses ``sys.argv`` and rebinds ``sys.stdout``/
    ``sys.stderr`` to ``<matrix_dir>/output.txt`` (``create_log_file``) —
    both are restored afterwards, along with ``sys.argv``. wandb mode is
    enforced from the config when the environment does not already set it
    (the legacy trainer calls ``wandb.login()`` unconditionally; offline it
    is a no-op).
    """
    import sys

    # cfg.compute.wandb_mode would otherwise be a dead field: the legacy
    # trainer only honors $WANDB_MODE (verifier finding). The launcher's
    # explicit export still wins.
    os.environ.setdefault("WANDB_MODE", cfg.compute.wandb_mode)

    if cfg.model.device == "cuda":
        import torch  # heavy

        if not torch.cuda.is_available():
            raise ArtifactError(
                "config requests model.device=cuda but CUDA is unavailable; "
                "refusing to start an 8B-model training on CPU."
            )

    if cfg.model.revision is not None:
        from subspaces.step1.resolve import resolve_model_identity

        observed = resolve_model_identity(cfg.model.name)
        if observed["revision"] != cfg.model.revision:
            raise ArtifactError(
                f"the local cache resolves {cfg.model.name} to revision "
                f"{observed['revision']} but the run pins "
                f"{cfg.model.revision}; refusing to train on drifted weights."
            )

    import importlib

    wandb = importlib.import_module("wandb")  # heavy
    legacy_trainer = importlib.import_module("subspaces.runners.train_matrix")  # heavy

    trainer_root = Path(legacy_trainer.PROJECT_ROOT).resolve()
    if trainer_root != paths.root.resolve():
        raise ArtifactError(
            f"legacy trainer PROJECT_ROOT {trainer_root} does not match the "
            f"project root {paths.root}; the trainer would write elsewhere."
        )

    real_argv, real_stdout, real_stderr = sys.argv, sys.stdout, sys.stderr
    sys.argv = argv
    try:
        legacy_trainer.main()
    finally:
        # best-effort cleanup — never mask the training error
        if getattr(wandb, "run", None) is not None:
            with contextlib.suppress(Exception):
                wandb.finish()
        if sys.stdout is not real_stdout:
            with contextlib.suppress(Exception):
                _retarget_logging_handlers((sys.stdout, sys.stderr), real_stderr)
            with contextlib.suppress(Exception):
                sys.stdout.close()
        sys.argv = real_argv
        sys.stdout = real_stdout
        sys.stderr = real_stderr


def _retarget_logging_handlers(old_streams: tuple, new_stream) -> None:
    """Point ``logging`` StreamHandlers away from streams about to close.

    Libraries that construct handlers DURING training (basicConfig et al.)
    capture the redirected ``output.txt`` stream object; closing it would
    turn every later ``logging`` call in the same process into
    ``--- Logging error --- ... I/O operation on closed file`` noise and
    swallow the message (observed in a GPU run when the scan stage
    reloaded the model after training).
    """
    import logging

    loggers = [logging.getLogger()]
    loggers += [
        logger
        for logger in logging.Logger.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    ]
    for logger in loggers:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and any(
                getattr(handler, "stream", None) is stream for stream in old_streams
            ):
                # per-handler: one read-only-stream handler (e.g. a
                # _StderrHandler subclass) must not abort the remaining
                # retargets (review finding F3)
                with contextlib.suppress(Exception):
                    handler.setStream(new_stream)


def _annotate_args_json(matrix_dir: Path, cfg: Step1Config) -> None:
    """Record the effective prompt format in the trainer's args.json.

    The legacy trainer has no ``--prompt_format`` flag — it renders through
    ``$FV_PROMPT_FORMAT`` (guarded against the config before training). The
    archived-args default ("missing means arrow") would mis-describe a
    non-arrow training run, so the verified effective format is recorded
    explicitly; ``check_provenance`` and the args sha256 then cover it.
    """
    args_path = matrix_dir / "args.json"
    if not args_path.is_file():
        return
    archived = json.loads(args_path.read_text(encoding="utf-8"))
    if "prompt_format" not in archived:
        archived["prompt_format"] = cfg.task.prompt_format
        write_json_atomic(args_path, archived)


def _validate_checkpoint(path: Path) -> None:
    """torch.load-validate a checkpoint (torch.save is not atomic: a kill
    mid-save leaves a well-named partial file that would otherwise wedge the
    run at the selected substep or poison a resume)."""
    import torch  # heavy

    try:
        torch.load(path, map_location="cpu")
    except Exception as err:
        raise ArtifactError(
            f"checkpoint {path} is unreadable (partial/corrupt write?): "
            f"{err}\nDelete it (and any later epochs) to retrain from the "
            "last good checkpoint."
        ) from err


def train_matrix(cfg: Step1Config, paths: ProjectPaths, run_dir: Path) -> dict:
    """Train the coefficient matrix under this run (``step1/matrix/``).

    The numerical core is the LEGACY trainer (``subspaces.runners.train_matrix``),
    driven in-process with explicit arguments — zero drift by construction,
    the same policy as the selected/eval substeps. The fixed committed
    split supplies the indist/ood keys; training monitors the held-out (OOD)
    tasks exactly as the historical procedure did — diagnostics only: the
    checkpoint choice is the fixed final epoch, never validation-based, so
    no held-out signal enters selection. Resumable: existing epoch
    checkpoints are kept and training continues from the last one (legacy
    ``--resume`` semantics); a fully trained matrix dir skips training.
    """
    train = cfg.matrix.train
    if train is None:
        raise ArtifactError("matrix.mode=train requires the matrix.train block")
    if cfg.model.dtype != "bfloat16":
        raise ArtifactError(
            "the legacy trainer loads the model in bfloat16; matrix.mode=train "
            f"requires model.dtype=bfloat16 (got {cfg.model.dtype!r})"
        )
    # empty string == unset, matching the renderer and resolve.py semantics
    env_format = os.environ.get("FV_PROMPT_FORMAT") or "arrow"
    if env_format != cfg.task.prompt_format:
        raise ArtifactError(
            "the legacy trainer renders training prompts via $FV_PROMPT_FORMAT "
            f"(effective {env_format!r}) but the config requests "
            f"{cfg.task.prompt_format!r}; export FV_PROMPT_FORMAT to match in "
            "the training job."
        )
    if cfg.task.task_split is None:
        raise ArtifactError(
            "matrix.mode=train requires task.task_split (the committed split "
            "supplies the historical indist/ood keys; never seed-generated)"
        )
    split = load_task_split(cfg.task.task_split, paths)
    if split.family != cfg.task.dataset_dir:
        raise ArtifactError(
            f"task split family {split.family!r} does not match "
            f"task.dataset_dir {cfg.task.dataset_dir!r}; refusing."
        )

    matrix_dir = run_dir / "step1" / "matrix"
    checkpoint_dir = matrix_dir / "checkpoints"
    wanted = set(range(train.epochs))
    have = _checkpoint_epochs(matrix_dir)
    if not wanted <= have:
        if checkpoint_dir.is_dir():
            # the legacy resume sort silently degrades on foreign filenames
            stray = sorted(
                entry.name
                for entry in checkpoint_dir.iterdir()
                if not _EPOCH_RE.match(entry.name)
            )
            if stray:
                raise ArtifactError(
                    f"unexpected entries in {checkpoint_dir}: {stray}; the "
                    "legacy resume sort would silently misbehave — clean "
                    "the directory first."
                )
        if have and sorted(have) != list(range(len(have))):
            raise ArtifactError(
                f"non-contiguous checkpoint epochs {sorted(have)} in "
                f"{checkpoint_dir}; the legacy trainer resumes from the "
                "highest epoch and would never fill the gaps — remove the "
                "checkpoints after the gap (or the whole directory)."
            )
        if have:  # resume source must be loadable, not a partial write
            _validate_checkpoint(checkpoint_dir / f"matrix_epoch{max(have)}.pth")
        log_root = (paths.root / "log").resolve()
        version = str(matrix_dir.resolve().relative_to(log_root))
        argv = _trainer_argv(cfg, split, version)
        if have:  # partial training: continue from the last checkpoint
            argv += ["--resume", f"resume-{run_dir.name}"]
        _invoke_legacy_trainer(cfg, paths, matrix_dir, argv)
        have = _checkpoint_epochs(matrix_dir)
        missing = sorted(wanted - have)
        if missing:
            raise ArtifactError(
                f"training ended without epoch checkpoint(s) {missing} under "
                f"{matrix_dir}/checkpoints; refusing to hand off an "
                "incomplete matrix."
            )
    _annotate_args_json(matrix_dir, cfg)
    provenance = check_provenance(matrix_dir, cfg)
    epoch = train.epochs - 1
    checkpoint = matrix_dir / "checkpoints" / f"matrix_epoch{epoch}.pth"
    _validate_checkpoint(checkpoint)
    return make_manifest(
        kind="matrix_ref",
        schema_version=MATRIX_REF_SCHEMA_VERSION,
        paths=paths,
        config={"matrix": to_dict(cfg.matrix)},
        payload={
            "impl": IMPL,
            "matrix_dir": paths.relativize(matrix_dir),
            "provenance": "trained",
            "provenance_check": provenance,
            "selected_checkpoint": {
                "file": checkpoint.name,
                "epoch": epoch,
                "sha256": sha256_file(checkpoint),
                "selection_rule": "trained_final_epoch",
            },
        },
    )
