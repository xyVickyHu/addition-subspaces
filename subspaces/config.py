"""Typed configuration: YAML validation and resolved-config output.

A config file fully describes one Step-1 variant (model, task family, prompt
format, sites, matrix training, sample protocol, selected selector, scan,
main selector, seeds, compute). Loading rejects unknown keys and wrong types;
every run dumps its fully resolved config next to its outputs.

Explicit CLI overrides are applied by the runners onto the loaded dataclasses
(no generic ``--set key=value`` mechanism).
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

SCHEMA_VERSION = 1

PROTOCOLS = ("corrected_holdout", "legacy_reproduction")
TASK_SETS = ("train", "eval", "all")

# run_name becomes a directory-name component; keep it shell/path-safe.
RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


class ConfigError(ValueError):
    pass


@dataclass
class ModelConfig:
    name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    revision: str | None = None  # resolved to the immutable snapshot hash at runtime
    dtype: str = "bfloat16"
    device: str = "cuda"


@dataclass
class TaskConfig:
    dataset_dir: str = "number_add"
    prompt_format: str = "arrow"
    n_shot: int = 5
    # Path (repo-relative) to a committed task-split file; never seed-generated.
    # Required for protocol=corrected_holdout; None means all tasks (legacy).
    task_split: str | None = None


@dataclass
class SitesConfig:
    inject_layer: str = "blocks.10.hook_resid_mid"
    # NOTE: the z read site is fixed by the extraction implementation
    # (attn.hook_result at the last token, recorded in the z-cache identity);
    # it is deliberately not configurable here.


@dataclass
class MatrixTrainConfig:
    """Canonical-run values (archived args.json of the 0204 matrix; lr and
    anneal are unrecorded there — 0.01 / constant-λ are the inferred ported
    defaults, step-1 methods notes Conflicts #3). All fields join the run
    identity through requested semantics."""

    lambda_l1: float = 0.05
    epochs: int = 50
    lr: float = 0.01
    bs: int = 128
    train_ratio: float = 0.8
    zero_one_interval: bool = True
    seed: int = 42
    test_gap: int = 5
    n_example: int = 100
    indist_limit: int = 2560
    ood_limit: int = 512
    anneal: int = 1  # 1 = constant λ (canonical); 2/3 = legacy ramp schedules


# Provenance fields that may be explicitly overridden for research runs.
# Every override is named in the config and recorded in the artifact.
OVERRIDABLE_PROVENANCE_FIELDS = (
    "model_name",
    "task_dir",
    "n_shot",
    "prompt_format",
    "inject_layer",
    # archived training split (indist/ood keys) vs the configured task_split
    # (guard added 2026-07-27 after the phi-4 add finding: its historical
    # matrix trained on a different random split, putting 4 of the 5
    # committed eval tasks inside its training set)
    "task_split",
)


@dataclass
class CheckpointSelectConfig:
    """Derived effective checkpoint (reuse branch): a rule over the trailing
    checkpoints instead of one pinned epoch. ``mean_last_k`` averages the last
    ``k`` epoch checkpoints; the mean is materialized as a content-addressed
    derived checkpoint with its own sha256 and full source provenance
    (motivation: single final checkpoints catch boundary heads mid-fall —
    step-1 methods notes Substep 2, largest_gap near-tie)."""

    rule: str = "mean_last_k"
    k: int = 3


@dataclass
class MatrixConfig:
    """Discriminated by ``mode``: exactly the active branch may be set."""

    mode: str = "reuse"  # reuse | train
    reuse_path: str | None = None  # reuse branch: matrix dir (absolute supported)
    # reuse branch: pin the exact checkpoint epoch (None = latest, recorded)
    checkpoint_epoch: int | None = None
    # reuse branch: derive the effective checkpoint (XOR with checkpoint_epoch)
    checkpoint_select: CheckpointSelectConfig | None = None
    # lineage-tree matrix-node name (presentation only; default derived from
    # the reuse_path basename, or run_name under mode=train)
    node_name: str | None = None
    # reuse branch: named provenance fields to skip (research override; recorded)
    override_provenance: list[str] = field(default_factory=list)
    train: MatrixTrainConfig | None = None  # train branch


@dataclass
class SampleSpec:
    examples_per_task: int
    seed: int = 42
    task_set: str = "train"  # train | eval | all


@dataclass
class SamplesConfig:
    activation: SampleSpec = field(
        default_factory=lambda: SampleSpec(examples_per_task=100, task_set="train")
    )
    # Corrected default: all 100 unique queries per task. 20/task is reserved
    # for the runtime-calibration config.
    scan: SampleSpec = field(
        default_factory=lambda: SampleSpec(examples_per_task=100, task_set="train")
    )
    # Held-out task-conditioned activation estimation (z for the selected
    # heads), created only AFTER selection by evaluate-headset. Distinct seed
    # from final_eval so evaluation prompts differ from z-estimation prompts.
    heldout_activation: SampleSpec = field(
        default_factory=lambda: SampleSpec(
            examples_per_task=100, seed=1043, task_set="eval"
        )
    )
    final_eval: SampleSpec = field(
        default_factory=lambda: SampleSpec(examples_per_task=100, task_set="eval")
    )
    # AIE-corrupted prompts (subspaces/step1/aie.py): train-task n-shot prompts whose
    # demo LABELS are permuted within-prompt (query + target intact). The
    # defaults mirror AIEConfig's corrupted_* fields so existing configs need
    # no edits; validation refuses a contradiction between the two blocks.
    aie_corrupted: SampleSpec = field(
        default_factory=lambda: SampleSpec(examples_per_task=25, task_set="train")
    )


@dataclass
class AIEConfig:
    """Todd et al. (arXiv:2310.15213) AIE baseline (``subspaces/step1/aie.py``):
    per-head average-indirect-effect scores on the committed TRAIN tasks,
    top-k selection by SIGNED score, and Todd-FV evaluation on the held-out
    tasks under the standard eval protocol. A parallel, fully fingerprinted
    artifact chain — it never enters the selected→scan→main lineage, and
    the whole block is excluded from run identity (``requested_semantics``)
    like ``main_selector``: variants live side-by-side as hash-named
    artifacts under one journal run."""

    # registered in subspaces.step1.aie.METHOD_VERSIONS
    methods: list[str] = field(default_factory=lambda: ["cie_replace", "zs_add_proxy"])
    corrupted_examples_per_task: int = 25
    corrupted_seed: int = 42
    corruption_tries: int = 20
    # no default: every cell names its k values explicitly (selected-matched +
    # paper-indicated counts; see configs/step1_*_aie.yaml)
    k_values: list[int] = field(default_factory=list)
    batch_size: int = 25  # cie_replace scoring batch (corrupted prompts)
    proxy_batch_size: int = 100  # zs_add_proxy scoring batch (zero-shot)
    # flat row-major cap over (layer_idx, head_idx) — smoke configs ONLY
    head_limit: int | None = None


@dataclass
class SelectedConfig:
    """Discriminated by ``(method, method_version)`` against the registry in
    ``subspaces.step1.selected.METHOD_PARAM_SCHEMAS``: only the method's own
    parameters may be set."""

    method: str = "elbow"  # elbow | fraction | fixed | largest_gap
    method_version: int = 1
    fraction: float | None = None  # fraction branch only
    fixed_threshold: float | None = None  # fixed branch only


@dataclass
class ScanConfig:
    # c_min is fixed at 0: the selectors define recovery gain against the
    # c=0 leave-one-out baseline (accs[0]); a shifted grid would silently
    # redefine every gain. Kept as a field so artifacts record it explicitly.
    c_min: int = 0
    c_max: int = 20
    # Debug/smoke knob: scan only the top-N selected heads (None = all).
    head_limit: int | None = None


def _default_unified_params() -> dict[str, float]:
    # Single source of truth for the adopted standard constants; lazy import
    # (subspaces.step1 modules import this module — same pattern as resolve_runtime).
    from subspaces.step1.selectors import UNIFIED_V1_PARAMS

    return dict(UNIFIED_V1_PARAMS)


@dataclass
class SelectorConfig:
    name: str = "unified"
    version: int = 1
    # Complete parameters, recorded explicitly in artifacts (no hidden
    # defaults); validated at load against the selector's exact schema.
    params: dict[str, float] = field(default_factory=_default_unified_params)
    # pin selector only: explicit 'L:H,...' list.
    pin_heads: str | None = None


@dataclass
class ScoringConfig:
    """Versioned generation/scoring contract (subspaces/step1/scoring.py registry).

    The scorer joins run identity (requested semantics) and every
    scan/headset artifact's config block — accuracies are never reported
    without the rule that produced them."""

    name: str = "exact_token_match"
    version: int = 1
    params: dict[str, float] | None = None


@dataclass
class RawCoefConfig:
    """Scientific contract for the ``raw_coef`` headset metric.

    ``raw_coef`` is the training-time function vector: every matrix head is
    weighted by its coefficient from the final training checkpoint.  The
    trailing checkpoint mean may be used to select selected heads, but it
    is never the coefficient source for this metric.
    """

    coefficient_source: str = "final_checkpoint"
    head_scope: str = "all"


@dataclass
class ComputeConfig:
    batch_size: int | None = None  # effective batch size is recorded per run
    wandb_mode: str = "offline"


@dataclass
class Step1Config:
    run_name: str = "step1"
    protocol: str = "corrected_holdout"
    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    sites: SitesConfig = field(default_factory=SitesConfig)
    matrix: MatrixConfig = field(default_factory=MatrixConfig)
    samples: SamplesConfig = field(default_factory=SamplesConfig)
    selected: SelectedConfig = field(default_factory=SelectedConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    main_selector: SelectorConfig = field(default_factory=SelectorConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    raw_coef: RawCoefConfig = field(default_factory=RawCoefConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    # optional AIE baseline block; absent → the aie subcommand refuses
    aie: AIEConfig | None = None

    def validate(self) -> None:
        if not RUN_NAME_RE.fullmatch(self.run_name):
            raise ConfigError(
                "run_name must match [A-Za-z0-9][A-Za-z0-9._-]{0,63} (it becomes "
                f"a directory-name component): {self.run_name!r}"
            )
        if self.protocol not in PROTOCOLS:
            raise ConfigError(f"protocol must be one of {PROTOCOLS}: {self.protocol}")
        # -- discriminated matrix schema --
        if self.matrix.mode == "reuse":
            if not self.matrix.reuse_path:
                raise ConfigError("matrix.mode=reuse requires matrix.reuse_path")
            if self.matrix.train is not None:
                raise ConfigError(
                    "matrix.mode=reuse must not set matrix.train (discriminated "
                    "schema: only the active branch may be present)"
                )
            if (
                self.matrix.checkpoint_epoch is not None
                and self.matrix.checkpoint_epoch < 0
            ):
                raise ConfigError("matrix.checkpoint_epoch must be >= 0")
            unknown_overrides = set(self.matrix.override_provenance) - set(
                OVERRIDABLE_PROVENANCE_FIELDS
            )
            if unknown_overrides:
                raise ConfigError(
                    f"matrix.override_provenance: unknown field(s) "
                    f"{sorted(unknown_overrides)}; allowed: "
                    f"{OVERRIDABLE_PROVENANCE_FIELDS}"
                )
            if self.matrix.checkpoint_select is not None:
                if self.matrix.checkpoint_epoch is not None:
                    raise ConfigError(
                        "matrix.checkpoint_epoch and matrix.checkpoint_select "
                        "are mutually exclusive (pin one epoch OR derive)"
                    )
                select = self.matrix.checkpoint_select
                if select.rule != "mean_last_k":
                    raise ConfigError(
                        "matrix.checkpoint_select.rule must be mean_last_k: "
                        f"{select.rule}"
                    )
                if select.k < 1:
                    raise ConfigError("matrix.checkpoint_select.k must be >= 1")
        elif self.matrix.mode == "train":
            if self.matrix.train is None:
                raise ConfigError("matrix.mode=train requires the matrix.train block")
            if self.matrix.reuse_path is not None:
                raise ConfigError("matrix.mode=train must not set matrix.reuse_path")
            if self.matrix.checkpoint_epoch is not None:
                raise ConfigError(
                    "matrix.mode=train must not set matrix.checkpoint_epoch"
                )
            if self.matrix.checkpoint_select is not None:
                raise ConfigError(
                    "matrix.mode=train must not set matrix.checkpoint_select "
                    "(train first, then average via a reuse config pointing at "
                    "the trained matrix dir)"
                )
            if self.matrix.override_provenance:
                raise ConfigError(
                    "matrix.mode=train must not set matrix.override_provenance"
                )
            train = self.matrix.train
            for name, minimum in (
                ("epochs", 1),
                ("bs", 1),
                ("test_gap", 1),
                ("n_example", 1),
                ("indist_limit", 1),
                ("ood_limit", 1),
            ):
                if getattr(train, name) < minimum:
                    raise ConfigError(f"matrix.train.{name} must be >= {minimum}")
            if not 0 < train.train_ratio < 1:
                raise ConfigError("matrix.train.train_ratio must be in (0, 1)")
            if train.lr <= 0:
                raise ConfigError("matrix.train.lr must be > 0")
            if train.lambda_l1 < 0:
                raise ConfigError("matrix.train.lambda_l1 must be >= 0")
            if train.anneal not in (1, 2, 3):
                raise ConfigError("matrix.train.anneal must be 1, 2, or 3")
        else:
            raise ConfigError(f"matrix.mode must be reuse|train: {self.matrix.mode}")
        if self.matrix.node_name is not None and not RUN_NAME_RE.fullmatch(
            self.matrix.node_name
        ):
            raise ConfigError(
                "matrix.node_name must match [A-Za-z0-9][A-Za-z0-9._-]{0,63} "
                f"(it becomes a directory-name component): {self.matrix.node_name!r}"
            )
        # -- prompt format: registry-validated at load (a typo would otherwise
        # pass the env-adoption + contradiction guards vacuously and fail as
        # a raw KeyError on the GPU node — review finding F1) --
        from subspaces.step1.samples import _load_prompt_format_registry

        known_formats = set(_load_prompt_format_registry().FORMATS)
        if self.task.prompt_format not in known_formats:
            raise ConfigError(
                f"task.prompt_format {self.task.prompt_format!r} is not a "
                f"registered format: {sorted(known_formats)}"
            )
        # -- scoring contract (registry-validated at load) --
        from subspaces.step1.scoring import ScoringError, validate_scoring

        try:
            validate_scoring(
                self.scoring.name, self.scoring.version, self.scoring.params
            )
        except ScoringError as err:
            raise ConfigError(f"scoring: {err}") from err
        if self.raw_coef.coefficient_source != "final_checkpoint":
            raise ConfigError(
                "raw_coef.coefficient_source must be final_checkpoint: "
                f"{self.raw_coef.coefficient_source!r}"
            )
        if self.raw_coef.head_scope != "all":
            raise ConfigError(
                f"raw_coef.head_scope must be all: {self.raw_coef.head_scope!r}"
            )
        # -- selector schema (validated here, not only at the CLI, so composed
        # runs get the same guarantee) --
        from subspaces.step1.selectors import SELECTOR_PARAM_SCHEMAS

        selector_key = (self.main_selector.name, self.main_selector.version)
        if selector_key not in SELECTOR_PARAM_SCHEMAS:
            known = ", ".join(f"{n} v{v}" for n, v in sorted(SELECTOR_PARAM_SCHEMAS))
            raise ConfigError(
                f"unknown main_selector {self.main_selector.name!r} "
                f"v{self.main_selector.version} (known: {known})"
            )
        expected_params = SELECTOR_PARAM_SCHEMAS[selector_key]
        actual_params = frozenset(self.main_selector.params)
        if actual_params != expected_params:
            raise ConfigError(
                f"main_selector {self.main_selector.name} "
                f"v{self.main_selector.version} requires EXACTLY the parameters "
                f"{sorted(expected_params)}; got {sorted(actual_params)}"
            )
        if self.main_selector.name == "pin":
            if not self.main_selector.pin_heads:
                raise ConfigError("main_selector pin requires pin_heads 'L:H,...'")
            from subspaces.head_sets import HeadSpecError, parse_heads

            try:
                parse_heads(self.main_selector.pin_heads)
            except HeadSpecError as err:
                raise ConfigError(f"main_selector.pin_heads: {err}") from err
        elif self.main_selector.pin_heads is not None:
            raise ConfigError(
                "main_selector.pin_heads is only valid with the pin selector"
            )
        # -- discriminated selected schema (registry-validated at load) --
        from subspaces.step1.selected import METHOD_PARAM_SCHEMAS

        selected = self.selected
        selected_key = (selected.method, selected.method_version)
        if selected_key not in METHOD_PARAM_SCHEMAS:
            registered = ", ".join(
                f"{name} v{version}" for name, version in sorted(METHOD_PARAM_SCHEMAS)
            )
            raise ConfigError(
                f"unknown selected-set method {selected.method!r} "
                f"v{selected.method_version} (registered: {registered})"
            )
        allowed_params = METHOD_PARAM_SCHEMAS[selected_key]
        # the parameter universe derives from the dataclass, so a future
        # method's new parameter field is checked without touching this loop
        selected_params = [
            f.name
            for f in dataclasses.fields(selected)
            if f.name not in ("method", "method_version")
        ]
        missing_fields = allowed_params - set(selected_params)
        if missing_fields:  # registry names a param with no config field
            raise ConfigError(
                f"selected-set method {selected.method!r} declares parameter(s) "
                f"{sorted(missing_fields)} with no SelectedConfig field"
            )
        for param in selected_params:
            value = getattr(selected, param)
            if param in allowed_params and value is None:
                raise ConfigError(f"selected.method={selected.method} requires {param}")
            if param not in allowed_params and value is not None:
                raise ConfigError(f"selected.method={selected.method} takes no {param}")
        if self.scan.c_min != 0:
            raise ConfigError(
                "scan.c_min must be 0 (selector gains are defined against the "
                f"c=0 leave-one-out baseline): got {self.scan.c_min}"
            )
        if self.scan.c_max < 1:
            raise ConfigError(f"scan.c_max must be >= 1: {self.scan.c_max}")
        for label, spec in (
            ("activation", self.samples.activation),
            ("scan", self.samples.scan),
            ("heldout_activation", self.samples.heldout_activation),
            ("final_eval", self.samples.final_eval),
            ("aie_corrupted", self.samples.aie_corrupted),
        ):
            if spec.task_set not in TASK_SETS:
                raise ConfigError(
                    f"samples.{label}.task_set must be one of {TASK_SETS}: "
                    f"{spec.task_set}"
                )
            if spec.examples_per_task < 1:
                raise ConfigError(f"samples.{label}.examples_per_task must be >= 1")
        # -- AIE baseline block (registry-validated at load) --
        if self.aie is not None:
            self._validate_aie()
        if self.scan.head_limit is not None and self.scan.head_limit < 1:
            raise ConfigError("scan.head_limit must be >= 1 (or null for all)")
        if self.protocol == "corrected_holdout":
            if self.task.task_split is None:
                raise ConfigError(
                    "protocol=corrected_holdout requires task.task_split "
                    "(a committed split file; never seed-generated)"
                )
            if self.samples.heldout_activation.seed == self.samples.final_eval.seed:
                raise ConfigError(
                    "corrected_holdout requires samples.heldout_activation.seed "
                    "!= samples.final_eval.seed (held-out z-estimation prompts "
                    "must differ from the final-evaluation prompts)"
                )
            for label, spec in (
                ("heldout_activation", self.samples.heldout_activation),
                ("final_eval", self.samples.final_eval),
            ):
                if spec.task_set != "eval":
                    raise ConfigError(
                        f"corrected_holdout requires samples.{label}.task_set=eval"
                    )
            for label, spec in (
                ("activation", self.samples.activation),
                ("scan", self.samples.scan),
                ("aie_corrupted", self.samples.aie_corrupted),
            ):
                if spec.task_set != "train":
                    raise ConfigError(
                        f"corrected_holdout requires samples.{label}.task_set=train "
                        "(held-out tasks are inaccessible before final evaluation)"
                    )

    def _validate_aie(self) -> None:
        """AIE block validation (methods against the subspaces.step1.aie registry,
        bounds, and the aie/samples.aie_corrupted consistency guard)."""
        from subspaces.step1.aie import METHOD_VERSIONS

        aie = self.aie
        assert aie is not None
        if self.protocol != "corrected_holdout":
            raise ConfigError(
                "aie requires protocol=corrected_holdout (train-task scores, "
                f"held-out evaluation): got {self.protocol!r}"
            )
        if not aie.methods:
            raise ConfigError("aie.methods must be a non-empty list")
        unknown_methods = [m for m in aie.methods if m not in METHOD_VERSIONS]
        if unknown_methods:
            raise ConfigError(
                f"aie.methods: unknown method(s) {unknown_methods}; "
                f"registered: {sorted(METHOD_VERSIONS)}"
            )
        if len(set(aie.methods)) != len(aie.methods):
            raise ConfigError(f"aie.methods contains duplicates: {aie.methods}")
        if not aie.k_values:
            raise ConfigError("aie.k_values must be a non-empty list")
        if len(set(aie.k_values)) != len(aie.k_values):
            raise ConfigError(f"aie.k_values contains duplicates: {aie.k_values}")
        for k in aie.k_values:
            if k < 1:
                raise ConfigError(f"aie.k_values entries must be >= 1: {k}")
        for name in (
            "corrupted_examples_per_task",
            "corruption_tries",
            "batch_size",
            "proxy_batch_size",
        ):
            if getattr(aie, name) < 1:
                raise ConfigError(f"aie.{name} must be >= 1")
        if aie.head_limit is not None and aie.head_limit < 1:
            raise ConfigError("aie.head_limit must be >= 1 (or null for the full grid)")
        spec = self.samples.aie_corrupted
        if (
            spec.examples_per_task != aie.corrupted_examples_per_task
            or spec.seed != aie.corrupted_seed
        ):
            raise ConfigError(
                "samples.aie_corrupted contradicts the aie block: spec is "
                f"({spec.examples_per_task}/task, seed {spec.seed}) but "
                f"aie.corrupted_examples_per_task="
                f"{aie.corrupted_examples_per_task}, "
                f"aie.corrupted_seed={aie.corrupted_seed}; make them agree "
                "(explicit contradiction refuses)"
            )


# -- generic dataclass <-> dict machinery -------------------------------------


def _build(cls: type, data: Any, where: str) -> Any:
    """Construct dataclass ``cls`` from ``data`` with unknown-key/type checks."""
    if not dataclasses.is_dataclass(cls):
        raise ConfigError(f"{where}: internal error, {cls} is not a dataclass")
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {sorted(unknown)}; known: {sorted(fields)}"
        )
    kwargs = {}
    for name, value in data.items():
        kwargs[name] = _coerce(hints[name], value, f"{where}.{name}")
    return cls(**kwargs)


def _coerce(hint: Any, value: Any, where: str) -> Any:
    origin = get_origin(hint)
    if origin in (Union, UnionType):
        args = [a for a in get_args(hint) if a is not type(None)]
        if value is None:
            if type(None) in get_args(hint):
                return None
            raise ConfigError(f"{where}: null is not allowed")
        return _coerce(args[0], value, where)
    if dataclasses.is_dataclass(hint):
        return _build(hint, value, where)
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping")
        _, val_t = get_args(hint)
        return {str(k): _coerce(val_t, v, f"{where}[{k}]") for k, v in value.items()}
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list")
        (item_t,) = get_args(hint)
        return [_coerce(item_t, v, f"{where}[{i}]") for i, v in enumerate(value)]
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if hint in (int, float, str, bool):
        if not isinstance(value, hint) or (hint is int and isinstance(value, bool)):
            raise ConfigError(
                f"{where}: expected {hint.__name__}, got {type(value).__name__} "
                f"({value!r})"
            )
        return value
    raise ConfigError(f"{where}: unsupported config type {hint!r}")


def to_dict(cfg: Any) -> dict:
    return dataclasses.asdict(cfg)


def load_step1_config(path: str | Path) -> Step1Config:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    declared = raw.pop("schema_version", SCHEMA_VERSION)
    if declared != SCHEMA_VERSION:
        raise ConfigError(
            f"{path}: schema_version {declared} not supported (expected "
            f"{SCHEMA_VERSION})"
        )
    if "significant" in raw:
        raise ConfigError(
            f"{path}: the 'significant:' section was renamed 'selected:' "
            "(camera-ready head-set terminology, 2026-09: selected = the "
            "sparse-optimization set, significant = the paired-test set); "
            "rename the key"
        )
    cfg = _build(Step1Config, raw, where="step1")
    cfg.validate()
    return cfg


def requested_semantics(cfg: Step1Config) -> dict:
    """Normalized requested semantics for the run identity.

    Locators (``matrix.reuse_path``) and presentation fields (``run_name``,
    ``compute``, ``model.device``/``dtype`` execution details stay recorded in
    the config files) are excluded here; content identity for the matrix comes
    from the resolved checkpoint SHA instead. The run ID itself is computed in
    ``subspaces.step1.resolve`` from these semantics plus resolved content IDs.
    """
    resolved = to_dict(cfg)
    resolved.pop("run_name", None)
    resolved.pop("compute", None)
    # The main selector is NOT part of the run identity: selection is pure CPU
    # over the persisted scan, and selector variants live side-by-side inside
    # ONE run (hash-named main_heads/headset_eval/heads artifacts). Changing
    # the selector must reuse the matrix, selected heads, caches, and scan.
    resolved.pop("main_selector", None)
    # The AIE baseline (subspaces/step1/aie.py) is likewise a parallel chain: its
    # methods/k/corruption knobs ride in the hash-named aie artifacts, so
    # variants share one journal run, and legacy configs' run IDs stay
    # byte-stable across the feature addition (both keys removed here).
    resolved.pop("aie", None)
    resolved.get("samples", {}).pop("aie_corrupted", None)
    resolved.get("matrix", {}).pop("reuse_path", None)
    # node_name is presentation (a directory-name component of the lineage
    # tree), like run_name; matrix content identity = the checkpoint sha.
    resolved.get("matrix", {}).pop("node_name", None)
    # task_split is a locator too: the split's CONTENT identity is carried by
    # identity.split_sha256, so relocating the file never forks a run.
    resolved.get("task", {}).pop("task_split", None)
    # device is an execution detail; dtype stays (it affects numerics).
    resolved.get("model", {}).pop("device", None)
    return resolved


def dump_resolved(cfg: Step1Config, path: str | Path) -> None:
    resolved = {"schema_version": SCHEMA_VERSION, **to_dict(cfg)}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        yaml.safe_dump(resolved, fh, sort_keys=False)
