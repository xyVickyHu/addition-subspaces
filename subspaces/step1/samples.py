"""Substep 3 — deterministic sample manifests (five kinds: activation / scan /
heldout_activation / final_eval / aie_corrupted).

Contract (kind ``samples``, one manifest per sample type, under
``step1/samples/``):

- generated with a LOCAL deterministic RNG seeded from the manifest's own
  ``seed`` — never from global random state;
- stable identities: ``task_id`` (task name), ``query_id`` (index of the query
  example in the task file's canonical JSON order), ``demo_ids`` (indices of the
  demonstration examples);
- stores semantic examples, rendered prompts, expected answers, prompt-format
  metadata, and task ordering;
- an existing sample artifact is NEVER silently regenerated;
- consumers must evaluate every listed example (no ``drop_last`` discards) and
  report denominators equal to the listed counts.

Two sampling semantics, kept strictly apart:

- **corrected** (``corrected_sample_plan``): stable IDs; at
  ``examples_per_task == n_examples`` (typically 100) every unique query is
  used exactly once, with demonstrations drawn deterministically without
  replacement from the remaining examples;
- **legacy_repro**: materialized by RUNNING the legacy draw code
  (``LimitedTaskDataset`` + the historical seeding) on identical inputs, so
  port-equivalence compares like with like. Implemented in M2 alongside the
  legacy instrumentation; never mixed with corrected sampling.

Held-out isolation: task lists are resolved through ``resolve_sample_tasks``,
which raises ``HoldoutViolation`` if a selection-side manifest (activation /
scan) would touch held-out tasks under the corrected protocol.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from subspaces.config import SampleSpec, Step1Config
from subspaces.paths import ProjectPaths
from subspaces.step1.split import TaskSplit

SAMPLES_SCHEMA_VERSION = 1
IMPL = {"module": "subspaces.step1.samples", "algorithm_version": 2}
# Five distinct roles: train-task activation (z + train-only mean_z),
# train-task scan, held-out activation (created only after selection, by
# evaluate-headset), held-out final evaluation, and train-task AIE-corrupted
# prompts (subspaces.step1.aie: n-shot prompts whose demo LABELS are permuted
# within-prompt; query + expected target stay intact).
SAMPLE_KINDS = (
    "activation",
    "scan",
    "heldout_activation",
    "final_eval",
    "aie_corrupted",
)
SELECTION_KINDS = ("activation", "scan", "aie_corrupted")
HELDOUT_KINDS = ("heldout_activation", "final_eval")
# Corruption rule identity for the aie_corrupted kind. The rule name/version
# joins the manifest identity together with the spec seed and the draw budget
# (``tries``), so a rule change forks the corrupted manifests without
# touching any other kind (the shared IMPL engine version stays put).
CORRUPTION_RULE = {"name": "demo_label_permutation", "version": 1}


class HoldoutViolation(RuntimeError):
    """A selection-side computation attempted to touch held-out tasks."""


def resolve_sample_tasks(
    spec: SampleSpec,
    kind: str,
    split: TaskSplit | None,
    protocol: str,
    all_tasks: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Resolve the task list for one sample manifest, enforcing isolation."""
    if kind not in SAMPLE_KINDS:
        raise ValueError(f"kind must be one of {SAMPLE_KINDS}: {kind!r}")
    if protocol == "corrected_holdout":
        if split is None:
            raise HoldoutViolation("corrected_holdout requires a committed task split")
        if kind in SELECTION_KINDS:
            if spec.task_set != "train":
                raise HoldoutViolation(
                    f"selection-side samples ({kind}) must use task_set=train "
                    f"under corrected_holdout, got {spec.task_set!r}"
                )
            return split.task_set("train")
        if spec.task_set != "eval":
            raise HoldoutViolation(
                f"held-out samples ({kind}) must use task_set=eval under "
                f"corrected_holdout, got {spec.task_set!r}"
            )
        return split.task_set("eval")
    if split is not None:
        return split.task_set(spec.task_set)
    if all_tasks is None:
        raise ValueError("no task split and no explicit task list provided")
    return tuple(all_tasks)


def corrected_sample_plan(
    n_examples: int,
    examples_per_task: int,
    n_shot: int,
    seed: int,
) -> list[dict]:
    """Deterministic per-task plan: [{"query_id", "demo_ids"}, ...].

    Uses a LOCAL ``random.Random(seed)``. When ``examples_per_task ==
    n_examples`` every unique query index is used exactly once (ascending
    order); for smaller counts a deterministic subset of queries is drawn
    without replacement. Demonstrations are drawn without replacement from the
    remaining examples and never include the query itself.
    """
    if n_examples < n_shot + 1:
        raise ValueError(
            f"need at least n_shot+1 examples (n_shot={n_shot}), " f"got {n_examples}"
        )
    if examples_per_task > n_examples:
        raise ValueError(
            f"examples_per_task={examples_per_task} exceeds the "
            f"{n_examples} unique queries available"
        )
    rng = random.Random(seed)
    if examples_per_task == n_examples:
        query_ids = list(range(n_examples))
    else:
        query_ids = sorted(rng.sample(range(n_examples), examples_per_task))
    plan = []
    for query_id in query_ids:
        pool = [i for i in range(n_examples) if i != query_id]
        demo_ids = rng.sample(pool, n_shot)
        plan.append({"query_id": query_id, "demo_ids": demo_ids})
    return plan


def permute_demo_labels(
    labels: list, rng: random.Random, tries: int
) -> tuple[list, int]:
    """``demo_label_permutation`` v1: permute a prompt's demo-label multiset.

    Draw up to ``tries`` permutations of ``labels`` from the LOCAL ``rng``
    and keep the FIRST draw with ZERO fixed points (permuted label != the
    original label at every position, compared by VALUE — a swap between two
    demos sharing one label value is a fixed point). When no zero-fixed-point
    draw exists within the budget (e.g. few distinct labels), the first draw
    attaining the minimal fixed-point count is kept. Returns
    ``(permuted_labels, n_fixed_points)``; the count is recorded per sample.
    """
    if tries < 1:
        raise ValueError(f"corruption tries must be >= 1: {tries}")
    if not labels:
        raise ValueError("cannot permute an empty demo-label list")
    best: list | None = None
    best_fixed: int | None = None
    for _ in range(tries):
        draw = rng.sample(labels, len(labels))
        fixed = sum(
            1
            for original, permuted in zip(labels, draw, strict=True)
            if permuted == original
        )
        if best_fixed is None or fixed < best_fixed:
            best, best_fixed = draw, fixed
        if best_fixed == 0:
            break
    assert best is not None and best_fixed is not None  # tries >= 1
    return best, best_fixed


def _load_prompt_format_registry():
    """Load ``subspaces/utils/prompt_formats.py`` standalone (no heavy subspaces.utils init)."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "utils" / "prompt_formats.py"
    spec = importlib.util.spec_from_file_location("_fv_prompt_formats", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _task_seed(seed: int, kind: str, task_id: str) -> int:
    """Stable per-(kind, task) RNG seed (never Python's randomized hash).

    The sample KIND is part of the derivation so the activation-estimation
    prompts and the scan/evaluation prompts are decoupled draws — z is never
    estimated on the exact prompts being scored.
    """
    import hashlib

    digest = hashlib.sha256(f"{seed}:{kind}:{task_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def render_prompt(
    examples: list[dict], demo_ids: list[int], query_id: int, fmt: dict
) -> tuple[str, str, str]:
    """Assemble the n-shot prompt AND the zero-shot prompt exactly like the
    legacy renderer (``format_input``; zero-shot = ``format_input([], x_q)``).

    The legacy evaluation paradigm: clean accuracy on the N-SHOT prompt;
    every FV intervention on the ZERO-SHOT prompt of the same example.
    """
    registry = _load_prompt_format_registry()
    prompt = "".join(
        registry.render_demo(examples[i]["input"], examples[i]["output"], fmt)
        for i in demo_ids
    )
    zero_shot = registry.render_query(examples[query_id]["input"], fmt)
    return prompt + zero_shot, zero_shot, str(examples[query_id]["output"])


def _corrupted_sample(
    examples: list[dict],
    entry: dict,
    fmt: dict,
    registry,
    rng: random.Random,
    tries: int,
) -> dict:
    """One aie_corrupted sample: same draw as an activation-kind sample (the
    per-(kind,task) seed already decouples the streams), demo labels permuted
    per ``demo_label_permutation`` v1, query + expected target intact."""
    demos = [examples[i] for i in entry["demo_ids"]]
    original_labels = [demo["output"] for demo in demos]
    permuted_labels, n_fixed = permute_demo_labels(original_labels, rng, tries)
    corrupted_demos = [
        {"input": demo["input"], "output": label}
        for demo, label in zip(demos, permuted_labels, strict=True)
    ]
    query = examples[entry["query_id"]]
    prompt = "".join(
        registry.render_demo(demo["input"], demo["output"], fmt)
        for demo in corrupted_demos
    )
    zero_shot = registry.render_query(query["input"], fmt)
    return {
        "query_id": entry["query_id"],
        "demo_ids": entry["demo_ids"],
        "query": query,
        "demos": corrupted_demos,
        "original_demo_outputs": original_labels,
        "n_fixed_points": n_fixed,
        "prompt": prompt + zero_shot,
        "zero_shot_prompt": zero_shot,
        "expected": str(query["output"]),
    }


def _legacy_sorted_task_index(task_id: str, tasks: tuple[str, ...]) -> int:
    """Legacy seeding indexed tasks in SORTED name order (recovery_scan's
    ``sorted(tasks.keys())``), independent of split-file order."""
    return sorted(tasks).index(task_id)


def _legacy_task_plan(
    examples: list[dict],
    spec: SampleSpec,
    kind: str,
    n_shot: int,
    task_index: int,
) -> list[dict]:
    """Legacy-lane plan: RUN the legacy draw ALGORITHMS under the one seeding
    convention the historical pipeline ever used (``random.seed(seed + ti)``
    with ti over sorted task names, as in ``recovery_scan``), then map the
    drawn prompts back to stable IDs.

    Role-faithful algorithms: activation-estimation draws mirror the legacy z
    sampler (query chosen WITH replacement + demos sampled per draw,
    ``compute_z_result_per_task``); evaluation draws run ``LimitedTaskDataset``.
    This reproduces the legacy SAMPLING ALGORITHMS deterministically for
    Lane-A comparisons — it does NOT claim to recover the original unseeded
    historical draws, which are unrecoverable by construction.
    """
    import random as global_random

    # Import the legacy draw code BEFORE seeding: the first subspaces.utils package
    # import consumes global RNG state, which would corrupt the first task's
    # draws in a fresh process (audit-verified regression).
    from subspaces.utils.data import LimitedTaskDataset  # noqa: F401  (heavy)

    index_of = {example["input"]: i for i, example in enumerate(examples)}
    global_random.seed(spec.seed + task_index)
    plan = []
    if kind in ("activation", "heldout_activation"):
        task_input = [example["input"] for example in examples]
        for _ in range(spec.examples_per_task):
            x_q = global_random.choice(task_input)
            remaining_input = [item for item in task_input if item != x_q]
            x_icl = global_random.sample(remaining_input, n_shot)
            plan.append(
                {
                    "query_id": index_of[x_q],
                    "demo_ids": [index_of[x] for x in x_icl],
                }
            )
        return plan

    dataset = LimitedTaskDataset(
        {"task": examples}, n_shot, limit=spec.examples_per_task
    )
    for i in range(len(dataset)):
        datapoint = dataset[i]  # {"task_name", "x_q", "x_icl"}
        plan.append(
            {
                "query_id": index_of[datapoint["x_q"]],
                "demo_ids": [index_of[x] for x in datapoint["x_icl"]],
            }
        )
    return plan


def samples_expected(
    cfg: Step1Config,
    spec: SampleSpec,
    kind: str,
    split: TaskSplit | None,
    paths: ProjectPaths,
) -> dict:
    """The sample manifest's identity envelope — computable BEFORE generation
    (dataset content fingerprint + split ref + spec/task config + impl), so
    the content-addressed cache location derives from it. The task-resolution
    guard runs here too: held-out isolation is enforced on every code path
    that even NAMES a sample manifest.
    """
    from subspaces.artifacts import dataset_fingerprint, file_ref

    task_dir = paths.task_dir(cfg.task.dataset_dir)
    all_tasks = tuple(sorted(p.stem for p in task_dir.glob("*.json")))
    resolve_sample_tasks(spec, kind, split, cfg.protocol, all_tasks=all_tasks)
    mode = "corrected" if cfg.protocol == "corrected_holdout" else "legacy"
    inputs: dict = {"dataset": {"content_fingerprint": dataset_fingerprint(task_dir)}}
    if split is not None and cfg.task.task_split:
        inputs["split"] = file_ref(cfg.task.task_split, paths)
    config = {
        "sample_kind": kind,
        "mode": mode,
        "spec": {
            "examples_per_task": spec.examples_per_task,
            "seed": spec.seed,
            "task_set": spec.task_set,
        },
        "task": {
            "dataset_dir": cfg.task.dataset_dir,
            "prompt_format": cfg.task.prompt_format,
            "n_shot": cfg.task.n_shot,
        },
    }
    if kind == "aie_corrupted":
        if cfg.aie is None:
            raise ValueError(
                "aie_corrupted samples require the aie config block "
                "(subspaces.config.AIEConfig); refusing to derive an identity"
            )
        if mode != "corrected":
            raise ValueError(
                "aie_corrupted samples are defined for the corrected protocol "
                f"only, got mode {mode!r}"
            )
        # the corruption rule + draw budget join the manifest identity (the
        # permutation seed is the spec seed, already in config.spec above)
        config["corruption"] = {**CORRUPTION_RULE, "tries": cfg.aie.corruption_tries}
    return {
        "kind": "samples",
        "schema_version": SAMPLES_SCHEMA_VERSION,
        "inputs": inputs,
        "config": config,
        "impl": IMPL,
    }


def generate_samples(
    cfg: Step1Config,
    spec: SampleSpec,
    kind: str,
    split: TaskSplit | None,
    paths: ProjectPaths,
    out_dir: Path,
) -> dict:
    """Materialize one sample manifest (never silently regenerated).

    The task-resolution guard runs FIRST (inside ``samples_expected``) so
    held-out isolation is enforced on every code path. Corrected protocol
    uses the deterministic local-RNG planner (all unique queries once at
    100/task); legacy_reproduction runs the legacy draw code under the
    documented seeding convention.
    """
    from subspaces.artifacts import make_manifest, reuse_or_refuse, write_json_atomic

    task_dir = paths.task_dir(cfg.task.dataset_dir)
    all_tasks = tuple(sorted(p.stem for p in task_dir.glob("*.json")))
    tasks = resolve_sample_tasks(spec, kind, split, cfg.protocol, all_tasks=all_tasks)
    registry = _load_prompt_format_registry()
    fmt = registry.get_format(cfg.task.prompt_format)

    expected = samples_expected(cfg, spec, kind, split, paths)
    mode = expected["config"]["mode"]
    inputs = expected["inputs"]
    config = expected["config"]
    out_path = Path(out_dir) / f"{kind}.json"
    existing = reuse_or_refuse(
        out_path,
        expected,
        expect_kind="samples",
        max_schema_version=SAMPLES_SCHEMA_VERSION,
    )
    if existing is not None:
        return existing

    per_task: dict[str, dict] = {}
    for task_id in tasks:
        examples = json.loads(
            (task_dir / f"{task_id}.json").read_text(encoding="utf-8")
        )
        if mode == "corrected":
            plan = corrected_sample_plan(
                len(examples),
                spec.examples_per_task,
                cfg.task.n_shot,
                _task_seed(spec.seed, kind, task_id),
            )
        else:
            plan = _legacy_task_plan(
                examples,
                spec,
                kind,
                cfg.task.n_shot,
                _legacy_sorted_task_index(task_id, tasks),
            )
        samples = []
        if kind == "aie_corrupted":
            # one label-permutation stream per task, distinct by derivation
            # from the plan stream (different label in the seed hash)
            corruption_rng = random.Random(
                _task_seed(spec.seed, f"{kind}:permutation", task_id)
            )
            tries = config["corruption"]["tries"]
            for entry in plan:
                samples.append(
                    _corrupted_sample(
                        examples, entry, fmt, registry, corruption_rng, tries
                    )
                )
        else:
            for entry in plan:
                prompt, zero_shot_prompt, expected_answer = render_prompt(
                    examples, entry["demo_ids"], entry["query_id"], fmt
                )
                samples.append(
                    {
                        "query_id": entry["query_id"],
                        "demo_ids": entry["demo_ids"],
                        "query": examples[entry["query_id"]],
                        "demos": [examples[i] for i in entry["demo_ids"]],
                        "prompt": prompt,
                        "zero_shot_prompt": zero_shot_prompt,
                        "expected": expected_answer,
                    }
                )
        per_task[task_id] = {"n_examples": len(examples), "samples": samples}

    manifest = make_manifest(
        kind="samples",
        schema_version=SAMPLES_SCHEMA_VERSION,
        paths=paths,
        config=config,
        inputs=inputs,
        payload={
            "impl": IMPL,
            "prompt_format": {"name": cfg.task.prompt_format, **fmt},
            "task_order": list(tasks),
            "tasks": per_task,
            "counts": {
                task_id: len(data["samples"]) for task_id, data in per_task.items()
            },
        },
    )
    write_json_atomic(out_path, manifest)
    return manifest
