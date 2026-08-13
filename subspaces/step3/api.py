"""Step 3 — information a head extracts from earlier tokens.

Implemented analysis (v1): the per-source-token FV-share decomposition by
token group (:mod:`subspaces.step3.token_groups`). For each requested head — by
default the MAIN heads of a Step-1 ``heads`` artifact (e.g. the largest_gap
selection) — the head's final-token output is decomposed over source tokens as
``alpha_t * O_h V_h z_t``, projected onto the head's per-task FV component,
normalized to signed shares, and aggregated per token group (bos, per-demo
input/arrow/output/sep, query input/arrow) over all prompts of all tasks.

Input contract
--------------
- heads: an explicit ``L:H`` list (requires ``--out-dir``) OR a Step-1
  ``heads`` artifact (``--heads-artifact ... --head-set main``; output lands
  next to the artifact by default);
- context (required for execution): model identity, task protocol, a sample
  manifest under ``samples.<name>`` (default name ``analysis``) providing the
  prompts, and ``activations`` pointing at the z-cache directory whose
  per-task means supply the head directions.

Output contract
---------------
kind ``step3_token_groups``: ``step3-tokengroups-<h10>.json`` (manifest with
per-head per-group statistics) + ``...-outcomes.npz`` (per-prompt group
shares) + ``...-plots/`` (bar charts; presentation-only). ``--dry-run``
validates inputs and prints the plan without writing anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from subspaces.context import AnalysisContext, ContextError
from subspaces.head_sets import Head, format_heads
from subspaces.paths import ProjectPaths

STEP3_SCHEMA_VERSION = 1

PLOT_INTERVALS = ("var", "std", "minmax")


@dataclass
class Step3Config:
    n_prompts_per_task: int = 40
    seed: int = 0
    n_boot: int = 2000
    samples_name: str = "analysis"
    interval: str = "var"  # plot presentation only; not part of identity

    def validate(self) -> None:
        if self.n_prompts_per_task < 1:
            raise ValueError(
                f"n_prompts_per_task must be >= 1: {self.n_prompts_per_task}"
            )
        if self.n_boot < 1:
            raise ValueError(f"n_boot must be >= 1: {self.n_boot}")
        if not self.samples_name:
            raise ValueError("samples_name must be non-empty")
        if self.interval not in PLOT_INTERVALS:
            raise ValueError(
                f"interval must be one of {PLOT_INTERVALS}: {self.interval!r}"
            )


def describe(
    cfg: Step3Config,
    heads: list[Head],
    paths: ProjectPaths,
    context: AnalysisContext | None = None,
) -> str:
    """Human-readable dry-run plan; performs validation, writes nothing."""
    cfg.validate()
    if context is not None:
        context.validate()
    lines = [
        "step3 token-group FV-share decomposition (dry run — writes nothing)",
        f"  heads:              {format_heads(heads)}",
        f"  n_prompts_per_task: {cfg.n_prompts_per_task}",
        f"  samples_name:       {cfg.samples_name}",
        f"  n_boot:             {cfg.n_boot}",
        f"  seed:               {cfg.seed}",
        f"  plot interval:      {cfg.interval}",
        f"  root:               {paths.root}",
    ]
    if context is not None:
        lines.append("  context:")
        lines.append(context.describe())
    else:
        lines.append("  context: none (pass --context for a reproducible record)")
    return "\n".join(lines)


def run(
    cfg: Step3Config,
    heads: list[Head],
    paths: ProjectPaths,
    context: AnalysisContext | None = None,
    *,
    heads_artifact: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    """Execute the token-group decomposition; returns the artifact manifest.

    Reuses an existing identical artifact (and re-renders plots for the
    requested interval); refuses to overwrite a different one.
    """
    cfg.validate()
    if not heads:
        raise ValueError(
            "the resolved head list is empty (e.g. an empty head set in the "
            "heads artifact); nothing to analyze."
        )
    if context is None:
        raise ContextError(
            "step3 execution requires --context (model, task split, prompt "
            "protocol, sample manifests, activation refs); only --dry-run is "
            "permissive without one."
        )
    context.validate_refs(paths)
    if cfg.samples_name not in context.samples:
        raise ContextError(
            f"step3 token-group analysis requires context.samples."
            f"{cfg.samples_name} (the analysis prompt manifest); found "
            f"{sorted(context.samples) or 'none'}."
        )
    if not context.activations:
        raise ContextError(
            "step3 token-group analysis requires context.activations (the "
            "z-cache directory whose per-task means give the head directions)."
        )
    if out_dir is not None:
        resolved_out = paths.resolve(out_dir)
    elif heads_artifact is not None:
        resolved_out = paths.resolve(heads_artifact).parent
        if "step1" in resolved_out.parts:
            raise ValueError(
                f"{resolved_out} lives in a frozen pre-tree run; step3 must "
                "not write into it by default — pass --out-dir (same rule as "
                "select-main/compose)."
            )
    else:
        raise ValueError(
            "--out-dir is required when heads are given explicitly "
            "(no heads artifact anchors the output location)"
        )

    from subspaces.step3.token_groups import run_token_groups  # heavy at call time

    return run_token_groups(
        cfg,
        heads,
        paths,
        context,
        heads_artifact=heads_artifact,
        out_dir=resolved_out,
    )
