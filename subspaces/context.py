"""Shared analysis-context contract for the Step-2/3 scaffolds.

Steps 2 and 3 need more than a head list to produce interpretable results:
the model identity, the task split, the prompt protocol, the sample manifests,
and the activation/cache references all determine what an analysis means. This
minimal contract carries those references explicitly (as locators + content
hashes where applicable) so a standalone Step-2/3 invocation is reproducible
without parsing any run manifest.

A context file is YAML:

.. code-block:: yaml

    schema_version: 1
    model: {name: meta-llama/Meta-Llama-3-8B-Instruct, revision: 8afb486c...}
    task:
      family: number_add
      prompt_format: arrow
      n_shot: 5
      task_split: configs/task_splits/number_add_paper.yaml
    samples:                       # optional: sample-manifest references
      final_eval: {path: log/cache/samples/<fp16>/final_eval.json}
    activations:                   # optional: z-cache reference(s) — either a
      path: log/cache/z/<fingerprint>          # single cache, or a named map:
    # activations:
    #   train: {path: log/cache/z/<fp-train>}
    #   heldout: {path: log/cache/z/<fp-heldout>}
    heads_artifact: .../main-unified-v1-<h8>/heads.json  # optional; either
    # the composed heads.json or a selector's main_heads.json
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONTEXT_SCHEMA_VERSION = 1


class ContextError(ValueError):
    pass


@dataclass
class AnalysisContext:
    model: dict
    task: dict
    samples: dict = field(default_factory=dict)
    activations: dict = field(default_factory=dict)
    heads_artifact: str | dict | None = None

    def validate(self) -> None:
        """Schema validation (cheap, dry-run friendly)."""
        if not self.model.get("name"):
            raise ContextError("context.model.name is required")
        for key in ("family", "prompt_format", "n_shot"):
            if self.task.get(key) in (None, ""):
                raise ContextError(f"context.task.{key} is required")
        if not isinstance(self.task["n_shot"], int):
            raise ContextError("context.task.n_shot must be an integer")
        if not isinstance(self.samples, dict) or not all(
            isinstance(ref, dict) for ref in self.samples.values()
        ):
            raise ContextError(
                "context.samples must be a mapping of name -> {path: ...}"
            )
        if not isinstance(self.activations, dict) or (
            self.activations
            and "path" not in self.activations
            and not all(isinstance(ref, dict) for ref in self.activations.values())
        ):
            raise ContextError(
                "context.activations must be a mapping: a single "
                "{path: log/cache/z/<fp>} reference or a named map of such "
                "references (e.g. train/heldout)"
            )

    def heads_artifact_path(self) -> str | None:
        if self.heads_artifact is None:
            return None
        if isinstance(self.heads_artifact, str):
            return self.heads_artifact
        path = self.heads_artifact.get("path")
        if not path:
            raise ContextError("context.heads_artifact requires a path")
        return path

    def activation_refs(self) -> dict[str, dict]:
        """Normalized activation references: ``{label: {path[, sha256]}}``.

        ``activations`` is either a single reference mapping (has a ``path``
        key) or a named map of such references (e.g. ``train``/``heldout``).
        """
        if not self.activations:
            return {}
        if "path" in self.activations:
            return {"activations": self.activations}
        refs = {}
        for name in sorted(self.activations):
            refs[f"activations.{name}"] = self.activations[name]
        return refs

    def validate_refs(self, paths) -> None:
        """Content-addressed reference validation for EXECUTION.

        Every referenced artifact must exist; kind is checked for the heads
        artifact; sha256/semantic fingerprints are verified when provided.
        Dry-run stays permissive (``validate`` only); ``run`` calls this.
        """
        from subspaces.artifacts import semantic_fingerprint, sha256_file
        from subspaces.head_sets import read_heads_manifest

        self.validate()
        if self.heads_artifact is not None:
            heads_path = paths.resolve(self.heads_artifact_path())
            manifest, _kind = read_heads_manifest(heads_path)
            if isinstance(self.heads_artifact, dict):
                expected = self.heads_artifact.get("semantic_fingerprint")
                actual = semantic_fingerprint(manifest)
                if expected is not None and expected != actual:
                    raise ContextError(
                        f"context.heads_artifact fingerprint mismatch for "
                        f"{heads_path}: expected {expected[:12]}, actual "
                        f"{actual[:12]}"
                    )

        refs = {f"samples.{name}": ref for name, ref in self.samples.items()}
        refs.update(self.activation_refs())
        for label, ref in refs.items():
            if (
                not isinstance(ref, dict)
                or not ref.get("path")
                or not isinstance(ref["path"], str)
            ):
                raise ContextError(
                    f"context.{label}: expected a mapping with a string path"
                )
            target = paths.resolve(ref["path"])
            if not target.exists():
                raise ContextError(f"context.{label}: {target} does not exist")
            expected_sha = ref.get("sha256")
            if expected_sha is not None and (
                not target.is_file() or sha256_file(target) != expected_sha
            ):
                raise ContextError(f"context.{label}: content mismatch for {target}")

    def describe(self) -> str:
        split = self.task.get("task_split") or "none (all tasks)"
        samples = ", ".join(sorted(self.samples)) if self.samples else "none"
        activation_refs = self.activation_refs()
        activations = (
            ", ".join(str(ref.get("path")) for ref in activation_refs.values())
            if activation_refs
            else "none"
        )
        return (
            f"  model:       {self.model.get('name')} "
            f"(revision {self.model.get('revision') or 'unresolved'})\n"
            f"  task:        {self.task.get('family')} "
            f"[{self.task.get('prompt_format')}, n_shot={self.task.get('n_shot')}]\n"
            f"  task_split:  {split}\n"
            f"  samples:     {samples}\n"
            f"  activations: {activations}"
        )


def load_context(path: str | Path) -> AnalysisContext:
    try:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except OSError as err:
        raise ContextError(f"{path}: cannot read context file ({err})") from err
    except yaml.YAMLError as err:
        raise ContextError(f"{path}: invalid context YAML ({err})") from err
    if not isinstance(raw, dict):
        raise ContextError(f"{path}: expected a mapping")
    version = raw.pop("schema_version", CONTEXT_SCHEMA_VERSION)
    if version != CONTEXT_SCHEMA_VERSION:
        raise ContextError(
            f"{path}: schema_version {version} not supported "
            f"(expected {CONTEXT_SCHEMA_VERSION})"
        )
    known = {"model", "task", "samples", "activations", "heads_artifact"}
    unknown = set(raw) - known
    if unknown:
        raise ContextError(f"{path}: unknown key(s) {sorted(unknown)}")
    context = AnalysisContext(
        model=raw.get("model") or {},
        task=raw.get("task") or {},
        samples=raw.get("samples") or {},
        activations=raw.get("activations") or {},
        heads_artifact=raw.get("heads_artifact"),
    )
    context.validate()
    return context
