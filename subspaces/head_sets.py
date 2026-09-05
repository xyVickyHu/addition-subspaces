"""Head-set parsing and loading for Steps 2/3 (and Step-1 outputs).

Two ways to specify heads:

- an explicit ``L:H`` list, e.g. ``"15:2,15:1,13:6"``;
- a Step-1 head artifact: either the composed ``heads`` artifact (exposing
  ``selected_heads``, ``main_heads``, and ``minor_heads``) or a selector's
  ``main_heads`` artifact directly (exposing ``main_heads``/``minor_heads``
  only) — selector nodes backfilled over saved scans carry no composed
  ``heads.json``, so Steps 2/3 accept both kinds.
"""

from __future__ import annotations

import json
from pathlib import Path

from subspaces.artifacts import ArtifactError, modernize

HEADS_SCHEMA_VERSION = 1
MAIN_HEADS_SCHEMA_VERSION = 1
HEAD_SET_NAMES = ("main", "selected", "minor")
# head-set availability per accepted artifact kind
_KIND_SETS = {
    "heads": ("main", "selected", "minor"),
    "main_heads": ("main", "minor"),
}
_KIND_MAX_SCHEMA = {
    "heads": HEADS_SCHEMA_VERSION,
    "main_heads": MAIN_HEADS_SCHEMA_VERSION,
}

Head = tuple[int, int]


class HeadSpecError(ValueError):
    pass


def parse_heads(spec: str) -> list[Head]:
    """Parse ``"15:2,15:1"`` into ``[(15, 2), (15, 1)]`` (order-preserving)."""
    if "(" in spec or ")" in spec:
        raise HeadSpecError(
            f"invalid head spec {spec!r}: use colon form 'L:H,L:H,...' "
            "(e.g. '15:2,15:1,13:6'), not '(L,H),...'"
        )
    heads: list[Head] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 2:
            raise HeadSpecError(
                f"invalid head token {token!r} in {spec!r}: expected 'L:H'"
            )
        try:
            layer_idx, head_idx = int(parts[0]), int(parts[1])
        except ValueError as err:
            raise HeadSpecError(
                f"invalid head token {token!r} in {spec!r}: indices must be integers"
            ) from err
        heads.append((layer_idx, head_idx))
    if not heads:
        raise HeadSpecError(f"empty head spec: {spec!r}")
    return heads


def format_heads(heads: list[Head]) -> str:
    return ",".join(f"{layer_idx}:{head_idx}" for layer_idx, head_idx in heads)


def validate_heads(
    heads: list[Head],
    *,
    n_layers: int | None = None,
    n_heads: int | None = None,
) -> list[Head]:
    seen: set[Head] = set()
    for layer_idx, head_idx in heads:
        if (layer_idx, head_idx) in seen:
            raise HeadSpecError(f"duplicate head {layer_idx}:{head_idx}")
        seen.add((layer_idx, head_idx))
        if layer_idx < 0 or head_idx < 0:
            raise HeadSpecError(f"negative head index {layer_idx}:{head_idx}")
        if n_layers is not None and layer_idx >= n_layers:
            raise HeadSpecError(
                f"layer {layer_idx} out of range (model has {n_layers} layers)"
            )
        if n_heads is not None and head_idx >= n_heads:
            raise HeadSpecError(
                f"head {head_idx} out of range (model has {n_heads} heads/layer)"
            )
    return heads


def resolve_heads_arg(
    *,
    heads: str | None,
    heads_artifact: str | Path | None,
    head_set: str = "main",
    paths=None,
) -> list[Head]:
    """Resolve the standard Step-2/3 head inputs (explicit list XOR artifact)."""
    if heads and heads_artifact:
        raise HeadSpecError("pass either --heads or --heads-artifact, not both")
    if heads:
        return validate_heads(parse_heads(heads))
    if heads_artifact:
        artifact_path = (
            paths.resolve(heads_artifact) if paths is not None else heads_artifact
        )
        return load_head_set(artifact_path, which=head_set)
    raise HeadSpecError(
        "specify heads via --heads 'L:H,...' or --heads-artifact <heads.json> "
        "(with --head-set main|selected|minor)"
    )


def read_heads_manifest(path: str | Path) -> tuple[dict, str]:
    """Read a Step-1 head artifact of either accepted kind.

    Returns ``(manifest, kind)`` for kind ``heads`` (composed) or
    ``main_heads`` (selector output); refuses anything else. Schema versions
    are bounded per kind (the ``read_manifest`` rule, generalized to a kind
    set).
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ArtifactError(f"artifact manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = modernize(json.load(fh))
    kind = manifest.get("kind")
    if kind not in _KIND_SETS:
        raise ArtifactError(
            f"{manifest_path}: expected kind in {tuple(_KIND_SETS)}, found {kind!r}"
        )
    version = manifest.get("schema_version")
    max_version = _KIND_MAX_SCHEMA[kind]
    if not isinstance(version, int) or version < 1 or version > max_version:
        raise ArtifactError(
            f"{manifest_path}: schema_version {version!r} not supported "
            f"(max {max_version})"
        )
    return manifest, kind


def load_head_set(path: str | Path, which: str = "main") -> list[Head]:
    """Load one head set from a Step-1 head artifact (``heads`` or
    ``main_heads`` kind)."""
    if which not in HEAD_SET_NAMES:
        raise HeadSpecError(f"which must be one of {HEAD_SET_NAMES}: {which!r}")
    manifest, kind = read_heads_manifest(path)
    if which not in _KIND_SETS[kind]:
        raise HeadSpecError(
            f"{path}: a {kind!r} artifact carries no {which!r} set "
            f"(available: {_KIND_SETS[kind]}); pass the composed heads.json "
            "for selected heads."
        )
    key = f"{which}_heads"
    if key not in manifest:
        raise HeadSpecError(f"{path}: head artifact has no {key!r} field")
    heads = [(int(layer_idx), int(head_idx)) for layer_idx, head_idx in manifest[key]]
    dims = manifest.get("model_dims") or {}
    return validate_heads(
        heads, n_layers=dims.get("n_layers"), n_heads=dims.get("n_heads")
    )
