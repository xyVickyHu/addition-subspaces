"""Lineage-tree placement for Step-1 artifacts (output layout v2).

Every artifact lives under its PRIMARY lineage parent; method/protocol
variants are prefix-named sibling directories (user-specified layout,
project plan, Milestone LT):

    log/runs/<node>__<ckptid12>/           matrix node (tree root)
      matrix_ref.json                      the node's identity record
      matrix/checkpoints/...               mode=train output (moved from staging)
      derived/checkpoints/...              checkpoint_select outputs
      sig-<method>-v<V>[-<p6>]/            significant variants, siblings
        significant_heads.json
        scan-<E>x<C>-<h6>/                 scan protocol variants, siblings
          head_scan.json, scan_outcomes.npz
          main-<selector>-v<V>-<h8>/       main-selector variants, siblings
            main_heads.json, heads.json,
            eval-<h10>.json (+ -outcomes.npz)
          recpos-<selector>-v<V>-<h8>/     recovery-positive selections (the
                                           outcome-consuming paired family;
                                           adopted default: paired_bh q=0.05)
                                           — same artifact contract as main-*
    log/journal/<run_name>__<jhash12>/     per-invocation config records
    log/cache/samples/<fp16>/<kind>.json   content-addressed sample manifests

There is no separate node.json: each node's primary artifact carries the
node identity (inputs fingerprints + config slice + impl) with the shared
refuse-on-mismatch guard. Non-tree inputs (sample manifests, z caches) stay
content-addressed and are referenced by fingerprint; nesting them under a
matrix would be scientifically wrong (they do not depend on it).

The matrix-node hash key is the EFFECTIVE checkpoint identity: the pinned
file's sha256, or — for derived checkpoints, whose saved-file bytes are not
contractual across torch versions — the tensor ``content_digest``.
"""

from __future__ import annotations

import re
from pathlib import Path

from subspaces.artifacts import ArtifactError, semantic_fingerprint
from subspaces.config import SignificantConfig, Step1Config
from subspaces.paths import ProjectPaths
from subspaces.step1.significant import METHOD_PARAM_SCHEMAS

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_node_name(raw: str) -> str:
    cleaned = _SANITIZE_RE.sub("-", raw).strip("-._")
    return (cleaned or "matrix")[:48]


def node_name(cfg: Step1Config) -> str:
    """Matrix-node display name: explicit config wins, else derived.

    Derived default = the reuse_path basename; when that is the generic
    ``matrix`` dir of a trained run (legacy ``<run>/step1/matrix`` or a tree
    node's ``<node>__<key>/matrix``), the owning run/node name is used. Set
    ``matrix.node_name`` explicitly when the default reads poorly — it is
    presentation only (never part of any identity)."""
    if cfg.matrix.node_name:
        return cfg.matrix.node_name
    if cfg.matrix.mode == "reuse" and cfg.matrix.reuse_path:
        reuse = Path(cfg.matrix.reuse_path)
        base = reuse.name
        if base == "matrix":
            parent = reuse.parent
            if parent.name == "step1":  # legacy flat run layout
                base = parent.parent.name.split("__")[0] or base
            else:  # tree node: <node>__<key12>/matrix
                base = parent.name.split("__")[0] or base
        return sanitize_node_name(base)
    return sanitize_node_name(cfg.run_name)


def checkpoint_key(selected_checkpoint: dict) -> str:
    """12-hex effective-checkpoint key (content digest beats file sha).

    Derived checkpoints always carry ``content_digest``; their byte sha is
    recorded under ``file_sha256`` (volatile) — the ``.get`` chain keeps
    this KeyError-free for both spellings."""
    key = (
        selected_checkpoint.get("content_digest")
        or selected_checkpoint.get("sha256")
        or selected_checkpoint.get("file_sha256")
    )
    if not key:
        raise ArtifactError(
            "selected_checkpoint carries neither content_digest nor a file "
            f"sha: {sorted(selected_checkpoint)}"
        )
    return key[:12]


def matrix_node_dir(paths: ProjectPaths, cfg: Step1Config, matrix_ref: dict) -> Path:
    key = checkpoint_key(matrix_ref["selected_checkpoint"])
    return paths.runs_dir / f"{node_name(cfg)}__{key}"


def sig_dirname(sig_cfg: SignificantConfig) -> str:
    """Sig-node dirname: method-v<V> plus a param hash for parameterized
    methods. The param key set derives from the registry schema, so a future
    parameter FIELD on SignificantConfig can never leak into other methods'
    dirnames."""
    key = (sig_cfg.method, sig_cfg.method_version)
    schema = METHOD_PARAM_SCHEMAS.get(key)
    if schema is None:
        raise ArtifactError(f"unknown significant method {key!r}; not in registry")
    name = f"sig-{sig_cfg.method}-v{sig_cfg.method_version}"
    params = {
        param: getattr(sig_cfg, param)
        for param in sorted(schema)
        if getattr(sig_cfg, param) is not None
    }
    if params:
        name += f"-{semantic_fingerprint(params)[:6]}"
    return name


def sig_dirname_from_manifest(significant: dict) -> str:
    """Sig-node dirname for an EXTERNAL significant artifact (adoption path):
    reconstructed from the artifact's own recorded config. The recorded
    identity slice carries only the active method's registered parameters —
    absent parameters reconstruct as None (their dataclass default)."""
    recorded = (significant.get("config") or {}).get("significant") or {}
    method = recorded.get("method", "elbow")
    method_version = recorded.get("method_version", 1)
    schema = METHOD_PARAM_SCHEMAS.get((method, method_version))
    if schema is None:
        raise ArtifactError(
            f"unknown significant method ({method!r}, {method_version}) "
            "recorded in the supplied artifact; not in this code's registry"
        )
    sig_cfg = SignificantConfig(
        method=method,
        method_version=method_version,
        **{param: recorded.get(param) for param in schema},
    )
    return sig_dirname(sig_cfg)


def sig_node_dir(matrix_node: Path, sig_cfg: SignificantConfig) -> Path:
    return matrix_node / sig_dirname(sig_cfg)


def scan_node_dir(sig_node: Path, cfg: Step1Config, scan_expected: dict) -> Path:
    """Scan protocol variants are siblings under their significant node —
    the scan level is NOT a grouping folder: two scans of the same head set
    legitimately differ (examples/task, scoring, c-grid, batch), and a main
    selection derives from ONE scan's curves."""
    tag = f"{cfg.samples.scan.examples_per_task}x{cfg.scan.c_max}"
    return sig_node / f"scan-{tag}-{semantic_fingerprint(scan_expected)[:6]}"


def main_dirname(
    selector_name: str,
    selector_version: int,
    params: dict,
    scan_manifest: dict,
) -> str:
    """Selection-node dirname. The dir prefix is the selector's ROLE
    (adopted 2026-08-04): outcome-consuming selectors (the paired McNemar
    family) select "recovery-positive" heads and get ``recpos-``; curve-level
    selectors keep ``main-``. The prefix is presentation only — the identity
    hash is prefix-independent, so the role split forks no artifact identity
    (pre-adoption paired nodes were renamed on disk, not recomputed)."""
    from subspaces.step1.selectors import OUTCOME_SELECTORS

    identity = semantic_fingerprint(
        {
            "params": params,
            "scan": semantic_fingerprint(scan_manifest),
            "selector": [selector_name, selector_version],
        }
    )[:8]
    role = (
        "recpos" if (selector_name, selector_version) in OUTCOME_SELECTORS else "main"
    )
    return f"{role}-{selector_name}-v{selector_version}-{identity}"


def main_node_dir(
    scan_node: Path,
    selector_name: str,
    selector_version: int,
    params: dict,
    scan_manifest: dict,
) -> Path:
    return scan_node / main_dirname(
        selector_name, selector_version, params, scan_manifest
    )
