"""Every manifest the current code writes must canonicalize onto the key paths
of its pre-rename twin.

The head-set terminology rename (2026-09-04) spells new manifests in the paper
vocabulary while ``artifacts.canonicalize`` hashes them over the legacy
spelling. A persisted key renamed WITHOUT a ``LEGACY_KEY_SPELLING`` entry would
silently fork identities (a regenerated scan gets a new fingerprint and new
selector-node names). This test runs the composed fake-repo pipeline, takes
every manifest it wrote, canonicalizes it, and checks that its normalized key
paths are a subset of the legacy inventory captured from real pre-rename
artifacts (``tests/fixtures/legacy_key_inventory.json``). A path outside the
inventory is either a renamed key (add it to the spelling table) or a genuinely
new field (add it to the inventory deliberately).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from subspaces.artifacts import VOLATILE_KEYS, canonicalize
from subspaces.config import load_step1_config
from subspaces.paths import ProjectPaths
from subspaces.step1 import pipeline

FIXTURES = Path(__file__).parent / "fixtures"
DYNAMIC_PARENTS = {
    "curves",
    "decisions",
    "per_task",
    "per_task_macro",
    "per_head",
    "per_head_k",
    "per_coef",
    "counts",
    "tasks",
    "selections",
    "task_slices",
    "by_task",
    "per_prompt",
    "label_share",
    "arrays",
    "chunks",
    "heads",
    "pca",
    "subspaces",
    "mod_vectors",
    "groups",
    "per_group",
}
ID_LIKE = re.compile(r"^(\d+(:\d+)*|\(\d+,\s*\d+\)|-?\d+(\.\d+)?|[0-9a-f]{6,})$")


def normalized_paths(obj, parent_key: str | None = None, prefix: str = "") -> set[str]:
    out: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in VOLATILE_KEYS:  # never part of any identity
                continue
            seg = str(key)
            if parent_key in DYNAMIC_PARENTS or ID_LIKE.match(seg):
                seg = "*"
            out |= normalized_paths(
                value, str(key), f"{prefix}.{seg}" if prefix else seg
            )
        if not obj:
            out.add(prefix + ".{}" if prefix else "{}")
    elif isinstance(obj, list):
        for item in obj[:5]:
            out |= normalized_paths(item, parent_key, prefix + "[]")
        if not obj:
            out.add(prefix + "[]")
    else:
        out.add(prefix)
    return out


def test_fresh_manifests_canonicalize_onto_legacy_key_paths(fake_repo, gpu_stubs):
    inventory = json.loads((FIXTURES / "legacy_key_inventory.json").read_text())[
        "kinds"
    ]
    paths = ProjectPaths.from_root(fake_repo)
    cfg = load_step1_config(fake_repo / "configs" / "step1_test.yaml")
    pipeline.run(cfg, paths)

    checked = 0
    problems: list[str] = []
    for manifest_path in sorted(paths.log_dir.rglob("*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or "kind" not in manifest:
            continue
        canonical = canonicalize(manifest)  # legacy spelling, incl. the kind
        kind = canonical["kind"]
        assert (
            kind in inventory
        ), f"{manifest_path}: kind {kind!r} has no legacy inventory"
        extra = normalized_paths(canonical) - set(inventory[kind])
        if extra:
            problems.append(f"{kind} ({manifest_path.name}): {sorted(extra)}")
        checked += 1
    assert checked >= 5, "the composed run wrote fewer manifests than expected"
    assert not problems, "key paths outside the legacy inventory:\n" + "\n".join(
        problems
    )
