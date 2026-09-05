"""Generate a Step-3 analysis-context YAML from a Step-1 head artifact (CPU).

Step-3's canonical inputs are the two DECOUPLED
selection-side draws of the same run: PROMPTS from the scan manifest and
DIRECTIONS from the activation-draw z cache. This walks the artifact lineage
instead of guessing: heads artifact (``heads.json`` or a selector node's
``main_heads.json``) -> ``inputs.head_scan.path`` -> ``head_scan.json`` ->
{``inputs.z_cache.content_fingerprint``, ``inputs.scan_samples.path``};
model identity from the z cache's meta; the ``task`` block from the scan
samples manifest. The emitted mapping is an ordinary context file
(``subspaces/context.py``) with the single-reference ``activations`` form step 3
requires, and carries no identity of its own. Review, commit (e.g. under
``configs/contexts/``), then::

    .venv/bin/python scripts/make_step3_context.py \\
        --heads-artifact log/runs/<node>/selected-*/scan-*/significant-paired_bh-v1-<h8>/main_heads.json \\
        --out configs/contexts/step3_<cell>.yaml
    sbatch -p YOUR_PARTITION scripts/step3_run.sbatch configs/contexts/step3_<cell>.yaml
"""

from __future__ import annotations

import argparse
import json
import sys

import yaml

from subspaces.artifacts import ArtifactError, semantic_fingerprint
from subspaces.head_sets import read_heads_manifest
from subspaces.paths import ProjectPaths
from subspaces.step2.context_gen import _load_json, walk_to_scan


def build_step3_context(
    paths: ProjectPaths, heads_artifact: str, *, samples_name: str = "analysis"
) -> dict:
    artifact_path = paths.resolve(heads_artifact)
    heads_manifest, _kind = read_heads_manifest(artifact_path)

    scan = walk_to_scan(paths, artifact_path)
    scan_inputs = scan.get("inputs") or {}
    z_ref = scan_inputs.get("z_cache") or {}
    z_fp = z_ref.get("content_fingerprint")
    if not z_fp:
        raise ArtifactError("head_scan.json carries no inputs.z_cache fingerprint")
    z_dir = paths.z_cache_dir / z_fp
    z_meta = _load_json(z_dir / "meta.json")

    samples_ref = scan_inputs.get("scan_samples") or {}
    if not samples_ref.get("path"):
        raise ArtifactError("head_scan.json carries no inputs.scan_samples path")
    samples_path = paths.resolve(samples_ref["path"])
    samples_manifest = _load_json(samples_path)
    task_cfg = (samples_manifest.get("config") or {}).get("task") or {}
    format_name = (samples_manifest.get("prompt_format") or {}).get("name")
    if format_name and task_cfg.get("prompt_format") not in (None, format_name):
        raise ArtifactError(
            f"scan samples manifest is inconsistent: config.task.prompt_format="
            f"{task_cfg.get('prompt_format')!r} vs prompt_format.name="
            f"{format_name!r}"
        )

    return {
        "schema_version": 1,
        "model": dict(z_meta["identity"]["model"]),
        "task": {
            "family": task_cfg.get("dataset_dir"),
            "prompt_format": task_cfg.get("prompt_format") or format_name,
            "n_shot": task_cfg.get("n_shot"),
        },
        "samples": {samples_name: {"path": paths.relativize(samples_path)}},
        "activations": {"path": paths.relativize(z_dir)},
        "heads_artifact": {
            "path": paths.relativize(artifact_path),
            "semantic_fingerprint": semantic_fingerprint(heads_manifest),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument(
        "--heads-artifact",
        required=True,
        help="heads.json or main_heads.json to analyze",
    )
    parser.add_argument(
        "--samples-name",
        default="analysis",
        help="context.samples entry name (step3 default: analysis)",
    )
    parser.add_argument("--out", required=True, help="output context YAML path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = ProjectPaths.from_root(args.root)
        context = build_step3_context(
            paths, args.heads_artifact, samples_name=args.samples_name
        )
    except (ArtifactError, OSError, KeyError, json.JSONDecodeError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    missing = [key for key, value in context["task"].items() if value is None]
    if missing:
        print(f"warning: context.task fields unresolved: {missing}", file=sys.stderr)
    out_path = paths.resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(context, sort_keys=False), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
