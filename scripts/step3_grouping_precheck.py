"""CPU tokenizer-only pre-validation of step-3 grouping for a set of cells.

For every cell in ``configs/step23_recpos_cells.tsv`` (or ``--cells``), loads
the cell's step-3 context + scan samples manifest and runs the REAL grouping
v2 math (``prompt_parts`` + ``token_group_spans_offsets`` on the HF fast
tokenizer's offset_mapping) over every record; the v1 prefix-consistency
count is reported alongside as the share of prompts where v1 == v2 spans by
construction. The BOS shim mirrors TransformerLens ``to_tokens``:
``[bos] + encode(text, add_special_tokens=False)`` (no bos for tokenizers
without one, e.g. Qwen2.5). Advisory only: the GPU run re-checks offsets per
prompt and enforces the reconstruction gate.

    PYTHONPATH=$PWD .venv/bin/python scripts/step3_grouping_precheck.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from subspaces.paths import ProjectPaths
from subspaces.step3.token_groups import (
    GroupingError,
    prompt_parts,
    token_group_spans,
    token_group_spans_offsets,
)


def shim_to_tokens(tokenizer):
    bos = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]

    def to_tokens(text: str) -> list[int]:
        return bos + tokenizer(text, add_special_tokens=False)["input_ids"]

    return to_tokens


def check_cell(paths: ProjectPaths, context_path: Path, tokenizers: dict) -> dict:
    context = yaml.safe_load(context_path.read_text(encoding="utf-8"))
    samples_ref = next(iter(context["samples"].values()))
    manifest = json.loads(
        paths.resolve(samples_ref["path"]).read_text(encoding="utf-8")
    )
    prompt_format = manifest["prompt_format"]
    n_shot = int(manifest["config"]["task"]["n_shot"])
    model_name = context["model"]["name"]
    if model_name not in tokenizers:
        from transformers import AutoTokenizer

        tokenizers[model_name] = AutoTokenizer.from_pretrained(
            model_name, revision=context["model"].get("revision")
        )
    to_tokens = shim_to_tokens(tokenizers[model_name])

    tokenizer = tokenizers[model_name]
    n_bos = 0 if tokenizer.bos_token_id is None else 1

    n_ok = n_fail = n_v1_consistent = 0
    group_orders = set()
    first_error = None
    for task_id, task in manifest["tasks"].items():
        for record in task["samples"]:
            prompt = record["prompt"]
            try:
                parts = prompt_parts(record, prompt_format, n_shot)
                encoded = tokenizer(
                    prompt, return_offsets_mapping=True, add_special_tokens=False
                )
                offsets = [
                    (int(start), int(end)) for start, end in encoded["offset_mapping"]
                ]
                spans = token_group_spans_offsets(parts, prompt, offsets, n_bos)
            except GroupingError as err:
                n_fail += 1
                if first_error is None:
                    first_error = f"{task_id}: {err}"
                continue
            n_ok += 1
            group_orders.add(tuple(name for name, _start, _end in spans))
            try:
                token_group_spans(to_tokens, parts, prompt)
                n_v1_consistent += 1
            except GroupingError:
                pass
    return {
        "format": prompt_format["name"],
        "n_shot": n_shot,
        "model": model_name,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "n_v1_consistent": n_v1_consistent,
        "n_group_orders": len(group_orders),
        "first_error": first_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--cells", default="configs/step23_recpos_cells.tsv")
    parser.add_argument(
        "--context-pattern", default="configs/contexts/step3_recpos_{slug}.yaml"
    )
    args = parser.parse_args()
    paths = ProjectPaths.from_root(args.root)
    tokenizers: dict = {}
    failures = 0
    for line in paths.resolve(args.cells).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        slug = line.split("\t", 1)[0]
        context_path = paths.resolve(args.context_pattern.format(slug=slug))
        result = check_cell(paths, context_path, tokenizers)
        status = (
            "OK" if result["n_fail"] == 0 and result["n_group_orders"] == 1 else "WARN"
        )
        if status == "WARN":
            failures += 1
        print(
            f"[{status}] {slug}: fmt={result['format']} n_shot={result['n_shot']} "
            f"ok={result['n_ok']} fail={result['n_fail']} "
            f"v1_consistent={result['n_v1_consistent']} "
            f"group_orders={result['n_group_orders']}"
            + (f" first_error={result['first_error']}" if result["first_error"] else "")
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
