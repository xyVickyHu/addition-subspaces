"""Lane-A GPU equivalence tests (marked ``gpu``; skipped without CUDA).

Run on the tiny-smoke GPU node BEFORE the calibration scan:

    .venv/bin/pytest -m gpu tests/test_gpu_lane_a.py

Old and new implementations are compared on IDENTICAL materialized prompts —
no claim is made about recovering historical unseeded draws.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]

REPO = Path(__file__).resolve().parents[1]
N_SHOT = 5
N_EXAMPLE = 6
SEED = 20260721


@pytest.fixture(scope="module")
def model():
    from subspaces.config import load_step1_config
    from subspaces.step1.eval_gpu import load_model

    cfg = load_step1_config(REPO / "configs" / "step1_number_add_llama3.yaml")
    return load_model(cfg)


def _task_examples() -> list[dict]:
    return json.loads(
        (REPO / "dataset_files" / "number_add" / "number-add7.json").read_text(
            encoding="utf-8"
        )
    )


def _replay_legacy_prompt_draws(examples: list[dict]) -> list[str]:
    """Reproduce EXACTLY the prompts legacy compute_z_result_per_task draws
    under a fixed global seed (mirrors its sampling loop line for line)."""
    from subspaces.utils.data import format_input

    task_input = [item["input"] for item in examples]
    random.seed(SEED)
    prompts = []
    for _ in range(N_EXAMPLE):
        x_q = random.choice(task_input)
        remaining_input = [item for item in task_input if item != x_q]
        x_icl = random.sample(remaining_input, N_SHOT)
        prompts.append(format_input(x_icl, x_q, examples))
    return prompts


def test_z_extraction_matches_legacy_on_identical_prompts(model):
    """torch.testing.assert_close between legacy compute_z_result_per_task and
    the manifest-driven extraction on the SAME prompts."""
    from subspaces.step1.eval_gpu import compute_z_for_tasks
    from subspaces.utils.activations import compute_z_result_per_task

    examples = _task_examples()
    prompts = _replay_legacy_prompt_draws(examples)

    random.seed(SEED)  # legacy draws internally under the same seed
    legacy_z = compute_z_result_per_task(
        model, {"number-add7": examples}, "number-add7", N_SHOT, N_EXAMPLE
    )

    manifest = {
        "task_order": ["number-add7"],
        "tasks": {
            "number-add7": {
                "samples": [{"prompt": p, "expected": "x"} for p in prompts]
            }
        },
    }
    new_z = compute_z_for_tasks(model, manifest, ["number-add7"])["number-add7"]
    torch.testing.assert_close(
        new_z.float(), legacy_z.cpu().float(), rtol=1e-4, atol=1e-4
    )


def test_eval_prompts_matches_legacy_and_batching_is_stable(model):
    from subspaces.step1.eval_gpu import eval_prompts
    from subspaces.utils.intervene import intervened_generation_with_accuracy

    examples = _task_examples()
    prompts = _replay_legacy_prompt_draws(examples)
    targets = [prompt.rsplit("#", 1)[-1].split("->")[0] for prompt in prompts]
    targets = [
        next(item["output"] for item in examples if item["input"] == query)
        for query in targets
    ]

    # single batch: identical plumbing by construction
    _acc, _gen, legacy_flags = intervened_generation_with_accuracy(
        model, prompts, targets, print_correct_list=True
    )
    new_flags, batch_sizes = eval_prompts(
        model, prompts, targets, batch_size=len(prompts)
    )
    assert new_flags == [int(bool(flag)) for flag in legacy_flags]
    assert batch_sizes == [len(prompts)]

    # smaller batches (incl. a final partial batch): high agreement expected;
    # bf16 padding effects may flip rare borderline generations
    rebatched_flags, rebatched_sizes = eval_prompts(
        model, prompts, targets, batch_size=4
    )
    assert sum(rebatched_sizes) == len(prompts)
    agreement = sum(
        int(a == b) for a, b in zip(new_flags, rebatched_flags, strict=True)
    ) / len(prompts)
    assert agreement >= 0.95


def test_intervened_path_matches_legacy_add_mode(model):
    """The intervened evaluation (mode-0 ADD on ZERO-SHOT prompts) must equal
    the direct legacy call on identical inputs — the path where the replace/
    zero-shot port bugs would hide."""
    import torch

    from subspaces.step1.eval_gpu import eval_prompts
    from subspaces.utils.intervene import intervened_generation_with_accuracy

    examples = _task_examples()
    zero_shot_prompts = [f"{example['input']}->" for example in examples[:8]]
    targets = [example["output"] for example in examples[:8]]
    # ONE tensor on the model's actual device+dtype, used by BOTH paths (the
    # direct legacy call takes it as-is; attempt-3 failure: a CPU vector here
    # crashed the legacy call itself, out of eval_prompts' reach)
    generator = torch.Generator().manual_seed(SEED)
    vector = (torch.randn(model.cfg.d_model, generator=generator) * 0.1).to(
        device=model.cfg.device, dtype=model.cfg.dtype
    )
    layer_name = "blocks.10.hook_resid_mid"

    _acc, _gen, legacy_flags = intervened_generation_with_accuracy(
        model,
        zero_shot_prompts,
        targets,
        layer_name,
        vector,
        intervention_mode=0,
        print_correct_list=True,
    )
    new_flags, _sizes = eval_prompts(
        model,
        zero_shot_prompts,
        targets,
        layer_name=layer_name,
        vector=vector,
        batch_size=len(zero_shot_prompts),
    )
    assert new_flags == [int(bool(flag)) for flag in legacy_flags]
