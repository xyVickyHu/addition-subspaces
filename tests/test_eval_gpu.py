"""CPU tests of the real eval_prompts internals (legacy generation mocked)."""

from __future__ import annotations

import pytest

from subspaces.artifacts import ArtifactError
from subspaces.step1.eval_gpu import eval_prompts

PROMPTS = [f"p{i}->" for i in range(7)]
TARGETS = [str(i) for i in range(7)]


def _mock_generation(flags_per_call):
    calls = []

    def fake(model, prompts, targets, **kwargs):
        calls.append((list(prompts), kwargs))
        flags = flags_per_call(len(prompts))
        return 0.0, ["x"] * len(prompts), flags

    return fake, calls


def test_partial_final_batch_included_and_sizes_recorded(monkeypatch):
    import subspaces.utils.intervene as intervene

    fake, calls = _mock_generation(lambda n: [True] * n)
    monkeypatch.setattr(intervene, "intervened_generation_with_accuracy", fake)

    correct, sizes = eval_prompts(object(), PROMPTS, TARGETS, batch_size=3)
    assert correct == [1] * 7
    assert sizes == [3, 3, 1]  # final partial batch evaluated, not dropped
    assert [len(p) for p, _ in calls] == [3, 3, 1]
    # mode-0 ADD semantics forwarded on every call
    assert all(k["intervention_mode"] == 0 for _, k in calls)


def test_dropped_example_refuses(monkeypatch):
    import subspaces.utils.intervene as intervene

    fake, _calls = _mock_generation(lambda n: [True] * max(n - 1, 0))
    monkeypatch.setattr(intervene, "intervened_generation_with_accuracy", fake)
    with pytest.raises(ArtifactError, match="dropped"):
        eval_prompts(object(), PROMPTS, TARGETS, batch_size=4)


def test_length_mismatch_refuses():
    with pytest.raises(ValueError, match="length mismatch"):
        eval_prompts(object(), PROMPTS, TARGETS[:-1], batch_size=4)


def test_vector_converted_to_model_device_and_dtype(monkeypatch):
    """A CPU / wrong-dtype vector must be converted before the legacy hook
    (mode-0 ADD rejects device and dtype mismatches on GPU)."""
    from types import SimpleNamespace

    import torch

    import subspaces.utils.intervene as intervene

    received = {}

    def fake(model, prompts, targets, layer_name=None, intervention_vector=None, **kw):
        received["vector"] = intervention_vector
        return 0.0, ["x"] * len(prompts), [True] * len(prompts)

    monkeypatch.setattr(intervene, "intervened_generation_with_accuracy", fake)
    model = SimpleNamespace(cfg=SimpleNamespace(device="cpu", dtype=torch.bfloat16))
    wrong = torch.randn(4, dtype=torch.float32)
    correct, _sizes = eval_prompts(
        model,
        PROMPTS[:2],
        TARGETS[:2],
        layer_name="blocks.0.hook_resid_mid",
        vector=wrong,
        batch_size=2,
    )
    assert correct == [1, 1]
    assert received["vector"].dtype == torch.bfloat16
    assert received["vector"].device.type == "cpu"


def test_gpu_peak_memory_none_off_gpu(monkeypatch):
    import torch

    from subspaces.step1.eval_gpu import gpu_peak_memory

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert gpu_peak_memory() is None


def test_gpu_peak_memory_shape_when_cuda_reports(monkeypatch):
    """Observability dict shape (values in GiB, rounded); CUDA calls mocked."""
    from types import SimpleNamespace

    import torch

    from subspaces.step1.eval_gpu import gpu_peak_memory

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda d: "FakeGPU")
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda d: 17 * 2**30)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda d: 18 * 2**30)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda d: SimpleNamespace(total_memory=40 * 2**30),
    )
    assert gpu_peak_memory() == {
        "device_name": "FakeGPU",
        "max_allocated_gib": 17.0,
        "max_reserved_gib": 18.0,
        "total_gib": 40.0,
    }
