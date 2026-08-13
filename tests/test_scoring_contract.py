"""CPU tests of the versioned scoring contract (registry + config + artifacts)."""

from __future__ import annotations

import json

import pytest

from subspaces.config import ConfigError, load_step1_config
from subspaces.step1.scoring import (
    DEFAULT_SCORING,
    SCORERS,
    ScoringError,
    validate_scoring,
)


def test_registry_contains_exactly_the_approved_scorers():
    # legacy v1 + the two corrected scorers approved by a resolved design
    # decision (2026-07-26); any further addition must update this pin
    # deliberately.
    assert set(SCORERS) == {
        ("exact_token_match", 1),
        ("numeric_parse", 1),
        ("normalized_exact", 1),
    }


def test_validate_scoring_accepts_the_default():
    out = validate_scoring("exact_token_match", 1, None)
    assert out == DEFAULT_SCORING


@pytest.mark.parametrize(
    "name,version,params,match",
    [
        ("exact_token_match", 2, None, "unknown scorer"),
        ("normalized_string", 1, None, "unknown scorer"),
        ("exact_token_match", 1, {"slack": 2.0}, "requires exactly params"),
    ],
)
def test_validate_scoring_refuses(name, version, params, match):
    with pytest.raises(ScoringError, match=match):
        validate_scoring(name, version, params)


def test_config_default_scoring_and_refusal(fake_paths):
    cfg = load_step1_config(fake_paths.root / "configs" / "step1_test.yaml")
    assert cfg.scoring.name == "exact_token_match"
    assert cfg.scoring.version == 1
    cfg.scoring.version = 3
    with pytest.raises(ConfigError, match="scoring"):
        cfg.validate()


def test_scan_and_headset_artifacts_record_scoring_and_macro(fake_repo, gpu_stubs):
    from subspaces.paths import ProjectPaths
    from subspaces.step1 import pipeline

    paths = ProjectPaths.from_root(fake_repo)
    cfg = load_step1_config(fake_repo / "configs" / "step1_test.yaml")
    summary = pipeline.run(cfg, paths)

    scan_node = paths.resolve(summary["nodes"]["scan"])
    scan = json.loads((scan_node / "head_scan.json").read_text(encoding="utf-8"))
    assert scan["config"]["scoring"] == {
        "name": "exact_token_match",
        "version": 1,
        "params": None,
    }
    assert set(scan["baselines_macro"]) == set(scan["baselines"])

    main_node = paths.resolve(summary["latest_selection"]["main_node"])
    headset_path = next(main_node.glob("eval-*.json"))
    headset = json.loads(headset_path.read_text(encoding="utf-8"))
    assert headset["config"]["scoring"]["name"] == "exact_token_match"
    assert set(headset["metrics_macro"]) == set(headset["metrics"])
    # equal per-task counts in the fake repo: micro == macro
    for variant, value in headset["metrics"].items():
        assert headset["metrics_macro"][variant] == pytest.approx(value)
    # aggregate denominators recorded
    assert headset["n_eval_total"] == sum(d["n"] for d in headset["per_task"].values())
