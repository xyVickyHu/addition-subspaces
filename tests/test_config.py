from pathlib import Path

import pytest
import yaml

from subspaces.config import (
    ConfigError,
    Step1Config,
    dump_resolved,
    load_step1_config,
    requested_semantics,
    to_dict,
)
from subspaces.paths import find_repo_root

REPO = find_repo_root(Path(__file__))


def test_load_committed_corrected_config():
    cfg = load_step1_config(REPO / "configs" / "step1_number_add_llama3.yaml")
    assert cfg.protocol == "corrected_holdout"
    assert cfg.scan.c_min == 0 and cfg.scan.c_max == 20
    assert cfg.samples.activation.task_set == "train"
    assert cfg.samples.final_eval.task_set == "eval"
    assert cfg.main_selector.name == "unified"
    assert cfg.main_selector.params == {
        "beta": 0.25,
        "floor": 0.12,
        "clean_min": 0.25,
        "minor_z": 3.5,
    }


def test_load_committed_legacy_config():
    cfg = load_step1_config(
        REPO / "configs" / "step1_number_add_llama3_legacy_repro.yaml"
    )
    assert cfg.protocol == "legacy_reproduction"
    assert cfg.scan.c_max == 12
    assert cfg.main_selector.name == "recovery_weak"
    assert cfg.main_selector.params == {
        "rel_floor": 0.2,
        "abs_floor": 0.01,
        "eps": 0.01,
    }


def test_raw_coef_contract_is_all_heads_from_final_checkpoint(tmp_path):
    cfg = Step1Config(protocol="legacy_reproduction")
    cfg.matrix.reuse_path = "x"
    cfg.validate()
    assert to_dict(cfg.raw_coef) == {
        "coefficient_source": "final_checkpoint",
        "head_scope": "all",
    }

    base = {"protocol": "legacy_reproduction", "matrix": MATRIX}
    for raw_coef, match in (
        ({"coefficient_source": "selection_checkpoint"}, "final_checkpoint"),
        ({"head_scope": "selected"}, "head_scope must be all"),
    ):
        with pytest.raises(ConfigError, match=match):
            load_step1_config(_write(tmp_path, {**base, "raw_coef": raw_coef}))


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_unknown_key_rejected(tmp_path):
    path = _write(tmp_path, {"protocol": "legacy_reproduction", "nope": 1})
    with pytest.raises(ConfigError, match="unknown key"):
        load_step1_config(path)


def test_nested_unknown_key_rejected(tmp_path):
    path = _write(
        tmp_path,
        {"protocol": "legacy_reproduction", "model": {"names": "x"}},
    )
    with pytest.raises(ConfigError, match="unknown key"):
        load_step1_config(path)


def test_wrong_type_rejected(tmp_path):
    path = _write(
        tmp_path,
        {"protocol": "legacy_reproduction", "task": {"n_shot": "five"}},
    )
    with pytest.raises(ConfigError, match="expected int"):
        load_step1_config(path)


MATRIX = {"mode": "reuse", "reuse_path": "matrices/toy"}


def test_corrected_requires_split_and_train_task_sets(tmp_path):
    with pytest.raises(ConfigError, match="task_split"):
        load_step1_config(
            _write(tmp_path, {"protocol": "corrected_holdout", "matrix": MATRIX})
        )
    payload = {
        "protocol": "corrected_holdout",
        "matrix": MATRIX,
        "task": {"task_split": "configs/task_splits/number_add_paper.yaml"},
        "samples": {"scan": {"examples_per_task": 20, "task_set": "all"}},
    }
    with pytest.raises(ConfigError, match="task_set=train"):
        load_step1_config(_write(tmp_path, payload))


def test_missing_reuse_path_rejected(tmp_path):
    with pytest.raises(ConfigError, match="reuse_path"):
        load_step1_config(_write(tmp_path, {"protocol": "legacy_reproduction"}))


def test_bad_scan_grid_rejected(tmp_path):
    payload = {
        "protocol": "legacy_reproduction",
        "matrix": MATRIX,
        "scan": {"c_min": 5, "c_max": 20},
    }
    with pytest.raises(ConfigError, match="c_min must be 0"):
        load_step1_config(_write(tmp_path, payload))
    payload["scan"] = {"c_min": 0, "c_max": 0}
    with pytest.raises(ConfigError, match="c_max must be >= 1"):
        load_step1_config(_write(tmp_path, payload))


def test_heldout_and_final_eval_seeds_must_differ_under_corrected(tmp_path):
    payload = {
        "protocol": "corrected_holdout",
        "matrix": MATRIX,
        "task": {"task_split": "configs/task_splits/number_add_paper.yaml"},
        "samples": {
            "heldout_activation": {
                "examples_per_task": 100,
                "seed": 42,
                "task_set": "eval",
            },
            "final_eval": {"examples_per_task": 100, "seed": 42, "task_set": "eval"},
        },
    }
    with pytest.raises(ConfigError, match="heldout_activation.seed"):
        load_step1_config(_write(tmp_path, payload))


def test_requested_semantics_excludes_selector():
    """The main selector is pure-CPU over the saved scan; selector variants
    must share ONE run identity (reuse the matrix/cache/scan)."""
    base = Step1Config(protocol="legacy_reproduction")
    base.matrix.reuse_path = "x"
    other = Step1Config(protocol="legacy_reproduction")
    other.matrix.reuse_path = "x"
    other.main_selector.name = "recovery_weak"
    other.main_selector.params = {"rel_floor": 0.2, "abs_floor": 0.01, "eps": 0.01}
    assert requested_semantics(base) == requested_semantics(other)


def test_discriminated_matrix_schema(tmp_path):
    reuse_with_train = {
        "protocol": "legacy_reproduction",
        "matrix": {"mode": "reuse", "reuse_path": "x", "train": {"seed": 1}},
    }
    with pytest.raises(ConfigError, match="must not set matrix.train"):
        load_step1_config(_write(tmp_path, reuse_with_train))

    train_with_reuse = {
        "protocol": "legacy_reproduction",
        "matrix": {"mode": "train", "reuse_path": "x", "train": {"seed": 1}},
    }
    with pytest.raises(ConfigError, match="must not set matrix.reuse_path"):
        load_step1_config(_write(tmp_path, train_with_reuse))

    train_without_block = {
        "protocol": "legacy_reproduction",
        "matrix": {"mode": "train"},
    }
    with pytest.raises(ConfigError, match="requires the matrix.train block"):
        load_step1_config(_write(tmp_path, train_without_block))

    valid_train = {
        "protocol": "legacy_reproduction",
        "matrix": {"mode": "train", "train": {"seed": 7}},
    }
    cfg = load_step1_config(_write(tmp_path, valid_train))
    assert cfg.matrix.train.seed == 7 and cfg.matrix.reuse_path is None


def test_discriminated_selected_schema(tmp_path):
    base = {"protocol": "legacy_reproduction", "matrix": MATRIX}
    with pytest.raises(ConfigError, match="elbow takes no"):
        load_step1_config(_write(tmp_path, {**base, "selected": {"fraction": 0.9}}))
    with pytest.raises(ConfigError, match="requires fraction"):
        load_step1_config(
            _write(tmp_path, {**base, "selected": {"method": "fraction"}})
        )
    with pytest.raises(ConfigError, match="requires fixed_threshold"):
        load_step1_config(_write(tmp_path, {**base, "selected": {"method": "fixed"}}))
    cfg = load_step1_config(
        _write(
            tmp_path,
            {**base, "selected": {"method": "fixed", "fixed_threshold": 0.2}},
        )
    )
    assert cfg.selected.fixed_threshold == 0.2 and cfg.selected.fraction is None


def test_discriminated_largest_gap_schema(tmp_path):
    base = {"protocol": "legacy_reproduction", "matrix": MATRIX}
    cfg = load_step1_config(
        _write(tmp_path, {**base, "selected": {"method": "largest_gap"}})
    )
    assert cfg.selected.method == "largest_gap"
    assert cfg.selected.fraction is None
    assert cfg.selected.fixed_threshold is None
    # parameter-free: the other branches' fields are rejected
    with pytest.raises(ConfigError, match="largest_gap takes no"):
        load_step1_config(
            _write(
                tmp_path,
                {**base, "selected": {"method": "largest_gap", "fraction": 0.9}},
            )
        )
    with pytest.raises(ConfigError, match="largest_gap takes no"):
        load_step1_config(
            _write(
                tmp_path,
                {
                    **base,
                    "selected": {
                        "method": "largest_gap",
                        "fixed_threshold": 0.2,
                    },
                },
            )
        )
    with pytest.raises(ConfigError, match="unknown selected-set method .*registered"):
        load_step1_config(
            _write(tmp_path, {**base, "selected": {"method": "biggest_gap"}})
        )


def test_committed_siggap_variant_configs_load():
    for name in ("step1_number_add_llama3_siggap.yaml",):
        cfg = load_step1_config(REPO / "configs" / name)
        assert cfg.selected.method == "largest_gap"
        assert cfg.protocol == "corrected_holdout"
        assert cfg.matrix.mode == "reuse"


def test_run_name_sanitized(tmp_path):
    for bad in ("../evil", "a/b", "", " lead", "x" * 80):
        payload = {
            "run_name": bad,
            "protocol": "legacy_reproduction",
            "matrix": MATRIX,
        }
        with pytest.raises(ConfigError, match="run_name"):
            load_step1_config(_write(tmp_path, payload))


def test_requested_semantics_excludes_presentation_fields():
    base = Step1Config(protocol="legacy_reproduction")
    base.matrix.reuse_path = "somewhere/toy"
    semantics = requested_semantics(base)
    assert "run_name" not in semantics
    assert "compute" not in semantics
    assert "reuse_path" not in semantics["matrix"]  # locator
    assert "device" not in semantics["model"]  # execution detail
    assert semantics["model"]["dtype"] == "bfloat16"  # affects numerics: kept
    assert semantics["scan"]["c_max"] == 20

    renamed = Step1Config(run_name="other", protocol="legacy_reproduction")
    renamed.matrix.reuse_path = "elsewhere/toy"
    assert requested_semantics(renamed) == semantics  # name+locator excluded

    changed = Step1Config(protocol="legacy_reproduction")
    changed.matrix.reuse_path = "somewhere/toy"
    changed.scan.c_max = 12
    assert requested_semantics(changed) != semantics


def test_selector_schema_validated_at_load(tmp_path):
    base = {"protocol": "legacy_reproduction", "matrix": MATRIX}
    with pytest.raises(ConfigError, match="unknown main_selector"):
        load_step1_config(
            _write(tmp_path, {**base, "main_selector": {"name": "magic"}})
        )
    incomplete = {
        "name": "unified",
        "params": {"beta": 0.25, "floor": 0.12, "clean_min": 0.25},  # minor_z gone
    }
    with pytest.raises(ConfigError, match="EXACTLY the parameters"):
        load_step1_config(_write(tmp_path, {**base, "main_selector": incomplete}))
    extra = {
        "name": "recovery_weak",
        "params": {"rel_floor": 0.2, "abs_floor": 0.01, "eps": 0.01, "bonus": 1.0},
    }
    with pytest.raises(ConfigError, match="EXACTLY the parameters"):
        load_step1_config(_write(tmp_path, {**base, "main_selector": extra}))
    with pytest.raises(ConfigError, match="pin requires pin_heads"):
        load_step1_config(
            _write(tmp_path, {**base, "main_selector": {"name": "pin", "params": {}}})
        )
    with pytest.raises(ConfigError, match="only valid with the pin selector"):
        load_step1_config(
            _write(
                tmp_path,
                {
                    **base,
                    "main_selector": {
                        "name": "unified",
                        "params": {
                            "beta": 0.25,
                            "floor": 0.12,
                            "clean_min": 0.25,
                            "minor_z": 3.5,
                        },
                        "pin_heads": "15:2",
                    },
                },
            )
        )
    valid_pin = {"name": "pin", "params": {}, "pin_heads": "15:2,15:1,13:6"}
    cfg = load_step1_config(_write(tmp_path, {**base, "main_selector": valid_pin}))
    assert cfg.main_selector.pin_heads == "15:2,15:1,13:6"
    # largest_gap is parameter-free: params must be an explicit EMPTY dict
    # (the dataclass default is unified's parameter set, which this selector
    # must reject).
    valid_gap = {"name": "largest_gap", "params": {}}
    cfg = load_step1_config(_write(tmp_path, {**base, "main_selector": valid_gap}))
    assert cfg.main_selector.name == "largest_gap"
    assert cfg.main_selector.params == {}
    stray = {"name": "largest_gap", "params": {"beta": 0.25}}
    with pytest.raises(ConfigError, match="EXACTLY the parameters"):
        load_step1_config(_write(tmp_path, {**base, "main_selector": stray}))
    # above_ablation: exactly {alpha, scanwise}; the _zero variant is
    # parameter-free (explicit empty dict, same rule as largest_gap).
    valid_above = {
        "name": "above_ablation",
        "params": {"alpha": 0.05, "scanwise": 0.0},
    }
    cfg = load_step1_config(_write(tmp_path, {**base, "main_selector": valid_above}))
    assert cfg.main_selector.params == {"alpha": 0.05, "scanwise": 0.0}
    with pytest.raises(ConfigError, match="EXACTLY the parameters"):
        load_step1_config(
            _write(
                tmp_path,
                {
                    **base,
                    "main_selector": {
                        "name": "above_ablation",
                        "params": {"alpha": 0.05},  # scanwise missing
                    },
                },
            )
        )
    valid_above_zero = {"name": "above_ablation_zero", "params": {}}
    cfg = load_step1_config(
        _write(tmp_path, {**base, "main_selector": valid_above_zero})
    )
    assert cfg.main_selector.name == "above_ablation_zero"
    with pytest.raises(ConfigError, match="EXACTLY the parameters"):
        load_step1_config(
            _write(
                tmp_path,
                {
                    **base,
                    "main_selector": {
                        "name": "above_ablation_zero",
                        "params": {"alpha": 0.05},
                    },
                },
            )
        )


def test_matrix_reuse_extras_validated(tmp_path):
    bad_epoch = {
        "protocol": "legacy_reproduction",
        "matrix": {**MATRIX, "checkpoint_epoch": -1},
    }
    with pytest.raises(ConfigError, match="checkpoint_epoch"):
        load_step1_config(_write(tmp_path, bad_epoch))
    bad_override = {
        "protocol": "legacy_reproduction",
        "matrix": {**MATRIX, "override_provenance": ["favorite_color"]},
    }
    with pytest.raises(ConfigError, match="unknown field"):
        load_step1_config(_write(tmp_path, bad_override))


def test_dump_resolved_round_trip(tmp_path):
    cfg = load_step1_config(REPO / "configs" / "step1_number_add_llama3.yaml")
    out = tmp_path / "resolved.yaml"
    dump_resolved(cfg, out)
    reloaded = load_step1_config(out)
    assert to_dict(reloaded) == to_dict(cfg)


def test_unknown_prompt_format_rejected_at_load(fake_paths):
    import yaml

    from subspaces.config import ConfigError, load_step1_config

    config_path = fake_paths.root / "configs" / "step1_test.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["task"]["prompt_format"] = "qaa"  # typo'd format must fail on CPU
    bad = fake_paths.root / "configs" / "step1_test_badfmt.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="not a registered format"):
        load_step1_config(bad)
