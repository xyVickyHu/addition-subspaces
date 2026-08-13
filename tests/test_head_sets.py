import pytest

from subspaces.artifacts import make_manifest, write_json_atomic
from subspaces.head_sets import (
    HeadSpecError,
    format_heads,
    load_head_set,
    parse_heads,
    resolve_heads_arg,
    validate_heads,
)


def test_parse_heads_colon_form():
    assert parse_heads("15:2,15:1,13:6") == [(15, 2), (15, 1), (13, 6)]
    assert parse_heads(" 15:2 , 13:6 ") == [(15, 2), (13, 6)]


def test_parse_heads_rejects_paren_form_with_guidance():
    with pytest.raises(HeadSpecError, match="colon form"):
        parse_heads("(15,2),(15,1)")


def test_parse_heads_rejects_garbage():
    with pytest.raises(HeadSpecError, match="expected 'L:H'"):
        parse_heads("15:2:3")
    with pytest.raises(HeadSpecError, match="integers"):
        parse_heads("a:b")
    with pytest.raises(HeadSpecError, match="empty"):
        parse_heads(" , ")


def test_validate_heads_bounds_and_duplicates():
    with pytest.raises(HeadSpecError, match="duplicate"):
        validate_heads([(15, 2), (15, 2)])
    with pytest.raises(HeadSpecError, match="layer 40 out of range"):
        validate_heads([(40, 0)], n_layers=32, n_heads=32)
    with pytest.raises(HeadSpecError, match="head 33 out of range"):
        validate_heads([(0, 33)], n_layers=32, n_heads=32)
    assert validate_heads([(15, 2)], n_layers=32, n_heads=32) == [(15, 2)]


def _write_heads_artifact(fake_paths, tmp_path):
    manifest = make_manifest(
        kind="heads",
        schema_version=1,
        paths=fake_paths,
        payload={
            "significant_heads": [[15, 2], [15, 1], [13, 6], [10, 0]],
            "main_heads": [[15, 2], [15, 1], [13, 6]],
            "minor_heads": [[15, 28]],
            "model_dims": {"n_layers": 32, "n_heads": 32},
        },
    )
    path = tmp_path / "heads.json"
    write_json_atomic(path, manifest)
    return path


def test_load_head_set_all_sets(fake_paths, tmp_path):
    path = _write_heads_artifact(fake_paths, tmp_path)
    assert load_head_set(path, "main") == [(15, 2), (15, 1), (13, 6)]
    assert load_head_set(path, "minor") == [(15, 28)]
    assert len(load_head_set(path, "significant")) == 4
    with pytest.raises(HeadSpecError, match="which must be one of"):
        load_head_set(path, "all")


def test_resolve_heads_arg_xor(fake_paths, tmp_path):
    path = _write_heads_artifact(fake_paths, tmp_path)
    assert resolve_heads_arg(heads="15:2", heads_artifact=None, paths=fake_paths) == [
        (15, 2)
    ]
    assert resolve_heads_arg(
        heads=None, heads_artifact=path, head_set="main", paths=fake_paths
    ) == [(15, 2), (15, 1), (13, 6)]
    with pytest.raises(HeadSpecError, match="not both"):
        resolve_heads_arg(heads="15:2", heads_artifact=path, paths=fake_paths)
    with pytest.raises(HeadSpecError, match="specify heads"):
        resolve_heads_arg(heads=None, heads_artifact=None, paths=fake_paths)


def test_format_heads_round_trip():
    heads = [(15, 2), (13, 6)]
    assert parse_heads(format_heads(heads)) == heads
