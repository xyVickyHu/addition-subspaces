"""Step-3 token-group decomposition: grouping math, statistics, artifact
identity/reuse/refusal, and the CLI happy path (model boundary stubbed)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml
from conftest import tree_snapshot

from subspaces.artifacts import make_manifest, write_json_atomic
from subspaces.paths import ProjectPaths
from subspaces.runners import step3 as step3_cli
from subspaces.step3 import token_groups as tg
from subspaces.step3.api import Step3Config

ARROW = {"name": "arrow", "in_pre": "", "out_pre": "->", "in_sep": "", "out_sep": "#"}


def char_tokens(text: str) -> list[int]:
    """Char-level fake tokenizer with a BOS token (id 0); prefix-consistent."""
    return [0] + [ord(ch) for ch in text]


def make_record(demos, query):
    prompt = "".join(f"{x}->{y}#" for x, y in demos) + f"{query[0]}->"
    return {
        "demos": [{"input": x, "output": y} for x, y in demos],
        "query": {"input": query[0], "output": query[1]},
        "prompt": prompt,
        "zero_shot_prompt": f"{query[0]}->",
        "expected": query[1],
    }


# --------------------------------------------------------------------------- #
# grouping                                                                     #
# --------------------------------------------------------------------------- #


def test_prompt_parts_roundtrip_and_roles():
    record = make_record([("1", "3"), ("2", "4")], ("5", "7"))
    parts = tg.prompt_parts(record, ARROW, n_shot=2)
    assert [name for name, _ in parts] == [
        "demo1_input",
        "demo1_arrow",
        "demo1_output",
        "demo1_sep",
        "demo2_input",
        "demo2_arrow",
        "demo2_output",
        "demo2_sep",
        "query_input",
        "query_arrow",
    ]
    assert "".join(text for _, text in parts) == record["prompt"]


def test_prompt_parts_rejects_wrong_shot_count_and_tampered_prompt():
    record = make_record([("1", "3")], ("5", "7"))
    with pytest.raises(tg.GroupingError):
        tg.prompt_parts(record, ARROW, n_shot=2)
    bad = make_record([("1", "3"), ("2", "4")], ("5", "7"))
    bad["prompt"] = bad["prompt"] + " "
    with pytest.raises(tg.GroupingError):
        tg.prompt_parts(bad, ARROW, n_shot=2)


def test_token_group_spans_partition_and_bos():
    record = make_record([("1", "3"), ("2", "4")], ("5", "7"))
    parts = tg.prompt_parts(record, ARROW, n_shot=2)
    spans = tg.token_group_spans(char_tokens, parts, record["prompt"])
    assert spans[0] == ("bos", 0, 1)
    assert spans[-1][0] == "query_arrow"
    # spans partition [0, seq): contiguous, ordered, covering
    prev_end = 0
    for _name, start, end in spans:
        assert start == prev_end
        assert end >= start
        prev_end = end
    assert prev_end == len(char_tokens(record["prompt"]))


def test_token_group_spans_empty_segment():
    fmt = dict(ARROW, out_sep="")  # a format with no separator text
    record = {
        "demos": [{"input": "1", "output": "3"}],
        "query": {"input": "5", "output": "7"},
        "prompt": "1->3" + "5->",
    }
    parts = tg.prompt_parts(record, fmt, n_shot=1)
    spans = tg.token_group_spans(char_tokens, parts, record["prompt"])
    sep = [s for s in spans if s[0] == "demo1_sep"][0]
    assert sep[1] == sep[2]  # empty span, kept as a zero-token group


def test_token_group_spans_prefix_inconsistency_raises():
    def merging_tokens(text: str) -> list[int]:
        # merges the bigram "->" ONLY when followed by a digit — so the
        # prefix ending at "...->" tokenizes differently than the full string
        ids, i = [0], 0
        while i < len(text):
            if text[i : i + 2] == "->" and i + 2 < len(text) and text[i + 2].isdigit():
                ids.append(9999)
                i += 3
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    record = make_record([("1", "3"), ("2", "4")], ("5", "7"))
    parts = tg.prompt_parts(record, ARROW, n_shot=2)
    with pytest.raises(tg.GroupingError):
        tg.token_group_spans(merging_tokens, parts, record["prompt"])


def _char_offsets(prompt: str) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(len(prompt))]


def test_offset_spans_reproduce_prefix_spans_when_consistent():
    """Grouping v2 == v1 bit-exactly on prefix-consistent tokenizations."""
    record = make_record([("1", "3"), ("22", "4")], ("5", "7"))
    parts = tg.prompt_parts(record, ARROW, n_shot=2)
    prompt = record["prompt"]
    v1 = tg.token_group_spans(char_tokens, parts, prompt)
    v2 = tg.token_group_spans_offsets(parts, prompt, _char_offsets(prompt), n_bos=1)
    assert v2 == v1


def test_offset_spans_handle_boundary_merges_first_char_rule():
    """A merge across a segment boundary (v1's hard failure) is assigned to
    the first character's group; the partition stays exact and ordered."""

    def pair_offsets(prompt: str) -> list[tuple[int, int]]:
        # greedy 2-char tokens: every other segment boundary is crossed
        return [(i, min(i + 2, len(prompt))) for i in range(0, len(prompt), 2)]

    record = make_record([("1", "3"), ("2", "4")], ("5", "7"))
    parts = tg.prompt_parts(record, ARROW, n_shot=2)
    prompt = record["prompt"]  # "1->3#2->4#5->" (13 chars -> 7 pair tokens)
    offsets = pair_offsets(prompt)
    spans = tg.token_group_spans_offsets(parts, prompt, offsets, n_bos=1)
    # exact ordered partition of the token axis (bos + 7 pair tokens)
    prev_end = 0
    for _name, start, end in spans:
        assert start == prev_end
        prev_end = end
    assert prev_end == 1 + len(offsets)
    by_name = {name: (start, end) for name, start, end in spans}
    # token 0 = "1-" starts at char 0 (demo1_input); token 1 = ">3" starts at
    # the arrow's second char; the output "3" was swallowed -> empty span
    assert by_name["demo1_input"] == (1, 2)
    assert by_name["demo1_arrow"] == (2, 3)
    assert by_name["demo1_output"] == (3, 3)

    # v1 refuses the same tokenization outright (ids encode the whole pair,
    # so a prefix ending mid-pair tokenizes differently than the full string)
    def pair_tokens(text: str) -> list[int]:
        chunks = [text[i : i + 2] for i in range(0, len(text), 2)]
        return [0] + [
            ord(chunk[0]) * 256 + (ord(chunk[1]) if len(chunk) > 1 else 0)
            for chunk in chunks
        ]

    with pytest.raises(tg.GroupingError):
        tg.token_group_spans(pair_tokens, parts, prompt)


def test_offset_spans_malformed_offsets_raise():
    record = make_record([("1", "3")], ("5", "7"))
    parts = tg.prompt_parts(record, ARROW, n_shot=1)
    prompt = record["prompt"]
    good = _char_offsets(prompt)
    with pytest.raises(tg.GroupingError, match="no offsets"):
        tg.token_group_spans_offsets(parts, prompt, [], n_bos=1)
    with pytest.raises(tg.GroupingError, match="not monotone"):
        tg.token_group_spans_offsets(parts, prompt, list(reversed(good)), n_bos=1)
    with pytest.raises(tg.GroupingError, match="cover"):
        tg.token_group_spans_offsets(parts, prompt, good[:-1], n_bos=1)
    with pytest.raises(tg.GroupingError, match="degenerate"):
        tg.token_group_spans_offsets(parts, prompt, [(0, 0)] + good[1:], n_bos=1)


# --------------------------------------------------------------------------- #
# proportions + statistics                                                     #
# --------------------------------------------------------------------------- #


def test_span_sums_partition_shares():
    record = make_record([("1", "3"), ("2", "4")], ("5", "7"))
    parts = tg.prompt_parts(record, ARROW, n_shot=2)
    spans = tg.token_group_spans(char_tokens, parts, record["prompt"])
    seq_len = len(char_tokens(record["prompt"]))
    rng = np.random.default_rng(7)
    contribs = rng.normal(0.2, 0.1, size=seq_len)
    shares = contribs / contribs.sum()
    group_shares = tg.span_sums(shares, spans)
    assert group_shares.shape == (len(spans),)
    assert group_shares.sum() == pytest.approx(1.0, abs=1e-12)


def test_summarize_groups_exact_stats():
    shares = np.array([[0.5, 0.5], [0.25, 0.75]])
    stats = tg.summarize_groups(shares, ["a", "b"], n_boot=100, seed=0)
    assert stats["a"]["mean"] == pytest.approx(0.375)
    assert stats["a"]["var"] == pytest.approx(np.array([0.5, 0.25]).var(ddof=1))
    assert stats["a"]["min"] == 0.25
    assert stats["a"]["max"] == 0.5
    assert stats["a"]["n"] == 2
    lo, hi = stats["b"]["ci"]
    assert lo <= stats["b"]["mean"] <= hi


def test_summarize_groups_empty_gives_null_cells():
    stats = tg.summarize_groups(np.empty((0, 2)), ["a", "b"], n_boot=10, seed=0)
    assert stats["a"] == {
        "mean": None,
        "var": None,
        "min": None,
        "max": None,
        "ci": None,
        "n": 0,
    }


def test_group_role_and_display_labels():
    assert tg.group_role("demo3_output") == "output"
    assert tg.group_role("query_arrow") == "arrow"
    assert tg.group_role("bos") == "bos"
    assert tg.group_display_label("demo3_output") == "out3"
    assert tg.group_display_label("query_input") == "in_q"


def test_identity_excludes_presentation_options():
    base = dict(
        heads=[(1, 1)],
        inputs={"z_cache": {"content_fingerprint": "f" * 16}},
        task={"family": "number_add", "prompt_format": "arrow", "n_shot": 2},
        model={"name": "test-model", "revision": None, "dtype": "bfloat16"},
    )
    from subspaces.artifacts import identity_of

    one = tg.build_identity(cfg=Step3Config(interval="var"), **base)
    two = tg.build_identity(cfg=Step3Config(interval="minmax"), **base)
    forked = tg.build_identity(cfg=Step3Config(n_prompts_per_task=7), **base)
    assert identity_of(one) == identity_of(two)
    assert identity_of(one) != identity_of(forked)


# --------------------------------------------------------------------------- #
# end-to-end over the fake repo (model boundary stubbed)                       #
# --------------------------------------------------------------------------- #

# non-square (n_layers=4, n_heads=6) so a layer/head index transposition in
# the z_results lookup cannot hide behind symmetric shapes
HEAD = (1, 2)
TASKS = ("number-add1", "number-add2")


class FakeModel:
    class cfg:
        device = "cpu"

    def to_tokens(self, text, prepend_bos=True):
        import torch

        return torch.tensor([char_tokens(text)])

    def tokenizer(self, text, return_offsets_mapping=True, add_special_tokens=False):
        """Char-level HF-tokenizer stand-in (ids match to_tokens minus BOS)."""
        return {
            "input_ids": [ord(ch) for ch in text],
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


def _fake_capture(model, prompt, layers, *, with_z):
    return {}, len(char_tokens(prompt))


def _fake_contributions(cache, model, layer_idx, head_idx, final_pos, direction):
    seq_len = final_pos + 1
    rng = np.random.default_rng(seq_len * 1009 + layer_idx * 31 + head_idx)
    return rng.normal(0.2, 0.05, size=seq_len)


@pytest.fixture()
def step3_stubs(monkeypatch):
    calls = {"model_loads": 0, "direction_sums": []}

    def fake_load_model(*_args, **_kwargs):
        calls["model_loads"] += 1
        return FakeModel()

    def recording_contributions(
        cache, model, layer_idx, head_idx, final_pos, direction
    ):
        calls["direction_sums"].append(float(direction.sum()))
        return _fake_contributions(
            cache, model, layer_idx, head_idx, final_pos, direction
        )

    monkeypatch.setattr(tg, "_load_model_pinned", fake_load_model)
    monkeypatch.setattr(tg, "_capture", _fake_capture)
    monkeypatch.setattr(tg, "head_source_contributions", recording_contributions)
    monkeypatch.setattr(tg, "reconstruction_cosine", lambda *args, **kwargs: 1.0)
    return calls


@pytest.fixture()
def analysis_repo(fake_repo):
    """fake_repo + a samples manifest, z cache, heads artifact, context YAML."""
    import torch

    from subspaces.step1.zcache import HOOK_SITE, store_zcache

    paths = ProjectPaths.from_root(fake_repo)

    records = {
        task_id: [
            # varying digit widths -> varying prompt lengths, so the stubbed
            # per-length contributions differ across prompts
            make_record(
                [(str(j + 1) * (j + 1), str(j + 3)), ("2", str(2 * j + 4))],
                (str(90 + j), str(92 + j)),
            )
            for j in range(3)
        ]
        for task_id in TASKS
    }
    samples = make_manifest(
        kind="samples",
        schema_version=1,
        paths=paths,
        config={
            "sample_kind": "analysis",
            "spec": {"examples_per_task": 3, "seed": 42, "task_set": "train"},
            "task": {
                "dataset_dir": "number_add",
                "prompt_format": "arrow",
                "n_shot": 2,
            },
        },
        payload={
            "impl": {"module": "subspaces.step1.samples", "algorithm_version": 2},
            "prompt_format": dict(ARROW),
            "task_order": list(TASKS),
            "tasks": {t: {"n_examples": 3, "samples": records[t]} for t in TASKS},
            "counts": {t: 3 for t in TASKS},
        },
    )
    samples_dir = fake_repo / "log" / "cache" / "samples" / "feedface00000000"
    samples_dir.mkdir(parents=True)
    samples_path = samples_dir / "analysis.json"
    write_json_atomic(samples_path, samples)

    z_identity = {
        "model": {"name": "test-model", "revision": None, "dtype": "bfloat16"},
        "dataset_fingerprint": "d" * 64,
        "samples": "s" * 64,
        "hook_site": HOOK_SITE,
        "impl": {"module": "subspaces.step1.zcache", "algorithm_version": 1},
    }
    z_results = {}
    for task_id in TASKS:
        seed = sum(ord(c) for c in task_id)
        z_results[task_id] = torch.randn(
            4, 6, 8, generator=torch.Generator().manual_seed(seed)
        )
    fingerprint = store_zcache(paths, z_identity, z_results)
    z_dir = f"log/cache/z/{fingerprint}"

    node_dir = fake_repo / "log" / "runs" / "toy__abc" / "main-largest_gap-v1-cafe0123"
    node_dir.mkdir(parents=True)
    heads = make_manifest(
        kind="heads",
        schema_version=1,
        paths=paths,
        payload={
            "significant_heads": [[1, 2], [2, 3]],
            "main_heads": [list(HEAD)],
            "minor_heads": [],
            "selector": {"name": "largest_gap", "version": 1, "params": {}},
            "model_dims": {"n_layers": 4, "n_heads": 6},
        },
    )
    heads_path = node_dir / "heads.json"
    write_json_atomic(heads_path, heads)

    context = {
        "schema_version": 1,
        "model": {"name": "test-model"},
        "task": {"family": "number_add", "prompt_format": "arrow", "n_shot": 2},
        "samples": {"analysis": {"path": str(samples_path.relative_to(fake_repo))}},
        "activations": {"path": z_dir},
        "heads_artifact": str(heads_path.relative_to(fake_repo)),
    }
    context_path = fake_repo / "configs" / "step3_context.yaml"
    context_path.write_text(yaml.safe_dump(context), encoding="utf-8")
    return {
        "root": fake_repo,
        "context": context_path,
        "node_dir": node_dir,
        "heads_path": heads_path,
        "z_path": fake_repo / z_dir / "z_results.pth",
    }


def _run_cli(analysis_repo, *extra):
    return step3_cli.main(
        [
            "--root",
            str(analysis_repo["root"]),
            "--context",
            str(analysis_repo["context"]),
            "--n-boot",
            "50",
            *extra,
        ]
    )


def test_cli_end_to_end_writes_artifact(analysis_repo, step3_stubs):
    assert _run_cli(analysis_repo) == 0
    node_dir = analysis_repo["node_dir"]
    manifests = sorted(node_dir.glob("step3-tokengroups-*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["kind"] == "step3_token_groups"
    assert "NaN" not in manifests[0].read_text(encoding="utf-8")
    assert manifest["group_order"][0] == "bos"
    assert manifest["group_order"][-2:] == ["query_input", "query_arrow"]
    record = manifest["per_head"]["(1,2)"]
    assert record["n_prompts"] == 6  # 2 tasks x 3 prompts
    assert record["reconstruction_cosine_min"] == 1.0
    assert manifest["n_tasks_gated"] == len(TASKS)
    assert manifest["n_prompts_grouped_per_task"] == {t: 3 for t in TASKS}
    for cell in record["groups"].values():
        assert set(cell) == {"mean", "var", "min", "max", "ci", "n"}
        assert cell["n"] == 6
        assert cell["min"] <= cell["mean"] + 1e-12
        assert cell["mean"] <= cell["max"] + 1e-12
    # npz sidecar verifies, rows sum to 1
    tg._verify_npz(node_dir, manifest)
    arrays = np.load(node_dir / manifest["outcomes"]["file"])
    shares = arrays["L1H2_shares"]
    assert shares.shape == (6, len(manifest["group_order"]))
    np.testing.assert_allclose(shares.sum(axis=1), 1.0, atol=1e-9)
    # per-task means recorded for both tasks
    assert set(record["per_task_mean"]) == set(TASKS)
    # the direction handed to the decomposition is exactly z[task][layer, head]
    # (non-square layers/heads, so a transposed lookup would show up here)
    import torch

    z_results = torch.load(analysis_repo["z_path"], map_location="cpu")
    expected_sums = [
        float(z_results[t][HEAD[0], HEAD[1]].to(torch.float32).sum()) for t in TASKS
    ]
    for observed in set(step3_stubs["direction_sums"]):
        assert any(
            observed == pytest.approx(expected, rel=1e-6) for expected in expected_sums
        )
    # plots rendered
    stem = manifests[0].name.removesuffix(".json")
    plots = list((node_dir / f"{stem}-plots").glob("token_groups_L1H2_var.png"))
    assert len(plots) == 1


def test_cli_reuse_skips_compute_and_renders_new_interval(analysis_repo, step3_stubs):
    assert _run_cli(analysis_repo) == 0
    assert step3_stubs["model_loads"] == 1
    assert _run_cli(analysis_repo, "--interval", "std") == 0
    assert step3_stubs["model_loads"] == 1  # reused: no second model load
    node_dir = analysis_repo["node_dir"]
    assert len(list(node_dir.glob("step3-tokengroups-*.json"))) == 1
    stem = next(node_dir.glob("step3-tokengroups-*.json")).name.removesuffix(".json")
    assert (node_dir / f"{stem}-plots" / "token_groups_L1H2_std.png").is_file()


def test_cli_identity_fork_creates_sibling(analysis_repo, step3_stubs):
    assert _run_cli(analysis_repo) == 0
    assert _run_cli(analysis_repo, "--n-prompts-per-task", "2") == 0
    manifests = list(analysis_repo["node_dir"].glob("step3-tokengroups-*.json"))
    assert len(manifests) == 2


def test_corrupt_npz_refuses_reuse(analysis_repo, step3_stubs):
    assert _run_cli(analysis_repo) == 0
    node_dir = analysis_repo["node_dir"]
    manifest_path = next(node_dir.glob("step3-tokengroups-*.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    npz_path = node_dir / manifest["outcomes"]["file"]
    arrays = dict(np.load(npz_path))
    arrays["L1H2_shares"] = arrays["L1H2_shares"] + 1.0
    np.savez(npz_path, **arrays)
    assert _run_cli(analysis_repo) == 2  # ArtifactError -> rc 2


def test_gate_failure_writes_no_artifact(analysis_repo, step3_stubs, monkeypatch):
    monkeypatch.setattr(tg, "reconstruction_cosine", lambda *a, **k: 0.5)
    before = tree_snapshot(analysis_repo["root"])
    assert _run_cli(analysis_repo) == 2
    after = tree_snapshot(analysis_repo["root"])
    assert not [p for p in after - before if "step3-tokengroups" in p]


def test_explicit_heads_require_out_dir(analysis_repo, step3_stubs, tmp_path):
    rc = step3_cli.main(
        [
            "--root",
            str(analysis_repo["root"]),
            "--context",
            str(analysis_repo["context"]),
            "--heads",
            "1:1",
        ]
    )
    assert rc == 2
    out_dir = tmp_path / "explicit"
    rc = step3_cli.main(
        [
            "--root",
            str(analysis_repo["root"]),
            "--context",
            str(analysis_repo["context"]),
            "--heads",
            "1:1",
            "--out-dir",
            str(out_dir),
            "--n-boot",
            "50",
        ]
    )
    assert rc == 0
    manifest_path = next(out_dir.glob("step3-tokengroups-*.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["head_set_provenance"] == {"source": "explicit"}
    assert "heads" not in manifest["inputs"]


def test_missing_samples_entry_fails_clearly(analysis_repo, step3_stubs, capsys):
    rc = _run_cli(analysis_repo, "--samples-name", "nope")
    assert rc == 2
    assert "context.samples.nope" in capsys.readouterr().err


def test_model_mismatch_refuses(analysis_repo, step3_stubs):
    context = yaml.safe_load(analysis_repo["context"].read_text(encoding="utf-8"))
    context["model"]["name"] = "other-model"
    analysis_repo["context"].write_text(yaml.safe_dump(context), encoding="utf-8")
    assert _run_cli(analysis_repo) == 2


def test_identity_tamper_refuses_reuse(analysis_repo, step3_stubs):
    assert _run_cli(analysis_repo) == 0
    manifest_path = next(analysis_repo["node_dir"].glob("step3-tokengroups-*.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"]["seed"] = 999
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    assert _run_cli(analysis_repo) == 2  # reuse_or_refuse -> ArtifactError


def test_empty_head_set_refuses(analysis_repo, step3_stubs, capsys):
    # the fixture's heads artifact has an empty minor tier (like the real
    # largest_gap nodes) — a full GPU run over zero heads must refuse
    rc = _run_cli(analysis_repo, "--head-set", "minor")
    assert rc == 2
    assert "empty" in capsys.readouterr().err.lower()
    assert not list(analysis_repo["node_dir"].glob("step3-tokengroups-*"))


def test_frozen_run_default_out_dir_refuses(analysis_repo, step3_stubs, capsys):
    import shutil

    frozen = (
        analysis_repo["root"] / "log" / "runs" / "old_flat__cafe" / "step1" / "heads"
    )
    frozen.mkdir(parents=True)
    shutil.copy(analysis_repo["heads_path"], frozen / "heads.json")
    rc = _run_cli(
        analysis_repo,
        "--heads-artifact",
        str(frozen / "heads.json"),
    )
    assert rc == 2
    assert "frozen" in capsys.readouterr().err.lower()
    assert not list(frozen.glob("step3-tokengroups-*"))
    # an explicit --out-dir unlocks the same artifact elsewhere
    out_dir = analysis_repo["root"] / "log" / "runs" / "gap-backfill"
    rc = _run_cli(
        analysis_repo,
        "--heads-artifact",
        str(frozen / "heads.json"),
        "--out-dir",
        str(out_dir),
    )
    assert rc == 0
    assert list(out_dir.glob("step3-tokengroups-*.json"))


def test_context_file_errors_map_to_rc2(analysis_repo, step3_stubs, capsys):
    rc = step3_cli.main(
        ["--root", str(analysis_repo["root"]), "--context", "no_such_context.yaml"]
    )
    assert rc == 2
    assert "cannot read context" in capsys.readouterr().err
    bad = analysis_repo["root"] / "configs" / "bad_context.yaml"
    bad.write_text("model: {name: [unclosed", encoding="utf-8")
    rc = step3_cli.main(["--root", str(analysis_repo["root"]), "--context", str(bad)])
    assert rc == 2
    assert "invalid context YAML" in capsys.readouterr().err


def test_blank_heads_flag_falls_back_to_artifact(analysis_repo, step3_stubs):
    # a shell wrapper passing --heads "" must behave exactly like omitting it:
    # heads resolve from the artifact and lineage/provenance stay honest
    assert _run_cli(analysis_repo, "--heads", "") == 0
    manifest_path = next(analysis_repo["node_dir"].glob("step3-tokengroups-*.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["head_set_provenance"]["source"] == "heads_artifact"
    assert "heads" in manifest["inputs"]


def test_selector_node_main_heads_artifact_accepted(analysis_repo, step3_stubs):
    """A selector node's main_heads.json (e.g. a recpos node) works directly:
    heads resolve, and the selector provenance comes from the top-level
    selector_name/selector_version/params fields of that kind."""
    root = analysis_repo["root"]
    paths = ProjectPaths.from_root(root)
    recpos_dir = root / "log" / "runs" / "toy__abc" / "recpos-paired_bh-v1-beef0123"
    recpos_dir.mkdir(parents=True)
    manifest = make_manifest(
        kind="main_heads",
        schema_version=1,
        paths=paths,
        payload={
            "impl": {"module": "subspaces.step1.selectors", "algorithm_version": 1},
            "selector_name": "paired_bh",
            "selector_version": 1,
            "params": {"q": 0.05},
            "main_heads": [list(HEAD)],
            "minor_heads": [],
            "decisions": {},
            "verdict": {},
        },
    )
    write_json_atomic(recpos_dir / "main_heads.json", manifest)

    # the wave pattern: the context pins the SAME artifact the CLI passes
    context = yaml.safe_load(analysis_repo["context"].read_text(encoding="utf-8"))
    context["heads_artifact"] = str((recpos_dir / "main_heads.json").relative_to(root))
    recpos_context = root / "configs" / "step3_context_recpos.yaml"
    recpos_context.write_text(yaml.safe_dump(context), encoding="utf-8")

    assert (
        step3_cli.main(
            [
                "--root",
                str(root),
                "--context",
                str(recpos_context),
                "--n-boot",
                "50",
                "--heads-artifact",
                str((recpos_dir / "main_heads.json").relative_to(root)),
                "--head-set",
                "main",
            ]
        )
        == 0
    )
    artifact_path = next(recpos_dir.glob("step3-tokengroups-*.json"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["head_set_provenance"]["selector"] == {
        "name": "paired_bh",
        "version": 1,
        "params": {"q": 0.05},
    }
    assert artifact["inputs"]["heads"]["semantic_fingerprint"]


def test_cli_heads_artifact_must_match_context_heads(
    analysis_repo, step3_stubs, capsys
):
    """A CLI --heads-artifact that disagrees in CONTENT with the context's
    pinned heads_artifact is refused (lineage honesty), while a same-content
    alias passes (covered by the recpos test above)."""
    root = analysis_repo["root"]
    paths = ProjectPaths.from_root(root)
    other_dir = root / "log" / "runs" / "toy__abc" / "main-other-v1-0dd0"
    other_dir.mkdir(parents=True)
    other = make_manifest(
        kind="main_heads",
        schema_version=1,
        paths=paths,
        payload={
            "impl": {"module": "subspaces.step1.selectors", "algorithm_version": 1},
            "selector_name": "unified",
            "selector_version": 1,
            "params": {},
            "main_heads": [[2, 3]],
            "minor_heads": [],
            "decisions": {},
            "verdict": {},
        },
    )
    write_json_atomic(other_dir / "main_heads.json", other)

    rc = _run_cli(
        analysis_repo,
        "--heads-artifact",
        str((other_dir / "main_heads.json").relative_to(root)),
        "--head-set",
        "main",
    )
    assert rc == 2
    assert "refusing mismatched head sources" in capsys.readouterr().err


def test_multi_ref_activations_context_refused(analysis_repo, step3_stubs, capsys):
    """step3 directions come from ONE z cache; the named train/heldout map
    used by step-2 contexts must be refused, not silently picked from."""
    context = yaml.safe_load(analysis_repo["context"].read_text(encoding="utf-8"))
    single_ref = context["activations"]
    context["activations"] = {
        "train": dict(single_ref),
        "heldout": dict(single_ref),
    }
    multi_path = analysis_repo["root"] / "configs" / "step3_context_multi.yaml"
    multi_path.write_text(yaml.safe_dump(context), encoding="utf-8")

    rc = step3_cli.main(
        [
            "--root",
            str(analysis_repo["root"]),
            "--context",
            str(multi_path),
            "--n-boot",
            "50",
        ]
    )
    assert rc == 2
    assert "exactly ONE activations" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# real model-facing math (fake tensors, no model load — closes the stub gap)   #
# --------------------------------------------------------------------------- #


def test_head_source_contributions_real_math_gqa():
    from types import SimpleNamespace

    import torch

    torch.manual_seed(0)
    n_heads, n_kv, d_head, d_model, seq = 4, 2, 3, 5, 7
    w_o_all = torch.randn(n_heads, d_head, d_model)
    model = SimpleNamespace(
        cfg=SimpleNamespace(n_heads=n_heads, device="cpu"),
        blocks=[SimpleNamespace(attn=SimpleNamespace(W_O=w_o_all))],
    )
    pattern = torch.rand(1, n_heads, seq, seq)
    pattern = pattern / pattern.sum(dim=-1, keepdim=True)
    v_all = torch.randn(1, seq, n_kv, d_head)
    cache = {(0, "pattern"): pattern, (0, "v"): v_all}
    direction = torch.randn(d_model)
    head_idx = 3  # GQA: query head 3 reads kv head 3 // (4//2) == 1
    final_pos = seq - 1

    contribs = tg.head_source_contributions(
        cache, model, 0, head_idx, final_pos, direction
    )
    alpha = pattern[0, head_idx, final_pos]
    reference = np.array(
        [
            float(alpha[t] * ((v_all[0, t, 1] @ w_o_all[head_idx]) @ direction))
            for t in range(seq)
        ]
    )
    np.testing.assert_allclose(contribs, reference, rtol=1e-5, atol=1e-7)
    # linearity: the contributions sum to <h(p), direction>
    h_p = (alpha[:, None] * (v_all[0, :, 1, :] @ w_o_all[head_idx])).sum(dim=0)
    assert contribs.sum() == pytest.approx(float(h_p @ direction), rel=1e-5)
    # reconstruction gate: hand-assembled hook_z (per QUERY head) gives cos ~ 1
    z_all = torch.zeros(1, seq, n_heads, d_head)
    z_all[0, final_pos, head_idx] = (alpha[:, None] * v_all[0, :, 1, :]).sum(dim=0)
    cache[(0, "z")] = z_all
    cosine = tg.reconstruction_cosine(cache, model, 0, head_idx, final_pos)
    assert cosine > 0.9999
    # and the gate DOES catch a wrong decomposition: corrupt the stored z
    z_all[0, final_pos, head_idx] = torch.randn(d_head)
    assert tg.reconstruction_cosine(cache, model, 0, head_idx, final_pos) < 0.999
