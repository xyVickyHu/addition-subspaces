"""Framework modules and their CLIs must never import the heavy legacy stack
(torch / transformers / transformer_lens); the legacy ``from subspaces import X``
contract keeps working via lazy re-export.

Subprocesses are used so each check sees a clean ``sys.modules``.
"""

from __future__ import annotations

import os
import subprocess
import sys

HEAVY = ("torch", "transformers", "transformer_lens")
_HEAVY_CHECK = (
    f"heavy = [m for m in {HEAVY!r} if m in sys.modules]; "
    "assert not heavy, f'heavy imports leaked: {heavy}'"
)


def _run(code: str) -> None:
    # The pytest process's mkl-service exports MKL_THREADING_LAYER=INTEL, which
    # breaks torch(libgomp)+numpy(MKL) coexistence in children; pin it to GNU.
    env = {**os.environ, "MKL_THREADING_LAYER": "GNU"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


def test_framework_imports_stay_light():
    _run(
        "import sys; "
        "import subspaces, subspaces.paths, subspaces.config, subspaces.artifacts, subspaces.head_sets, "
        "subspaces.step1.split, subspaces.step1.pipeline, subspaces.step1.selectors, "
        "subspaces.step2.api, subspaces.step3.api; " + _HEAVY_CHECK
    )


def test_cli_dry_run_stays_light():
    _run(
        "import sys; "
        "from subspaces.runners import step2, step3; "
        "rc2 = step2.main(['--heads', '15:2', '--dry-run']); "
        "rc3 = step3.main(['--heads', '15:2', '--dry-run']); "
        "assert rc2 == 0 and rc3 == 0; " + _HEAVY_CHECK
    )


def test_select_main_cli_stays_light(tmp_path):
    """select-main is pure CPU: the full CLI path (load scan, run selector,
    write the artifact) must never import the heavy stack."""
    scan_path = tmp_path / "head_scan.json"
    code = (
        "import sys, json; "
        "from subspaces.artifacts import make_manifest, write_json_atomic; "
        "from subspaces.paths import ProjectPaths; "
        "paths = ProjectPaths.from_root(); "
        "scan = make_manifest(kind='head_scan', schema_version=1, paths=paths, "
        "payload={'curves': {'15:2': {'0': 0.1, '1': 0.6}}, "
        "'baselines': {'clean_acc': 0.9, 'full_significant_acc': 0.8}, "
        "'n_eval_examples_per_head_per_c': 300, 'c_grid': [0, 1]}); "
        f"write_json_atomic({str(scan_path)!r}, scan); "
        "from subspaces.runners import step1; "
        f"rc = step1.main(['select-main', '--scan', {str(scan_path)!r}, "
        "'--selector', 'unified']); "
        "assert rc == 0; " + _HEAVY_CHECK
    )
    _run(code)


def test_legacy_reexport_still_resolves():
    # Accessing a legacy name lazily loads subspaces.utils (heavy stack allowed here).
    _run(
        "import sys; import subspaces; "
        "assert 'subspaces.utils' not in sys.modules; "
        "fn = subspaces.load_task_data; "
        "assert callable(fn); "
        "assert 'subspaces.utils' in sys.modules"
    )


def test_unknown_attribute_raises():
    _run(
        "import subspaces\n"
        "try:\n"
        "    subspaces.definitely_not_a_name\n"
        "except AttributeError as err:\n"
        "    assert 'definitely_not_a_name' in str(err)\n"
        "else:\n"
        "    raise AssertionError('expected AttributeError')\n"
    )
