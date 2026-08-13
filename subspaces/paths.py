"""Project directory layout.

``ProjectPaths`` is constructed at the CLI boundary (``subspaces/runners/*``) and passed
into library functions explicitly. Library code must not resolve repository paths
at import time.

The logical names map onto the CURRENT physical layout (``dataset_files/``,
``log/``, ...). The planned physical renames (``dataset_files`` -> ``dataset``,
etc.) are one-line changes here and nowhere else in new code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class RepoRootNotFoundError(RuntimeError):
    pass


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` (default: this file) to the pyproject.toml dir."""
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RepoRootNotFoundError(f"no pyproject.toml above {here}")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def from_root(cls, root: str | Path | None = None) -> ProjectPaths:
        return cls(root=Path(root).resolve() if root is not None else find_repo_root())

    # -- inputs ------------------------------------------------------------
    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset_files"

    def task_dir(self, name: str) -> Path:
        return self.dataset_dir / name

    @property
    def configs_dir(self) -> Path:
        return self.root / "configs"

    # -- outputs -----------------------------------------------------------
    @property
    def log_dir(self) -> Path:
        return self.root / "log"

    @property
    def runs_dir(self) -> Path:
        return self.log_dir / "runs"

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    @property
    def journal_dir_root(self) -> Path:
        return self.log_dir / "journal"

    def journal_dir(self, run_id: str) -> Path:
        return self.journal_dir_root / run_id

    @property
    def cache_dir(self) -> Path:
        return self.log_dir / "cache"

    @property
    def z_cache_dir(self) -> Path:
        return self.cache_dir / "z"

    @property
    def samples_cache_dir(self) -> Path:
        return self.cache_dir / "samples"

    @property
    def reference_matrices_dir(self) -> Path:
        return self.root / "model" / "reference_matrices"

    @property
    def doc_dir(self) -> Path:
        return self.root / "doc"

    # -- helpers -----------------------------------------------------------
    def resolve(self, path: str | Path) -> Path:
        """Absolute paths pass through; relative paths resolve against the root.

        Matrices and other artifacts outside the repository are supported by
        passing absolute paths.
        """
        p = Path(path)
        return p if p.is_absolute() else (self.root / p)

    def relativize(self, path: str | Path) -> str:
        """Repo-relative string when possible, else the absolute path string."""
        p = Path(path).resolve()
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)
