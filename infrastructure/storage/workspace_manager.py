"""Single-root workspace manager for all runtime artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path

from utils.validation import is_within_directory


class WorkspaceManager:
    """Manages the unified runtime workspace directory layout."""

    SUBDIRS = (
        "uploads",
        "datasets",
        "weights",
        "runs",
        "testing_snapshots",
        "exports",
        "sessions",
        "temp",
        "logs",
    )

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.ensure_layout()

    def ensure_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in self.SUBDIRS:
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    @property
    def uploads_dir(self) -> Path:
        return self.root / "uploads"

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def weights_dir(self) -> Path:
        return self.root / "weights"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def testing_snapshots_dir(self) -> Path:
        return self.root / "testing_snapshots"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    @property
    def sessions_dir(self) -> Path:
        return self.root / "sessions"

    @property
    def temp_dir(self) -> Path:
        return self.root / "temp"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def directory_size_bytes(self, path: Path) -> int:
        if not path.exists():
            return 0
        total = 0
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total

    def usage_breakdown(self) -> dict[str, int]:
        self.ensure_layout()
        breakdown: dict[str, int] = {}
        for sub in self.SUBDIRS:
            breakdown[sub] = self.directory_size_bytes(self.root / sub)
        breakdown["total"] = sum(breakdown.values())
        return breakdown

    def clear_subdir(self, subdir_name: str) -> int:
        """Delete all contents inside a managed workspace subdirectory and return freed bytes."""
        if subdir_name not in self.SUBDIRS:
            raise ValueError(f"Unsupported workspace subdirectory: {subdir_name}")
        target = (self.root / subdir_name).resolve()
        if not is_within_directory(target, self.root):
            raise ValueError("Attempted to clear directory outside workspace root.")

        freed = self.directory_size_bytes(target)
        for child in target.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except OSError:
                    pass
        target.mkdir(parents=True, exist_ok=True)
        return freed
