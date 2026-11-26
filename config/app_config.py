"""Runtime configuration for AI Training Studio."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from config.defaults import (
    DEFAULT_MONITOR_REFRESH_SECONDS,
    MAX_ARCHIVE_FILES,
    MAX_EXTRACTED_SIZE_MB,
    MAX_UPLOAD_ARCHIVE_MB,
)


@dataclass
class AppConfig:
    """Centralized application configuration with one configurable workspace root."""

    project_root: Path
    workspace_root: Path
    max_upload_mb: int = MAX_UPLOAD_ARCHIVE_MB
    max_extracted_mb: int = MAX_EXTRACTED_SIZE_MB
    max_archive_files: int = MAX_ARCHIVE_FILES
    monitor_refresh_seconds: float = DEFAULT_MONITOR_REFRESH_SECONDS
    is_colab: bool = False

    @classmethod
    def from_env(cls, workspace_override: str | Path | None = None) -> "AppConfig":
        project_root = Path(__file__).resolve().parent.parent
        env_workspace = os.environ.get("AI_STUDIO_WORKSPACE")
        if workspace_override is not None:
            workspace_root = Path(workspace_override).expanduser().resolve()
        elif env_workspace:
            workspace_root = Path(env_workspace).expanduser().resolve()
        else:
            workspace_root = (project_root / "workspace").resolve()

        is_colab = "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ or Path("/content").exists()
        return cls(
            project_root=project_root,
            workspace_root=workspace_root,
            is_colab=is_colab,
        )
