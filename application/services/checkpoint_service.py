"""CheckpointService: discover, list, and package downloadable checkpoints and run artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from core.schemas import CheckpointArtifact, JobState
from infrastructure.process.job_manager import JobManager
from infrastructure.storage.workspace_manager import WorkspaceManager


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


class CheckpointService:
    """Lists trained weights (best.pt / last.pt) and config/log artifacts for download or inference."""

    def __init__(self, workspace: WorkspaceManager, job_manager: JobManager) -> None:
        self.workspace = workspace
        self.job_manager = job_manager

    def list_model_checkpoints(self) -> list[CheckpointArtifact]:
        """Return all best.pt and last.pt checkpoints across runs, sorted newest first."""
        artifacts: list[CheckpointArtifact] = []
        jobs_by_id: dict[str, JobState] = {j.run_id: j for j in self.job_manager.list_all_jobs()}

        for run_dir in sorted(self.workspace.runs_dir.glob("*"), reverse=True):
            if not run_dir.is_dir():
                continue
            run_id = run_dir.name
            job = jobs_by_id.get(run_id)
            run_name = job.run_name if job else run_id
            best_map = job.best_map50_95 if job else None

            weights_dir = run_dir / "weights"
            fallback_weights_dir = run_dir / "ultralytics" / "weights"

            for kind, fname in (("best", "best.pt"), ("last", "last.pt")):
                candidate = weights_dir / fname
                if not candidate.exists():
                    candidate = fallback_weights_dir / fname
                if candidate.exists() and candidate.is_file():
                    artifacts.append(
                        CheckpointArtifact(
                            name=f"{run_name} / {fname}",
                            kind=kind,
                            path=str(candidate.resolve()),
                            run_id=run_id,
                            run_name=run_name,
                            size_bytes=candidate.stat().st_size,
                            modified_at=_mtime_iso(candidate),
                            best_map50_95=best_map,
                        )
                    )

        artifacts.sort(key=lambda a: (a.modified_at, a.kind == "best"), reverse=True)
        return artifacts

    def list_run_artifacts(self, run_id: str) -> list[CheckpointArtifact]:
        """Return all downloadable files (best.pt, last.pt, config, data.yaml, runtime_meta, log) for one run."""
        run_dir = self.workspace.runs_dir / run_id
        if not run_dir.exists():
            return []

        job = self.job_manager.get_job_by_run_id(run_id)
        run_name = job.run_name if job else run_id
        best_map = job.best_map50_95 if job else None

        candidates = [
            ("best", run_dir / "weights" / "best.pt"),
            ("last", run_dir / "weights" / "last.pt"),
            ("config", run_dir / "train_config.json"),
            ("config", run_dir / "data.yaml"),
            ("metadata", run_dir / "runtime_meta.json"),
            ("log", run_dir / "train.log"),
        ]

        items: list[CheckpointArtifact] = []
        for kind, path in candidates:
            if path.exists() and path.is_file():
                items.append(
                    CheckpointArtifact(
                        name=path.name,
                        kind=kind,
                        path=str(path.resolve()),
                        run_id=run_id,
                        run_name=run_name,
                        size_bytes=path.stat().st_size,
                        modified_at=_mtime_iso(path),
                        best_map50_95=best_map,
                    )
                )
        return items
