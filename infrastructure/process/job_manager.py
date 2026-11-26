"""Disconnect-safe background training job manager with duplicate-start protection."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config.defaults import GRACEFUL_STOP_TIMEOUT_SECONDS, HEARTBEAT_STALE_SECONDS
from core.schemas import JobState, TrainRequest, utc_now_iso
from infrastructure.process.state_store import read_json, write_json_atomic
from infrastructure.storage.workspace_manager import WorkspaceManager

ACTIVE_STATUSES = {"queued", "starting", "running", "stopping"}
TERMINAL_STATUSES = {"completed", "stopped", "failed"}


def is_pid_alive(pid: int | None) -> bool:
    """Check whether an OS process with the given PID is still alive (and not zombie)."""
    if pid is None or pid <= 0:
        return False
    try:
        import psutil

        if not psutil.pid_exists(pid):
            return False
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except ImportError:
        pass
    except Exception:
        return False

    # Fallback without psutil
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def is_heartbeat_stale(heartbeat_iso: str | None, threshold_seconds: float = HEARTBEAT_STALE_SECONDS) -> bool:
    if not heartbeat_iso:
        return True
    try:
        dt = datetime.fromisoformat(heartbeat_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - dt).total_seconds()
        return age > threshold_seconds
    except ValueError:
        return True


class JobManager:
    """Spawns, tracks, reconnects to, and safely stops background training workers."""

    def __init__(self, workspace: WorkspaceManager, project_root: Path) -> None:
        self.workspace = workspace
        self.project_root = Path(project_root).resolve()
        self.active_job_pointer = self.workspace.sessions_dir / "active_job.json"

    def _state_file_for_run(self, run_dir: Path) -> Path:
        return run_dir / "job_state.json"

    def _stop_flag_for_run(self, run_dir: Path) -> Path:
        return run_dir / "stop_requested.flag"

    def list_all_jobs(self) -> list[JobState]:
        jobs: list[JobState] = []
        if not self.workspace.runs_dir.exists():
            return jobs
        for state_file in self.workspace.runs_dir.glob("*/job_state.json"):
            data = read_json(state_file)
            if not data:
                continue
            try:
                job = self.reconcile_job_state(JobState.from_dict(data))
                jobs.append(job)
            except Exception:
                continue
        jobs.sort(key=lambda j: j.started_at, reverse=True)
        return jobs

    def get_job_by_run_id(self, run_id: str) -> JobState | None:
        run_dir = self.workspace.runs_dir / run_id
        data = read_json(self._state_file_for_run(run_dir))
        if not data:
            return None
        return self.reconcile_job_state(JobState.from_dict(data))

    def get_active_job(self) -> JobState | None:
        """Return currently active (or most recently tracked) job, reconciling real PID state."""
        pointer = read_json(self.active_job_pointer)
        if pointer and pointer.get("run_id"):
            job = self.get_job_by_run_id(str(pointer["run_id"]))
            if job is not None:
                return job

        # Fallback: scan runs for any job in active status
        for job in self.list_all_jobs():
            if job.status in ACTIVE_STATUSES:
                return job
        return None

    def has_running_job(self) -> JobState | None:
        job = self.get_active_job()
        if job and job.status in ACTIVE_STATUSES and is_pid_alive(job.pid):
            return job
        return None

    def reconcile_job_state(self, job: JobState) -> JobState:
        """Verify claimed running job against real OS PID so UI never shows stale fake 'running'."""
        if job.status in TERMINAL_STATUSES:
            return job

        # Give newly spawned jobs a short grace period before PID verification marks them dead
        if job.pid is None:
            try:
                started_dt = datetime.fromisoformat(job.started_at)
                if started_dt.tzinfo is None:
                    started_dt = started_dt.replace(tzinfo=timezone.utc)
                if (datetime.now(timezone.utc) - started_dt).total_seconds() < 10:
                    return job
            except ValueError:
                pass

        if job.pid is not None and is_pid_alive(job.pid):
            return job

        # Process has exited without writing terminal state (e.g., OOM kill or crash)
        run_dir = Path(job.run_dir) if job.run_dir else (self.workspace.runs_dir / job.run_id)
        if job.stop_requested or self._stop_flag_for_run(run_dir).exists():
            job.status = "stopped"
            job.error_message = None
        else:
            job.status = "failed"
            if not job.error_message:
                job.error_message = (
                    "Training worker process หยุดทำงานโดยไม่คาดคิด (อาจเกิดจาก VRAM/RAM ไม่พอหรือ Runtime ถูกรีสตาร์ท) "
                    "กรุณาตรวจสอบ Training Log"
                )
        job.finished_at = job.finished_at or utc_now_iso()
        job.updated_at = utc_now_iso()
        if run_dir.exists():
            write_json_atomic(self._state_file_for_run(run_dir), job.to_dict())
        return job

    def start_job(self, request: TrainRequest) -> JobState:
        """Start a detached training subprocess after verifying no duplicate job is running."""
        running = self.has_running_job()
        if running is not None:
            raise RuntimeError(
                f"มีงานเทรนกำลังทำงานอยู่แล้ว (Run: {running.run_name}, PID: {running.pid}) "
                "กรุณารอให้เสร็จหรือกดหยุดเทรนก่อนเริ่มงานใหม่"
            )

        run_dir = (self.workspace.runs_dir / request.run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        request.output_dir = str(run_dir)

        spec_path = run_dir / "job_spec.json"
        state_path = self._state_file_for_run(run_dir)
        log_path = run_dir / "train.log"
        metrics_path = run_dir / "metrics.jsonl"

        write_json_atomic(spec_path, request.to_dict())

        now = utc_now_iso()
        job = JobState(
            job_id=request.run_id,
            session_id=request.session_id,
            run_id=request.run_id,
            run_name=request.run_name,
            dataset_id=request.dataset_id,
            model_family=request.model_family,
            task=request.task,
            model_variant=request.model_variant,
            status="starting",
            pid=None,
            current_epoch=0,
            total_epochs=request.epochs,
            progress_pct=0.0,
            elapsed_seconds=0.0,
            run_dir=str(run_dir),
            log_path=str(log_path),
            metrics_path=str(metrics_path),
            started_at=now,
            updated_at=now,
            heartbeat_at=now,
        )
        write_json_atomic(state_path, job.to_dict())
        write_json_atomic(
            self.active_job_pointer,
            {"run_id": request.run_id, "session_id": request.session_id, "updated_at": now},
        )

        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{self.project_root}{os.pathsep}{existing_pp}" if existing_pp else str(self.project_root)
        )
        env["PYTHONUNBUFFERED"] = "1"
        env["AI_STUDIO_WORKSPACE"] = str(self.workspace.root)

        cmd = [
            sys.executable,
            "-m",
            "core.training.worker",
            "--job-spec",
            str(spec_path),
            "--state-file",
            str(state_path),
            "--workspace",
            str(self.workspace.root),
        ]

        log_handle = log_path.open("a", encoding="utf-8")
        try:
            popen_kwargs: dict = {
                "cwd": str(self.project_root),
                "env": env,
                "stdin": subprocess.DEVNULL,
                "stdout": log_handle,
                "stderr": subprocess.STDOUT,
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs["start_new_session"] = True

            proc = subprocess.Popen(cmd, **popen_kwargs)
        finally:
            log_handle.close()

        job.pid = proc.pid
        job.updated_at = utc_now_iso()
        write_json_atomic(state_path, job.to_dict())
        return job

    def request_stop(self, run_id: str | None = None, force: bool = False) -> JobState | None:
        """Signal the background worker to stop gracefully (or terminate if force=True)."""
        job = self.get_job_by_run_id(run_id) if run_id else self.get_active_job()
        if job is None:
            return None

        if job.status in TERMINAL_STATUSES:
            return job

        run_dir = Path(job.run_dir) if job.run_dir else (self.workspace.runs_dir / job.run_id)
        stop_flag = self._stop_flag_for_run(run_dir)
        stop_flag.write_text(utc_now_iso(), encoding="utf-8")

        job.stop_requested = True
        job.status = "stopping"
        job.updated_at = utc_now_iso()
        write_json_atomic(self._state_file_for_run(run_dir), job.to_dict())

        if force and job.pid and is_pid_alive(job.pid):
            self._terminate_pid(job.pid)
            job.status = "stopped"
            job.finished_at = utc_now_iso()
            job.updated_at = utc_now_iso()
            write_json_atomic(self._state_file_for_run(run_dir), job.to_dict())

        return job

    def wait_for_stop(self, run_id: str, timeout: float = GRACEFUL_STOP_TIMEOUT_SECONDS) -> JobState | None:
        """Wait up to timeout for graceful stop, then force-terminate if still stuck."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.get_job_by_run_id(run_id)
            if job is None or job.status in TERMINAL_STATUSES:
                return job
            time.sleep(0.5)
        return self.request_stop(run_id, force=True)

    def _terminate_pid(self, pid: int) -> None:
        try:
            import psutil

            proc = psutil.Process(pid)
            for child in proc.children(recursive=True):
                child.terminate()
            proc.terminate()
            _, alive = psutil.wait_procs([proc], timeout=5)
            for p in alive:
                p.kill()
            return
        except Exception:
            pass

        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
