"""Detached background training worker process.

Run as:
    python -m core.training.worker --job-spec <path> --state-file <path> --workspace <path>

Runs completely outside Gradio HTTP callbacks so browser disconnects never interrupt training.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

from core.schemas import EpochMetric, JobState, TrainRequest, utc_now_iso
from infrastructure.monitoring.system_monitor import get_package_versions
from infrastructure.process.state_store import append_jsonl, read_json, write_json_atomic
from infrastructure.storage.workspace_manager import WorkspaceManager
from models.registry import build_default_registry


class _StateUpdater:
    """Thread-safe JobState updater shared between training callbacks and heartbeat thread."""

    def __init__(self, state_path: Path, initial_state: JobState) -> None:
        self.state_path = state_path
        self.state = initial_state
        self._lock = threading.Lock()

    def update(self, **kwargs) -> JobState:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self.state, k, v)
            now = utc_now_iso()
            self.state.updated_at = now
            self.state.heartbeat_at = now
            write_json_atomic(self.state_path, self.state.to_dict())
            return self.state

    def heartbeat(self) -> None:
        with self._lock:
            now = utc_now_iso()
            self.state.heartbeat_at = now
            self.state.updated_at = now
            write_json_atomic(self.state_path, self.state.to_dict())


def _save_run_metadata(request: TrainRequest, run_dir: Path) -> None:
    """Store training config, dataset YAML copy, and runtime package metadata together in run_dir."""
    write_json_atomic(run_dir / "train_config.json", request.to_dict())
    write_json_atomic(
        run_dir / "runtime_meta.json",
        {
            "recorded_at": utc_now_iso(),
            "pid": os.getpid(),
            "packages": get_package_versions(),
        },
    )
    src_yaml = Path(request.dataset_yaml)
    if src_yaml.exists():
        shutil.copy2(src_yaml, run_dir / "data.yaml")


def run_worker(job_spec_path: Path, state_file_path: Path, workspace_root: Path) -> int:
    spec_data = read_json(job_spec_path)
    if not spec_data:
        print(f"[Worker] ERROR: Missing or unreadable job spec: {job_spec_path}", flush=True)
        return 1

    request = TrainRequest.from_dict(spec_data)
    run_dir = Path(request.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    stop_flag_path = run_dir / "stop_requested.flag"

    existing = read_json(state_file_path)
    if existing:
        job_state = JobState.from_dict(existing)
    else:
        job_state = JobState(
            job_id=request.run_id,
            session_id=request.session_id,
            run_id=request.run_id,
            run_name=request.run_name,
            dataset_id=request.dataset_id,
            model_family=request.model_family,
            task=request.task,
            model_variant=request.model_variant,
            status="starting",
            total_epochs=request.epochs,
            run_dir=str(run_dir),
            log_path=str(run_dir / "train.log"),
            metrics_path=str(metrics_path),
        )

    updater = _StateUpdater(state_file_path, job_state)
    updater.update(pid=os.getpid(), status="running")

    heartbeat_stop = threading.Event()

    def _heartbeat_loop() -> None:
        while not heartbeat_stop.wait(5.0):
            try:
                updater.heartbeat()
            except Exception:
                pass

    hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    hb_thread.start()

    workspace = WorkspaceManager(workspace_root)
    start_ts = time.time()
    best_map50: float | None = None
    best_map50_95: float | None = None
    best_epoch: int | None = None

    def _should_stop() -> bool:
        return stop_flag_path.exists()

    def _on_epoch_end(metric: EpochMetric) -> None:
        nonlocal best_map50, best_map50_95, best_epoch
        append_jsonl(metrics_path, metric.to_dict())

        if metric.map50 is not None:
            best_map50 = metric.map50 if best_map50 is None else max(best_map50, metric.map50)
        if metric.map50_95 is not None:
            if best_map50_95 is None or metric.map50_95 >= best_map50_95:
                best_map50_95 = metric.map50_95
                best_epoch = metric.epoch

        elapsed = max(0.0, time.time() - start_ts)
        progress = min(100.0, (metric.epoch / max(1, request.epochs)) * 100.0)
        remaining_epochs = max(0, request.epochs - metric.epoch)
        sec_per_epoch = elapsed / max(1, metric.epoch)
        eta = sec_per_epoch * remaining_epochs if remaining_epochs > 0 else 0.0

        # Atomically sync live checkpoints into run_dir/weights/ so Testing & Export never see partial writes
        live_weights = run_dir / "ultralytics" / "weights"
        canon_weights = run_dir / "weights"
        canon_weights.mkdir(parents=True, exist_ok=True)
        best_path = job_state.best_checkpoint_path
        last_path = job_state.last_checkpoint_path

        for fname in ("best.pt", "last.pt"):
            src = live_weights / fname
            if src.exists():
                dst = canon_weights / fname
                tmp_dst = canon_weights / f".tmp_{fname}"
                try:
                    shutil.copy2(src, tmp_dst)
                    os.replace(tmp_dst, dst)
                    if fname == "best.pt":
                        best_path = str(dst)
                    else:
                        last_path = str(dst)
                except OSError:
                    tmp_dst.unlink(missing_ok=True)

        updater.update(
            status="stopping" if _should_stop() else "running",
            stop_requested=_should_stop(),
            current_epoch=metric.epoch,
            total_epochs=request.epochs,
            progress_pct=round(progress, 2),
            elapsed_seconds=round(elapsed, 2),
            eta_seconds=round(eta, 2),
            best_map50=best_map50,
            best_map50_95=best_map50_95,
            best_epoch=best_epoch,
            latest_metrics=metric.to_dict(),
            best_checkpoint_path=best_path,
            last_checkpoint_path=last_path,
        )

    def _on_log(msg: str) -> None:
        print(msg, flush=True)

    try:
        _save_run_metadata(request, run_dir)
        registry = build_default_registry()
        adapter = registry.get(request.model_family, request.task)

        validation_errors = adapter.validate_train_request(request)
        if validation_errors:
            raise ValueError("; ".join(validation_errors))

        result = adapter.train(
            request=request,
            weights_cache_dir=workspace.weights_dir,
            on_epoch_end=_on_epoch_end,
            should_stop=_should_stop,
            on_log=_on_log,
        )

        elapsed = max(0.0, time.time() - start_ts)
        was_stopped = bool(result.get("stopped_by_user")) or _should_stop()
        final_status = "stopped" if was_stopped else "completed"
        final_progress = updater.state.progress_pct if was_stopped else 100.0

        updater.update(
            status=final_status,
            stop_requested=was_stopped,
            progress_pct=final_progress,
            elapsed_seconds=round(elapsed, 2),
            eta_seconds=0.0,
            best_checkpoint_path=result.get("best_checkpoint_path") or updater.state.best_checkpoint_path,
            last_checkpoint_path=result.get("last_checkpoint_path") or updater.state.last_checkpoint_path,
            finished_at=utc_now_iso(),
        )
        print(f"[Worker] Finished with status={final_status}", flush=True)
        return 0
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[Worker] FAILED: {exc}\n{tb}", flush=True)
        updater.update(
            status="failed",
            elapsed_seconds=round(max(0.0, time.time() - start_ts), 2),
            eta_seconds=None,
            error_message=str(exc),
            error_detail=tb,
            finished_at=utc_now_iso(),
        )
        return 1
    finally:
        heartbeat_stop.set()


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Training Studio background worker")
    parser.add_argument("--job-spec", required=True, type=Path)
    parser.add_argument("--state-file", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    args = parser.parse_args()
    code = run_worker(args.job_spec, args.state_file, args.workspace)
    sys.exit(code)


if __name__ == "__main__":
    main()
