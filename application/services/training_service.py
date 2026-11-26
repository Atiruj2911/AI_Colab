"""TrainingService: validate, launch, monitor, reconnect, and safely stop background training."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from config.defaults import DEFAULT_MODEL_FAMILY, DEFAULT_TASK, MAX_LOG_LINES_UI
from core.schemas import JobState, TrainRequest
from core.training.metrics import extract_metric_series, load_epoch_metrics
from infrastructure.monitoring.system_monitor import get_gpu_vram_status
from infrastructure.process.job_manager import ACTIVE_STATUSES, JobManager, is_heartbeat_stale, is_pid_alive
from infrastructure.process.state_store import tail_text_file
from models.registry import ModelRegistry
from utils.validation import sanitize_identifier

from application.services.dataset_service import DatasetService
from application.services.session_service import SessionService


class TrainingService:
    """Coordinates dataset validation, model adapter checks, and detached worker execution."""

    def __init__(
        self,
        registry: ModelRegistry,
        job_manager: JobManager,
        dataset_service: DatasetService,
        session_service: SessionService,
    ) -> None:
        self.registry = registry
        self.job_manager = job_manager
        self.dataset_service = dataset_service
        self.session_service = session_service

    def start_training(
        self,
        dataset_id: str,
        model_variant: str,
        pretrained_weights: str,
        epochs: int,
        batch_size: int,
        imgsz: int,
        device: str = "auto",
        seed: int = 42,
        patience: int = 50,
        workers: int = 2,
        run_name: str | None = None,
        mode: str = "train",
        optimizer: str = "auto",
        lr0: float | None = None,
        save_best: bool = True,
        save_last: bool = True,
        model_family: str = DEFAULT_MODEL_FAMILY,
        task: str = DEFAULT_TASK,
    ) -> JobState:
        from core.schemas import AutoSavePolicy

        dataset = self.dataset_service.get_dataset(dataset_id)
        if dataset is None:
            raise ValueError("กรุณาอัปโหลดและเลือก Dataset ที่ถูกต้องก่อนเริ่มการเทรน")
        if not dataset.is_valid or not dataset.resolved_yaml_path:
            raise ValueError(
                "Dataset นี้ยังไม่ผ่านการตรวจสอบ (พบ Error) กรุณาแก้ไขในหน้า Dataset ก่อนเริ่มการเทรน"
            )

        session = self.session_service.get_or_create_session()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_run_name = sanitize_identifier(run_name or f"{dataset.dataset_name}_{model_variant}", fallback="run")
        run_id = f"{clean_run_name}_{stamp}"

        extra_params: dict[str, Any] = {}
        if optimizer and optimizer != "auto":
            extra_params["optimizer"] = optimizer
        if lr0 is not None and float(lr0) > 0:
            extra_params["lr0"] = float(lr0)

        request = TrainRequest(
            session_id=session["session_id"],
            run_id=run_id,
            run_name=clean_run_name,
            dataset_id=dataset.dataset_id,
            dataset_yaml=dataset.resolved_yaml_path,
            model_family=model_family,
            task=task,
            model_variant=model_variant,
            pretrained_weights=pretrained_weights,
            mode=mode,
            epochs=int(epochs),
            batch_size=int(batch_size),
            imgsz=int(imgsz),
            device=device,
            seed=int(seed),
            patience=int(patience),
            workers=int(workers),
            auto_save=AutoSavePolicy(enabled=True, save_best=bool(save_best), save_last=bool(save_last)),
            extra_params=extra_params,
        )

        adapter = self.registry.get(model_family, task)
        errors = adapter.validate_train_request(request)
        if errors:
            raise ValueError("การตั้งค่าไม่ถูกต้อง: " + "; ".join(errors))

        job = self.job_manager.start_job(request)
        self.session_service.update_session(
            active_dataset_id=dataset.dataset_id,
            active_run_id=job.run_id,
            last_config=request.to_dict(),
        )
        return job

    def stop_training(self, run_id: str | None = None) -> JobState | None:
        return self.job_manager.request_stop(run_id=run_id, force=False)

    def get_current_job(self, run_id: str | None = None) -> JobState | None:
        if run_id:
            return self.job_manager.get_job_by_run_id(run_id)
        active = self.job_manager.get_active_job()
        if active is not None:
            return active
        all_jobs = self.job_manager.list_all_jobs()
        return all_jobs[0] if all_jobs else None

    def list_runs(self) -> list[JobState]:
        return self.job_manager.list_all_jobs()

    def get_monitor_snapshot(self, run_id: str | None = None) -> dict[str, Any]:
        """Read persisted job state, metrics history, logs, and live GPU status (safe after reconnect)."""
        job = self.get_current_job(run_id=run_id)
        gpu_status = get_gpu_vram_status()
        if job is None:
            return {
                "job": None,
                "alive": False,
                "heartbeat_stale": False,
                "metrics_records": [],
                "series": extract_metric_series([]),
                "log_tail": "",
                "gpu": gpu_status,
            }

        alive = job.status in ACTIVE_STATUSES and is_pid_alive(job.pid)
        stale = alive and is_heartbeat_stale(job.heartbeat_at)
        records = load_epoch_metrics(job.metrics_path)
        series = extract_metric_series(records)
        log_tail = tail_text_file(Path(job.log_path), max_lines=MAX_LOG_LINES_UI) if job.log_path else ""

        return {
            "job": job,
            "alive": alive,
            "heartbeat_stale": stale,
            "metrics_records": records,
            "series": series,
            "log_tail": log_tail,
            "gpu": gpu_status,
        }
