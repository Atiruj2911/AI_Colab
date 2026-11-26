"""AppController: single application facade connecting Gradio UI pages to core services."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from config.app_config import AppConfig
from core.export.exporter import ModelExporter
from core.inference.predictor import ImagePredictor
from core.schemas import CheckpointArtifact, DatasetValidationReport, ExportArtifact, InferenceResult, JobState
from infrastructure.monitoring.system_monitor import (
    get_available_devices,
    get_disk_status,
    get_gpu_vram_status,
    get_package_versions,
)
from infrastructure.process.job_manager import JobManager
from infrastructure.storage.workspace_manager import WorkspaceManager
from models.registry import ModelRegistry, build_default_registry
from utils.logging_utils import setup_logger

from application.services.annotation_service import AnnotationService
from application.services.checkpoint_service import CheckpointService
from application.services.dataset_service import DatasetService
from application.services.export_service import ExportService
from application.services.inference_service import InferenceService
from application.services.session_service import SessionService
from application.services.training_service import TrainingService


class AppController:
    """Orchestrates all Phase 1 services and isolates UI callbacks from implementation details."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig.from_env()
        self.workspace = WorkspaceManager(self.config.workspace_root)
        self.logger: logging.Logger = setup_logger(
            "ai_training_studio",
            log_file=self.workspace.logs_dir / "app.log",
        )
        self.registry: ModelRegistry = build_default_registry()
        self.job_manager = JobManager(self.workspace, self.config.project_root)

        self.session_service = SessionService(self.workspace)
        self.dataset_service = DatasetService(self.config, self.workspace)
        self.checkpoint_service = CheckpointService(self.workspace, self.job_manager)
        self.training_service = TrainingService(
            registry=self.registry,
            job_manager=self.job_manager,
            dataset_service=self.dataset_service,
            session_service=self.session_service,
        )
        self.inference_service = InferenceService(
            predictor=ImagePredictor(self.registry, self.workspace.temp_dir / "inference")
        )
        self.export_service = ExportService(
            exporter=ModelExporter(self.registry, self.workspace.exports_dir),
            job_manager=self.job_manager,
        )
        self.annotation_service = AnnotationService(self.workspace, self.dataset_service, self.registry)

        # Ensure session exists immediately
        self.session_service.get_or_create_session()

    def close(self) -> None:
        """Close file logging handlers cleanly (useful in tests and graceful shutdown)."""
        for handler in list(self.logger.handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
                self.logger.removeHandler(handler)

    # ---------- System & Session ----------
    def get_system_overview(self) -> dict[str, Any]:
        return {
            "versions": get_package_versions(),
            "gpu": get_gpu_vram_status(),
            "disk": get_disk_status(self.workspace.root),
            "workspace_root": str(self.workspace.root),
            "workspace_usage": self.workspace.usage_breakdown(),
            "devices": get_available_devices(),
            "is_colab": self.config.is_colab,
            "session": self.session_service.get_or_create_session(),
            "active_job": self.training_service.get_current_job(),
            "dataset_count": len(self.dataset_service.list_datasets()),
            "run_count": len(self.training_service.list_runs()),
            "checkpoint_count": len(self.checkpoint_service.list_model_checkpoints()),
        }

    def clear_workspace_folder(self, subdir_name: str, confirmed: bool) -> tuple[bool, str]:
        if not confirmed:
            return False, "กรุณาติ๊กยืนยันก่อนลบข้อมูลในพื้นที่จัดเก็บ"
        if subdir_name == "runs" and self.job_manager.has_running_job() is not None:
            return False, "ไม่สามารถล้างโฟลเดอร์ runs ระหว่างที่กำลังเทรนอยู่ได้ กรุณาหยุดการเทรนก่อน"
        try:
            freed = self.workspace.clear_subdir(subdir_name)
            from utils.validation import format_bytes

            self.logger.info("Cleared workspace subdir '%s' (freed %s)", subdir_name, format_bytes(freed))
            return True, f"ล้างข้อมูลใน '{subdir_name}/' เรียบร้อยแล้ว (คืนพื้นที่ {format_bytes(freed)})"
        except Exception as exc:
            self.logger.exception("Failed clearing workspace subdir %s", subdir_name)
            return False, f"ไม่สามารถล้างข้อมูลได้: {exc}"

    # ---------- Datasets ----------
    def upload_dataset(
        self,
        uploaded_files: list[str | Path],
        dataset_name: str | None = None,
    ) -> DatasetValidationReport:
        self.logger.info("Ingesting uploaded dataset (files=%d, name=%s)", len(uploaded_files), dataset_name)
        report = self.dataset_service.ingest_upload(uploaded_files, dataset_name=dataset_name)
        self.session_service.update_session(active_dataset_id=report.dataset_id)
        return report

    def create_demo_dataset(self) -> DatasetValidationReport:
        self.logger.info("Generating synthetic demo YOLO dataset")
        report = self.dataset_service.create_demo_dataset()
        self.session_service.update_session(active_dataset_id=report.dataset_id)
        return report

    def list_datasets(self) -> list[DatasetValidationReport]:
        return self.dataset_service.list_datasets()

    def get_dataset(self, dataset_id: str) -> DatasetValidationReport | None:
        return self.dataset_service.get_dataset(dataset_id)

    def get_dataset_previews(self, dataset_id: str, draw_boxes: bool = True) -> list[tuple[str, str]]:
        return self.dataset_service.get_preview_gallery(dataset_id, draw_boxes=draw_boxes)

    # ---------- Training & Monitor ----------
    def start_training(self, **kwargs: Any) -> JobState:
        self.logger.info("Starting training job: %s", {k: v for k, v in kwargs.items()})
        return self.training_service.start_training(**kwargs)

    def stop_training(self, run_id: str | None = None) -> JobState | None:
        self.logger.info("Requesting graceful stop for run_id=%s", run_id)
        return self.training_service.stop_training(run_id=run_id)

    def get_monitor_snapshot(self, run_id: str | None = None) -> dict[str, Any]:
        return self.training_service.get_monitor_snapshot(run_id=run_id)

    def list_runs(self) -> list[JobState]:
        return self.training_service.list_runs()

    # ---------- Checkpoints ----------
    def list_checkpoints(self) -> list[CheckpointArtifact]:
        return self.checkpoint_service.list_model_checkpoints()

    def list_run_artifacts(self, run_id: str) -> list[CheckpointArtifact]:
        return self.checkpoint_service.list_run_artifacts(run_id)

    # ---------- Testing / Inference ----------
    def run_inference(
        self,
        weights_path: str | Path,
        image_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
    ) -> InferenceResult:
        self.logger.info("Running inference weights=%s conf=%.2f iou=%.2f", weights_path, conf, iou)
        return self.inference_service.detect_image(
            weights_path=weights_path,
            image_path=image_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
        )

    # ---------- Export ----------
    def export_model(
        self,
        weights_path: str | Path,
        formats: list[str],
        imgsz: int = 640,
        run_id: str | None = None,
    ) -> tuple[Path, list[ExportArtifact]]:
        self.logger.info("Exporting weights=%s formats=%s", weights_path, formats)
        return self.export_service.export_model(
            weights_path=weights_path,
            formats=formats,
            imgsz=imgsz,
            run_id=run_id,
        )

    def create_testing_snapshot(self, run_id: str | None = None) -> dict[str, Any]:
        """Manually snapshot the latest best.pt for Testing without disturbing the Training Worker."""
        from datetime import datetime
        from infrastructure.storage.checkpoint_snapshot import safe_snapshot_checkpoint

        job = self.training_service.get_current_job(run_id=run_id)
        if job is None:
            raise ValueError("ยังไม่มี Training Run — กรุณาเริ่ม Training หรือเลือก Run ก่อน")

        candidates = []
        if job.best_checkpoint_path:
            candidates.append(Path(job.best_checkpoint_path))
        if job.run_dir:
            candidates.append(Path(job.run_dir) / "weights" / "best.pt")
            candidates.append(Path(job.run_dir) / "ultralytics" / "weights" / "best.pt")

        src = next((p for p in candidates if p.is_file()), None)
        if src is None:
            raise FileNotFoundError(
                "Training ยังไม่มี Best Checkpoint (best.pt) — รอให้จบอย่างน้อย 1 Epoch ก่อนแล้วกด Update Best Model ใหม่"
            )

        snapshot_path = safe_snapshot_checkpoint(
            src_path=src,
            dest_dir=self.workspace.testing_snapshots_dir / job.run_id,
            prefix="best_test",
        )
        return {
            "snapshot_path": str(snapshot_path),
            "run_id": job.run_id,
            "run_name": job.run_name,
            "best_epoch": job.best_epoch or job.current_epoch,
            "best_map50_95": job.best_map50_95,
            "best_map50": job.best_map50,
            "size_bytes": snapshot_path.stat().st_size,
            "updated_at": datetime.now().strftime("%H:%M:%S"),
        }

    def export_pt_checkpoint(
        self,
        run_id: str | None = None,
        which: str = "best",
        include_bundle: bool = False,
    ) -> dict[str, Any]:
        """Safely snapshot best.pt or last.pt for download while Training continues uninterrupted."""
        import zipfile
        from datetime import datetime
        from infrastructure.storage.checkpoint_snapshot import safe_snapshot_checkpoint

        if which not in {"best", "last"}:
            raise ValueError("ระบุชนิด Checkpoint เป็น 'best' หรือ 'last'")
        job = self.training_service.get_current_job(run_id=run_id)
        if job is None:
            raise ValueError("ยังไม่มี Training Run ให้ Export")

        fname = f"{which}.pt"
        candidates = []
        if which == "best" and job.best_checkpoint_path:
            candidates.append(Path(job.best_checkpoint_path))
        if which == "last" and job.last_checkpoint_path:
            candidates.append(Path(job.last_checkpoint_path))
        if job.run_dir:
            candidates.append(Path(job.run_dir) / "weights" / fname)
            candidates.append(Path(job.run_dir) / "ultralytics" / "weights" / fname)

        src = next((p for p in candidates if p.is_file()), None)
        if src is None:
            raise FileNotFoundError(f"ยังไม่พบไฟล์ {fname} ใน Run '{job.run_name}'")

        out_dir = self.workspace.exports_dir / job.run_id
        snap = safe_snapshot_checkpoint(src_path=src, dest_dir=out_dir, prefix=f"{job.run_name}_{which}")

        if include_bundle:
            zip_path = snap.with_suffix(".zip")
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(snap, arcname=fname)
                if job.run_dir:
                    for extra in ("data.yaml", "train_config.json", "runtime_meta.json"):
                        ep = Path(job.run_dir) / extra
                        if ep.is_file():
                            zf.write(ep, arcname=extra)
            out_file = zip_path
        else:
            out_file = snap

        return {
            "file_path": str(out_file),
            "filename": out_file.name,
            "which": which,
            "run_id": job.run_id,
            "size_bytes": out_file.stat().st_size,
            "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
