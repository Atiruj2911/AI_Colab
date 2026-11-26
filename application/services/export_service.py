"""ExportService: export trained models to .pt and .onnx with clear diagnostics."""

from __future__ import annotations

from pathlib import Path

from core.export.exporter import ModelExporter
from core.schemas import ExportArtifact
from infrastructure.process.job_manager import JobManager


class ExportService:
    """Validates export selections and runs ModelExporter."""

    def __init__(self, exporter: ModelExporter, job_manager: JobManager) -> None:
        self.exporter = exporter
        self.job_manager = job_manager

    def export_model(
        self,
        weights_path: str | Path,
        formats: list[str],
        imgsz: int = 640,
        run_id: str | None = None,
    ) -> tuple[Path, list[ExportArtifact]]:
        if not weights_path:
            raise ValueError("กรุณาเลือกโมเดล (.pt) ที่ต้องการ Export")
        if not formats:
            raise ValueError("กรุณาเลือกอย่างน้อย 1 รูปแบบการ Export (.pt หรือ .onnx)")

        w_path = Path(weights_path)
        if not w_path.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {w_path}")

        run_dir: Path | None = None
        run_name = w_path.stem
        if run_id:
            job = self.job_manager.get_job_by_run_id(run_id)
            if job:
                run_name = job.run_name
                if job.run_dir:
                    run_dir = Path(job.run_dir)
        elif w_path.parent.name == "weights":
            candidate_run_dir = w_path.parent.parent
            if (candidate_run_dir / "train_config.json").exists():
                run_dir = candidate_run_dir
                run_name = candidate_run_dir.name

        return self.exporter.export_formats(
            weights_path=w_path,
            formats=formats,
            run_name=run_name,
            imgsz=int(imgsz),
            run_dir=run_dir,
        )
