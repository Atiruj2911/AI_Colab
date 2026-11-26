"""Model export coordinator delegating to the registered ModelAdapter."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from core.schemas import ExportArtifact, utc_now_iso
from models.registry import ModelRegistry
from utils.validation import sanitize_identifier


class ModelExporter:
    """Exports trained weights to requested formats and packages run config alongside artifacts."""

    def __init__(self, registry: ModelRegistry, exports_dir: Path) -> None:
        self.registry = registry
        self.exports_dir = Path(exports_dir)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    def export_formats(
        self,
        weights_path: Path,
        formats: list[str],
        run_name: str = "model",
        imgsz: int = 640,
        model_family: str = "yolo11",
        task: str = "detect",
        run_dir: Path | None = None,
    ) -> tuple[Path, list[ExportArtifact]]:
        adapter = self.registry.get(model_family, task)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = sanitize_identifier(run_name, fallback="model")
        export_bundle_dir = self.exports_dir / f"{safe_name}_{stamp}"
        export_bundle_dir.mkdir(parents=True, exist_ok=True)

        artifacts: list[ExportArtifact] = []
        for fmt in formats:
            artifact = adapter.export_model(
                weights_path=Path(weights_path),
                export_format=fmt,
                output_dir=export_bundle_dir,
                imgsz=imgsz,
            )
            artifacts.append(artifact)

        # Include run configuration & data.yaml when exporting from a tracked run
        if run_dir and Path(run_dir).exists():
            for cfg_name in ("train_config.json", "data.yaml", "runtime_meta.json"):
                src = Path(run_dir) / cfg_name
                if src.exists():
                    shutil.copy2(src, export_bundle_dir / cfg_name)

        manifest = {
            "exported_at": utc_now_iso(),
            "source_weights": str(weights_path),
            "model_family": model_family,
            "task": task,
            "imgsz": imgsz,
            "artifacts": [a.to_dict() for a in artifacts],
        }
        (export_bundle_dir / "export_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return export_bundle_dir, artifacts
