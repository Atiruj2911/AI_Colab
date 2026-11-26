"""Inference runner delegating to the registered ModelAdapter."""

from __future__ import annotations

import uuid
from pathlib import Path

from core.schemas import InferenceResult
from models.registry import ModelRegistry


class ImagePredictor:
    """Runs single-image object detection via ModelRegistry."""

    def __init__(self, registry: ModelRegistry, temp_dir: Path) -> None:
        self.registry = registry
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def predict(
        self,
        weights_path: Path,
        image_path: Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        model_family: str = "yolo11",
        task: str = "detect",
    ) -> InferenceResult:
        adapter = self.registry.get(model_family, task)
        out_path = self.temp_dir / f"pred_{uuid.uuid4().hex[:10]}.jpg"
        return adapter.predict_image(
            weights_path=Path(weights_path),
            image_path=Path(image_path),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            output_image_path=out_path,
        )
