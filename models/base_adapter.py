"""Abstract model adapter interface so UI & services stay decoupled from YOLO11."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from core.schemas import DetectionBox, EpochMetric, ExportArtifact, InferenceResult, TrainRequest


@dataclass
class ModelCapabilities:
    model_family: str
    task: str
    display_name: str
    variants: dict[str, dict[str, str]]
    supported_modes: list[str] = field(default_factory=lambda: ["train"])
    supported_export_formats: list[str] = field(default_factory=lambda: ["pt", "onnx"])


class BaseModelAdapter(ABC):
    """Consistent contract for all current (YOLO11 Detection) and future (YOLOv8, Seg, Cls) adapters."""

    @abstractmethod
    def get_capabilities(self) -> ModelCapabilities:
        """Describe supported variants, modes, and export formats."""

    @abstractmethod
    def validate_train_request(self, request: TrainRequest) -> list[str]:
        """Return a list of validation error strings (empty if valid)."""

    @abstractmethod
    def train(
        self,
        request: TrainRequest,
        weights_cache_dir: Path,
        on_epoch_end: Callable[[EpochMetric], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        on_log: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Run training synchronously inside the worker subprocess and return final run artifacts."""

    @abstractmethod
    def predict_image(
        self,
        weights_path: Path,
        image_path: Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        output_image_path: Path | None = None,
    ) -> InferenceResult:
        """Run single-image inference and optionally write annotated image."""

    def predict_boxes_batch(
        self,
        weights_path: Path,
        image_paths: list[Path],
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
    ) -> list[tuple[str, int, int, list[DetectionBox]]]:
        """Run detection on many images with one model load. Optional capability for Phase 2 auto-label."""
        raise NotImplementedError("Adapter does not support batch prediction yet")

    @abstractmethod
    def export_model(
        self,
        weights_path: Path,
        export_format: str,
        output_dir: Path,
        imgsz: int = 640,
    ) -> ExportArtifact:
        """Export weights to target format and return real artifact metadata."""
