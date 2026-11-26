"""Shared domain schemas/dataclasses designed to support Phase 1 and Phase 2-5 growth."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ValidationIssue:
    """Actionable error or warning produced during dataset/config validation."""

    severity: str  # "error" | "warning"
    code: str
    message: str
    path: str | None = None
    hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SplitStats:
    split_name: str
    images_dir: str | None = None
    labels_dir: str | None = None
    image_count: int = 0
    label_count: int = 0
    matched_pairs: int = 0
    images_without_labels: int = 0
    orphan_labels: int = 0
    empty_labels: int = 0
    malformed_label_files: int = 0
    unreadable_images: int = 0
    total_boxes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetValidationReport:
    is_valid: bool
    dataset_id: str
    dataset_name: str
    dataset_root: str
    data_yaml_path: str | None = None
    resolved_yaml_path: str | None = None
    class_names: dict[int, str] = field(default_factory=dict)
    splits: dict[str, SplitStats] = field(default_factory=dict)
    total_images: int = 0
    total_labels: int = 0
    total_boxes: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    structure_tree: str = ""
    sample_images: list[str] = field(default_factory=list)
    validated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["class_names"] = {str(k): v for k, v in self.class_names.items()}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetValidationReport":
        raw_names = data.get("class_names", {}) or {}
        class_names = {int(k): str(v) for k, v in raw_names.items()}
        splits = {
            name: SplitStats(**stats)
            for name, stats in (data.get("splits", {}) or {}).items()
        }
        errors = [ValidationIssue(**item) for item in (data.get("errors", []) or [])]
        warnings = [ValidationIssue(**item) for item in (data.get("warnings", []) or [])]
        return cls(
            is_valid=bool(data.get("is_valid", False)),
            dataset_id=str(data.get("dataset_id", "")),
            dataset_name=str(data.get("dataset_name", "")),
            dataset_root=str(data.get("dataset_root", "")),
            data_yaml_path=data.get("data_yaml_path"),
            resolved_yaml_path=data.get("resolved_yaml_path"),
            class_names=class_names,
            splits=splits,
            total_images=int(data.get("total_images", 0)),
            total_labels=int(data.get("total_labels", 0)),
            total_boxes=int(data.get("total_boxes", 0)),
            class_counts=dict(data.get("class_counts", {}) or {}),
            errors=errors,
            warnings=warnings,
            structure_tree=str(data.get("structure_tree", "")),
            sample_images=list(data.get("sample_images", []) or []),
            validated_at=str(data.get("validated_at", utc_now_iso())),
        )


@dataclass
class AutoSavePolicy:
    """Schema ready for Phase 4 configurable auto-save policies (MVP uses standard best+last)."""

    enabled: bool = True
    save_best: bool = True
    save_last: bool = True
    min_map_improvement: float = 0.0
    save_every_n_epochs: int | None = None
    save_every_n_minutes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainRequest:
    """Canonical training request passed from Application -> Worker -> ModelAdapter."""

    session_id: str
    run_id: str
    run_name: str
    dataset_id: str
    dataset_yaml: str
    model_family: str = "yolo11"
    task: str = "detect"
    model_variant: str = "yolo11s"
    pretrained_weights: str = "yolo11s.pt"
    mode: str = "train"  # Phase 1: "train"; Phase 4: "fine_tune", "resume"
    epochs: int = 50
    batch_size: int = 16
    imgsz: int = 640
    device: str = "auto"
    seed: int = 42
    patience: int = 50
    workers: int = 2
    output_dir: str = ""
    auto_save: AutoSavePolicy = field(default_factory=AutoSavePolicy)
    extra_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainRequest":
        auto_save_data = data.get("auto_save") or {}
        return cls(
            session_id=str(data["session_id"]),
            run_id=str(data["run_id"]),
            run_name=str(data["run_name"]),
            dataset_id=str(data["dataset_id"]),
            dataset_yaml=str(data["dataset_yaml"]),
            model_family=str(data.get("model_family", "yolo11")),
            task=str(data.get("task", "detect")),
            model_variant=str(data.get("model_variant", "yolo11s")),
            pretrained_weights=str(data.get("pretrained_weights", "yolo11s.pt")),
            mode=str(data.get("mode", "train")),
            epochs=int(data.get("epochs", 50)),
            batch_size=int(data.get("batch_size", 16)),
            imgsz=int(data.get("imgsz", 640)),
            device=str(data.get("device", "auto")),
            seed=int(data.get("seed", 42)),
            patience=int(data.get("patience", 50)),
            workers=int(data.get("workers", 2)),
            output_dir=str(data.get("output_dir", "")),
            auto_save=AutoSavePolicy(**auto_save_data),
            extra_params=dict(data.get("extra_params") or {}),
        )


@dataclass
class EpochMetric:
    epoch: int
    total_epochs: int
    train_box_loss: float | None = None
    train_cls_loss: float | None = None
    train_dfl_loss: float | None = None
    val_box_loss: float | None = None
    val_cls_loss: float | None = None
    val_dfl_loss: float | None = None
    train_loss_total: float | None = None
    val_loss_total: float | None = None
    precision: float | None = None
    recall: float | None = None
    map50: float | None = None
    map50_95: float | None = None
    elapsed_seconds: float | None = None
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JobState:
    """Persisted state of a background training job so UI reconnects cleanly."""

    job_id: str
    session_id: str
    run_id: str
    run_name: str
    dataset_id: str
    model_family: str
    task: str
    model_variant: str
    status: str  # "queued" | "starting" | "running" | "stopping" | "completed" | "stopped" | "failed"
    pid: int | None = None
    current_epoch: int = 0
    total_epochs: int = 0
    progress_pct: float = 0.0
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    best_map50: float | None = None
    best_map50_95: float | None = None
    best_epoch: int | None = None
    latest_metrics: dict[str, Any] = field(default_factory=dict)
    best_checkpoint_path: str | None = None
    last_checkpoint_path: str | None = None
    run_dir: str = ""
    log_path: str = ""
    metrics_path: str = ""
    started_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    heartbeat_at: str = field(default_factory=utc_now_iso)
    finished_at: str | None = None
    stop_requested: bool = False
    error_message: str | None = None
    error_detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CheckpointArtifact:
    name: str
    kind: str  # "best" | "last" | "export" | "config" | "metadata" | "log"
    path: str
    run_id: str
    run_name: str
    size_bytes: int
    modified_at: str
    best_map50_95: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DetectionBox:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceResult:
    model_path: str
    image_width: int
    image_height: int
    inference_ms: float
    detections: list[DetectionBox] = field(default_factory=list)
    annotated_image_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExportArtifact:
    format: str
    path: str
    filename: str
    size_bytes: int
    success: bool
    message: str = ""
    error_detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
