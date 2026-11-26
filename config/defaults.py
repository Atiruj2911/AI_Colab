"""Default constants, model specs, and safe limits for AI Training Studio."""

from __future__ import annotations

APP_NAME = "AI Training Studio"
APP_VERSION = "0.1.0-mvp"
APP_PHASE = "Phase 1 — End-to-End MVP"

# Supported model family and tasks in Phase 1 (extensible via ModelRegistry in Phase 4)
DEFAULT_MODEL_FAMILY = "yolo11"
DEFAULT_TASK = "detect"

YOLO11_MODEL_VARIANTS = {
    "YOLO11n": {
        "model_id": "yolo11n",
        "pretrained_weights": "yolo11n.pt",
        "description": "Nano — fastest training and lowest VRAM usage",
    },
    "YOLO11s": {
        "model_id": "yolo11s",
        "pretrained_weights": "yolo11s.pt",
        "description": "Small — balanced speed and accuracy",
    },
    "YOLO11m": {
        "model_id": "yolo11m",
        "pretrained_weights": "yolo11m.pt",
        "description": "Medium — higher accuracy with moderate VRAM",
    },
    "YOLO11l": {
        "model_id": "yolo11l",
        "pretrained_weights": "yolo11l.pt",
        "description": "Large — high accuracy for larger GPUs",
    },
    "YOLO11x": {
        "model_id": "yolo11x",
        "pretrained_weights": "yolo11x.pt",
        "description": "Extra Large — highest capacity and VRAM demand",
    },
}

DEFAULT_MODEL_VARIANT_LABEL = "YOLO11s"

# Training defaults & safety bounds
DEFAULT_EPOCHS = 50
MIN_EPOCHS = 1
MAX_EPOCHS = 1000

DEFAULT_BATCH_SIZE = 16
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 256

DEFAULT_IMG_SIZE = 640
ALLOWED_IMG_SIZES = [320, 416, 512, 640, 768, 800, 960, 1024, 1280]

DEFAULT_SEED = 42
DEFAULT_PATIENCE = 50
DEFAULT_WORKERS = 2

# Inference defaults
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45

# Export formats (Phase 1 = PyTorch .pt only)
PHASE1_EXPORT_FORMATS = ["pt"]
PLANNED_EXPORT_FORMATS = [
    ("torchscript", "TorchScript (.torchscript) — เพิ่มเติมใน Phase 5"),
    ("openvino", "OpenVINO — เพิ่มเติมใน Phase 5"),
    ("tensorrt", "TensorRT (.engine) — เพิ่มเติมใน Phase 5"),
]

# Upload & Archive safety limits
MAX_UPLOAD_ARCHIVE_MB = 2048  # 2 GB
MAX_EXTRACTED_SIZE_MB = 8192  # 8 GB
MAX_ARCHIVE_FILES = 100_000
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ALLOWED_LABEL_EXTENSIONS = {".txt"}
ALLOWED_YAML_EXTENSIONS = {".yaml", ".yml"}

# Monitoring & process settings
DEFAULT_MONITOR_REFRESH_SECONDS = 3.0
MIN_MONITOR_REFRESH_SECONDS = 1.5
MAX_MONITOR_REFRESH_SECONDS = 30.0
HEARTBEAT_STALE_SECONDS = 120.0
GRACEFUL_STOP_TIMEOUT_SECONDS = 45.0
MAX_LOG_LINES_UI = 250
SAMPLE_PREVIEW_COUNT = 8
