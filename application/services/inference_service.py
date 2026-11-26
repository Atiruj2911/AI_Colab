"""InferenceService: validate model & image inputs and run object detection."""

from __future__ import annotations

from pathlib import Path

from core.inference.predictor import ImagePredictor
from core.schemas import InferenceResult


class InferenceService:
    """Coordinates single-image testing on trained or uploaded YOLO weights."""

    def __init__(self, predictor: ImagePredictor) -> None:
        self.predictor = predictor

    def detect_image(
        self,
        weights_path: str | Path,
        image_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
    ) -> InferenceResult:
        if not weights_path:
            raise ValueError("กรุณาเลือกไฟล์โมเดล (.pt) หรืออัปโหลดไฟล์น้ำหนักก่อนทดสอบ")
        if not image_path:
            raise ValueError("กรุณาอัปโหลดรูปภาพสำหรับทดสอบตรวจจับ")
        if not (0.01 <= float(conf) <= 0.99):
            raise ValueError("ค่า Confidence ต้องอยู่ระหว่าง 0.01 ถึง 0.99")
        if not (0.05 <= float(iou) <= 0.99):
            raise ValueError("ค่า IoU (NMS) ต้องอยู่ระหว่าง 0.05 ถึง 0.99")

        return self.predictor.predict(
            weights_path=Path(weights_path),
            image_path=Path(image_path),
            conf=float(conf),
            iou=float(iou),
            imgsz=int(imgsz),
        )
