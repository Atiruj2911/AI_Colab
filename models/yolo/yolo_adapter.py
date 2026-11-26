"""Ultralytics YOLO11 Detection adapter implementing BaseModelAdapter."""

from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from config.defaults import (
    ALLOWED_IMG_SIZES,
    MAX_BATCH_SIZE,
    MAX_EPOCHS,
    MIN_BATCH_SIZE,
    MIN_EPOCHS,
    YOLO11_MODEL_VARIANTS,
)
from core.schemas import DetectionBox, EpochMetric, ExportArtifact, InferenceResult, TrainRequest
from models.base_adapter import BaseModelAdapter, ModelCapabilities


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        num = float(val)
        if num != num:  # NaN
            return None
        return num
    except (TypeError, ValueError):
        return None


def _resolve_device(requested_device: str) -> str | int:
    import torch

    dev = (requested_device or "auto").strip().lower()
    if dev == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        return "cpu"
    if dev.isdigit():
        if torch.cuda.is_available() and int(dev) < torch.cuda.device_count():
            return int(dev)
        return "cpu"
    return dev


def _extract_epoch_metric_from_trainer(
    trainer: Any,
    start_time: float,
    total_epochs: int,
) -> EpochMetric:
    """Extract real epoch metrics from Ultralytics BaseTrainer (with results.csv fallback)."""
    epoch_num = int(getattr(trainer, "epoch", 0)) + 1
    metrics_dict: dict[str, Any] = dict(getattr(trainer, "metrics", {}) or {})

    train_box = train_cls = train_dfl = None
    try:
        tloss = getattr(trainer, "tloss", None)
        if tloss is not None and hasattr(trainer, "label_loss_items"):
            labeled = trainer.label_loss_items(tloss, prefix="train")
            if isinstance(labeled, dict):
                train_box = _safe_float(labeled.get("train/box_loss"))
                train_cls = _safe_float(labeled.get("train/cls_loss"))
                train_dfl = _safe_float(labeled.get("train/dfl_loss"))
    except Exception:
        pass

    val_box = _safe_float(metrics_dict.get("val/box_loss"))
    val_cls = _safe_float(metrics_dict.get("val/cls_loss"))
    val_dfl = _safe_float(metrics_dict.get("val/dfl_loss"))
    precision = _safe_float(metrics_dict.get("metrics/precision(B)"))
    recall = _safe_float(metrics_dict.get("metrics/recall(B)"))
    map50 = _safe_float(metrics_dict.get("metrics/mAP50(B)"))
    map50_95 = _safe_float(metrics_dict.get("metrics/mAP50-95(B)"))

    # Fallback to results.csv if any value was missing during callback timing
    csv_path = getattr(trainer, "csv", None)
    if csv_path is None and getattr(trainer, "save_dir", None):
        csv_path = Path(trainer.save_dir) / "results.csv"
    if csv_path and Path(csv_path).exists():
        try:
            with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                last_row = {k.strip(): v.strip() for k, v in rows[-1].items() if k}
                train_box = train_box if train_box is not None else _safe_float(last_row.get("train/box_loss"))
                train_cls = train_cls if train_cls is not None else _safe_float(last_row.get("train/cls_loss"))
                train_dfl = train_dfl if train_dfl is not None else _safe_float(last_row.get("train/dfl_loss"))
                val_box = val_box if val_box is not None else _safe_float(last_row.get("val/box_loss"))
                val_cls = val_cls if val_cls is not None else _safe_float(last_row.get("val/cls_loss"))
                val_dfl = val_dfl if val_dfl is not None else _safe_float(last_row.get("val/dfl_loss"))
                precision = precision if precision is not None else _safe_float(last_row.get("metrics/precision(B)"))
                recall = recall if recall is not None else _safe_float(last_row.get("metrics/recall(B)"))
                map50 = map50 if map50 is not None else _safe_float(last_row.get("metrics/mAP50(B)"))
                map50_95 = map50_95 if map50_95 is not None else _safe_float(last_row.get("metrics/mAP50-95(B)"))
        except Exception:
            pass

    train_parts = [v for v in (train_box, train_cls, train_dfl) if v is not None]
    val_parts = [v for v in (val_box, val_cls, val_dfl) if v is not None]

    return EpochMetric(
        epoch=epoch_num,
        total_epochs=total_epochs,
        train_box_loss=train_box,
        train_cls_loss=train_cls,
        train_dfl_loss=train_dfl,
        val_box_loss=val_box,
        val_cls_loss=val_cls,
        val_dfl_loss=val_dfl,
        train_loss_total=sum(train_parts) if train_parts else None,
        val_loss_total=sum(val_parts) if val_parts else None,
        precision=precision,
        recall=recall,
        map50=map50,
        map50_95=map50_95,
        elapsed_seconds=max(0.0, time.time() - start_time),
    )


class Yolo11DetectionAdapter(BaseModelAdapter):
    """Phase 1 adapter for Ultralytics YOLO11 Object Detection (n / s / m)."""

    VALID_VARIANTS = {spec["model_id"] for spec in YOLO11_MODEL_VARIANTS.values()}

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_family="yolo11",
            task="detect",
            display_name="Ultralytics YOLO11 (Object Detection)",
            variants=YOLO11_MODEL_VARIANTS,
            supported_modes=["train", "fine_tune", "resume"],
            supported_export_formats=["pt", "onnx"],
        )

    def validate_train_request(self, request: TrainRequest) -> list[str]:
        errors: list[str] = []
        if request.model_variant not in self.VALID_VARIANTS:
            errors.append(
                f"ไม่รองรับขนาดโมเดล '{request.model_variant}' (Phase 1 รองรับ: {', '.join(sorted(self.VALID_VARIANTS))})"
            )
        if request.mode not in {"train", "fine_tune", "resume"}:
            errors.append(
                f"โหมด '{request.mode}' ไม่ถูกต้อง (รองรับ: train, fine_tune, resume)"
            )
        if request.mode == "resume":
            wp = Path(request.pretrained_weights)
            if not wp.is_file():
                errors.append(f"โหมด Resume ต้องระบุไฟล์ Checkpoint (last.pt) ที่มีอยู่จริง: {request.pretrained_weights}")
        if request.mode == "fine_tune":
            wp = Path(request.pretrained_weights)
            known_pretrained = {spec["pretrained_weights"] for spec in YOLO11_MODEL_VARIANTS.values()}
            if request.pretrained_weights not in known_pretrained and not wp.is_file():
                errors.append(f"โหมด Fine-tune ไม่พบไฟล์ Weight: {request.pretrained_weights}")
        if not (MIN_EPOCHS <= request.epochs <= MAX_EPOCHS):
            errors.append(f"Epochs ต้องอยู่ระหว่าง {MIN_EPOCHS} ถึง {MAX_EPOCHS}")
        if not (MIN_BATCH_SIZE <= request.batch_size <= MAX_BATCH_SIZE):
            errors.append(f"Batch Size ต้องอยู่ระหว่าง {MIN_BATCH_SIZE} ถึง {MAX_BATCH_SIZE}")
        if request.imgsz not in ALLOWED_IMG_SIZES:
            errors.append(f"Image Size ต้องเป็นหนึ่งในค่าที่รองรับ: {ALLOWED_IMG_SIZES}")
        if not Path(request.dataset_yaml).exists():
            errors.append(f"ไม่พบไฟล์ data.yaml ที่ระบุ: {request.dataset_yaml}")
        return errors

    def _resolve_weights_target(self, pretrained_weights: str, weights_cache_dir: Path) -> str:
        weights_cache_dir.mkdir(parents=True, exist_ok=True)
        candidate = Path(pretrained_weights)
        if candidate.is_file():
            return str(candidate.resolve())
        cached = weights_cache_dir / candidate.name
        return str(cached)

    def train(
        self,
        request: TrainRequest,
        weights_cache_dir: Path,
        on_epoch_end: Callable[[EpochMetric], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        on_log: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        import os

        from ultralytics import YOLO

        weights_target = self._resolve_weights_target(request.pretrained_weights, weights_cache_dir)
        device = _resolve_device(request.device)
        # Avoid Windows multiprocessing DataLoader spawn deadlocks inside detached workers
        workers = 0 if os.name == "nt" else max(0, int(request.workers))

        if on_log:
            on_log(
                f"[YOLO11Adapter] Loading weights: {weights_target} | device={device} | "
                f"epochs={request.epochs} | batch={request.batch_size} | imgsz={request.imgsz}"
            )

        model = YOLO(weights_target)
        start_time = time.time()
        stopped_by_user = {"flag": False}

        def _handle_batch_end(trainer: Any) -> None:
            if should_stop and should_stop():
                stopped_by_user["flag"] = True
                trainer.stop = True

        def _handle_fit_epoch_end(trainer: Any) -> None:
            metric = _extract_epoch_metric_from_trainer(trainer, start_time, request.epochs)
            if on_epoch_end:
                on_epoch_end(metric)
            if should_stop and should_stop():
                stopped_by_user["flag"] = True
                trainer.stop = True

        model.add_callback("on_train_batch_end", _handle_batch_end)
        model.add_callback("on_fit_epoch_end", _handle_fit_epoch_end)

        run_dir = Path(request.output_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        train_kwargs: dict[str, Any] = {
            "data": str(Path(request.dataset_yaml).resolve()),
            "epochs": int(request.epochs),
            "batch": int(request.batch_size),
            "imgsz": int(request.imgsz),
            "device": device,
            "seed": int(request.seed),
            "patience": int(request.patience),
            "workers": workers,
            "project": str(run_dir),
            "name": "ultralytics",
            "exist_ok": True,
            "save": bool(request.auto_save.save_best or request.auto_save.save_last),
            "plots": True,
            "verbose": True,
        }
        if request.mode == "resume":
            train_kwargs["resume"] = True
        extra = request.extra_params or {}
        if extra.get("optimizer") and str(extra["optimizer"]) != "auto":
            train_kwargs["optimizer"] = str(extra["optimizer"])
        if extra.get("lr0") is not None:
            train_kwargs["lr0"] = float(extra["lr0"])

        model.train(**train_kwargs)

        trainer = getattr(model, "trainer", None)
        save_dir = Path(trainer.save_dir) if trainer and getattr(trainer, "save_dir", None) else (run_dir / "ultralytics")
        weights_dir = save_dir / "weights"
        best_pt = weights_dir / "best.pt"
        last_pt = weights_dir / "last.pt"

        # Copy canonical checkpoints to run_dir/weights/ for consistent run layout
        canonical_weights_dir = run_dir / "weights"
        canonical_weights_dir.mkdir(parents=True, exist_ok=True)
        final_best: str | None = None
        final_last: str | None = None

        if best_pt.exists():
            dst = canonical_weights_dir / "best.pt"
            shutil.copy2(best_pt, dst)
            final_best = str(dst)
        if last_pt.exists():
            dst = canonical_weights_dir / "last.pt"
            shutil.copy2(last_pt, dst)
            final_last = str(dst)

        return {
            "save_dir": str(save_dir),
            "best_checkpoint_path": final_best,
            "last_checkpoint_path": final_last,
            "stopped_by_user": stopped_by_user["flag"],
            "device_used": str(device),
        }

    def predict_image(
        self,
        weights_path: Path,
        image_path: Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        output_image_path: Path | None = None,
    ) -> InferenceResult:
        from ultralytics import YOLO

        weights_path = Path(weights_path)
        image_path = Path(image_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {weights_path}")
        if not image_path.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์รูปภาพ: {image_path}")

        with Image.open(image_path) as im:
            img_w, img_h = im.size

        model = YOLO(str(weights_path))
        t0 = time.perf_counter()
        results = model.predict(
            source=str(image_path),
            conf=float(conf),
            iou=float(iou),
            imgsz=int(imgsz),
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        detections: list[DetectionBox] = []
        annotated_path_str: str | None = None

        if results:
            res = results[0]
            names = res.names or {}
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy_list = boxes.xyxy.cpu().tolist()
                conf_list = boxes.conf.cpu().tolist()
                cls_list = boxes.cls.cpu().tolist()
                for xyxy, score, cls_id in zip(xyxy_list, conf_list, cls_list):
                    cid = int(cls_id)
                    detections.append(
                        DetectionBox(
                            class_id=cid,
                            class_name=str(names.get(cid, f"class_{cid}")),
                            confidence=float(score),
                            x1=float(xyxy[0]),
                            y1=float(xyxy[1]),
                            x2=float(xyxy[2]),
                            y2=float(xyxy[3]),
                        )
                    )

            if output_image_path is not None:
                output_image_path = Path(output_image_path)
                output_image_path.parent.mkdir(parents=True, exist_ok=True)
                plotted_bgr = res.plot()
                plotted_rgb = plotted_bgr[:, :, ::-1]
                Image.fromarray(plotted_rgb).save(output_image_path)
                annotated_path_str = str(output_image_path)

        return InferenceResult(
            model_path=str(weights_path),
            image_width=img_w,
            image_height=img_h,
            inference_ms=round(elapsed_ms, 2),
            detections=detections,
            annotated_image_path=annotated_path_str,
        )

    def predict_boxes_batch(
        self,
        weights_path: Path,
        image_paths: list[Path],
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
    ) -> list[tuple[str, int, int, list[DetectionBox]]]:
        from ultralytics import YOLO

        weights_str = str(weights_path)
        known_pretrained = {"yolo11n.pt", "yolo11s.pt", "yolo11m.pt"}
        if weights_str not in known_pretrained and not Path(weights_path).exists():
            raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {weights_path}")
        model = YOLO(weights_str)
        output: list[tuple[str, int, int, list[DetectionBox]]] = []
        for image_path in image_paths:
            image_path = Path(image_path)
            with Image.open(image_path) as im:
                img_w, img_h = im.size
            results = model.predict(
                source=str(image_path),
                conf=float(conf),
                iou=float(iou),
                imgsz=int(imgsz),
                verbose=False,
            )
            detections: list[DetectionBox] = []
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                res = results[0]
                names = res.names or {}
                for xyxy, score, cls_id in zip(
                    res.boxes.xyxy.cpu().tolist(),
                    res.boxes.conf.cpu().tolist(),
                    res.boxes.cls.cpu().tolist(),
                ):
                    cid = int(cls_id)
                    detections.append(
                        DetectionBox(
                            class_id=cid,
                            class_name=str(names.get(cid, f"class_{cid}")),
                            confidence=float(score),
                            x1=float(xyxy[0]),
                            y1=float(xyxy[1]),
                            x2=float(xyxy[2]),
                            y2=float(xyxy[3]),
                        )
                    )
            output.append((str(image_path), img_w, img_h, detections))
        return output

    def export_model(
        self,
        weights_path: Path,
        export_format: str,
        output_dir: Path,
        imgsz: int = 640,
    ) -> ExportArtifact:
        fmt = export_format.strip().lower()
        weights_path = Path(weights_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not weights_path.exists():
            return ExportArtifact(
                format=fmt,
                path="",
                filename="",
                size_bytes=0,
                success=False,
                message=f"ไม่พบไฟล์ต้นทาง: {weights_path}",
            )

        if fmt == "pt":
            dest = output_dir / f"{weights_path.stem}.pt"
            shutil.copy2(weights_path, dest)
            return ExportArtifact(
                format="pt",
                path=str(dest),
                filename=dest.name,
                size_bytes=dest.stat().st_size,
                success=True,
                message="เตรียมไฟล์ PyTorch (.pt) พร้อมดาวน์โหลดแล้ว",
            )

        if fmt == "onnx":
            try:
                import onnx  # noqa: F401
            except ImportError as exc:
                return ExportArtifact(
                    format="onnx",
                    path="",
                    filename="",
                    size_bytes=0,
                    success=False,
                    message=(
                        "ไม่สามารถ Export เป็น ONNX ได้ เนื่องจากยังไม่ได้ติดตั้งแพ็กเกจ 'onnx' "
                        "กรุณาติดตั้งด้วย: pip install onnx onnxslim onnxruntime"
                    ),
                    error_detail=str(exc),
                )

            try:
                from ultralytics import YOLO

                # Work on a temp copy inside output_dir so original run folder is untouched
                staged_pt = output_dir / f"_export_src_{weights_path.name}"
                shutil.copy2(weights_path, staged_pt)
                try:
                    model = YOLO(str(staged_pt))
                    exported_path = Path(
                        model.export(format="onnx", imgsz=int(imgsz), simplify=False)
                    )
                finally:
                    if staged_pt.exists():
                        staged_pt.unlink(missing_ok=True)

                if not exported_path.exists():
                    return ExportArtifact(
                        format="onnx",
                        path="",
                        filename="",
                        size_bytes=0,
                        success=False,
                        message="กระบวนการ Export ONNX ไม่สร้างไฟล์ผลลัพธ์",
                    )

                final_onnx = output_dir / f"{weights_path.stem}.onnx"
                if exported_path.resolve() != final_onnx.resolve():
                    shutil.move(str(exported_path), str(final_onnx))

                return ExportArtifact(
                    format="onnx",
                    path=str(final_onnx),
                    filename=final_onnx.name,
                    size_bytes=final_onnx.stat().st_size,
                    success=True,
                    message="Export โมเดล ONNX (.onnx) สำเร็จ",
                )
            except Exception as exc:
                return ExportArtifact(
                    format="onnx",
                    path="",
                    filename="",
                    size_bytes=0,
                    success=False,
                    message=f"Export ONNX ล้มเหลว: {exc}",
                    error_detail=repr(exc),
                )

        return ExportArtifact(
            format=fmt,
            path="",
            filename="",
            size_bytes=0,
            success=False,
            message=f"รูปแบบ Export '{fmt}' ยังไม่รองรับใน Phase 1 (รองรับเฉพาะ .pt และ .onnx)",
        )
