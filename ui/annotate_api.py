"""FastAPI routes for the dedicated Roboflow-style Annotation Workspace.

Reuses AppController / DatasetService / AnnotationService directly — zero duplicated dataset logic.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from application.app_controller import AppController
from core.dataset.entries import SPLITS
from utils.validation import format_bytes

STATIC_DIR = Path(__file__).resolve().parent / "static"
_MAX_IMPORT_BYTES = 512 * 1024 * 1024


def _safe_pt_name(filename: str) -> str:
    stem = Path(filename or "").name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    if not cleaned.lower().endswith(".pt"):
        cleaned = f"{cleaned or 'model'}.pt"
    return cleaned


def _imported_models(controller: AppController) -> list[dict[str, str]]:
    folder = controller.workspace.weights_dir / "imported"
    if not folder.is_dir():
        return []
    models: list[dict[str, str]] = []
    for path in sorted(folder.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True):
        if not path.is_file():
            continue
        models.append(
            {
                "label": f"Imported {path.name} ({format_bytes(path.stat().st_size)})",
                "path": str(path.resolve()),
            }
        )
    return models


PRETRAINED_MODELS = [
    {"label": "yolo11n.pt (โมเดลเล็ก เร็ว — โหลดอัตโนมัติ)", "path": "yolo11n.pt"},
    {"label": "yolo11s.pt (โมเดลกลาง — โหลดอัตโนมัติ)", "path": "yolo11s.pt"},
]


class BoxPayload(BaseModel):
    label: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


class SaveLabelsPayload(BaseModel):
    boxes: list[BoxPayload]


class ClassNamePayload(BaseModel):
    name: str


class ApplyAugPayload(BaseModel):
    """Module-level so FastAPI can resolve it as a JSON body. A class nested inside the router factory is treated as a missing query parameter."""

    copies_per_image: int = 2
    seed: int = 42
    new_name: str | None = None
    ranges: dict[str, dict[str, Any]]


class AutoLabelSinglePayload(BaseModel):
    split: str
    key: str
    weights_path: str
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    auto_add_classes: bool = True


class AutoLabelBatchPayload(BaseModel):
    split: str = "train"
    weights_path: str
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    scope: str = "unlabeled"
    auto_add_classes: bool = True
    allow_overwrite: bool = False


def create_annotate_router(controller: AppController) -> APIRouter:
    router = APIRouter()
    ann = controller.annotation_service

    @router.get("/annotate", response_class=HTMLResponse)
    def serve_annotator_page() -> HTMLResponse:
        html_path = STATIC_DIR / "annotator.html"
        if not html_path.exists():
            raise HTTPException(status_code=500, detail="annotator.html not found")
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

    @router.get("/api/annotate/datasets")
    def list_datasets() -> dict[str, Any]:
        items = []
        for ds in controller.list_datasets():
            if ds.total_images <= 0:
                continue
            items.append(
                {
                    "dataset_id": ds.dataset_id,
                    "dataset_name": ds.dataset_name,
                    "total_images": ds.total_images,
                    "is_valid": ds.is_valid,
                    "class_count": len(ds.class_names),
                }
            )
        return {"datasets": items}

    @router.get("/api/annotate/dataset/{dataset_id}/images")
    def list_images(
        dataset_id: str,
        split: str = Query("train"),
        mode: str = Query("all"),
    ) -> dict[str, Any]:
        if split not in SPLITS:
            raise HTTPException(status_code=400, detail=f"Invalid split: {split}")
        try:
            rows = ann.list_images(dataset_id, split, mode, limit=1000)
            classes = ann.list_classes(dataset_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "dataset_id": dataset_id,
            "split": split,
            "mode": mode,
            "images": [
                {
                    "split": r["split"],
                    "key": r["key"],
                    "has_label": r["has_label"],
                    "box_count": r["box_count"],
                }
                for r in rows
            ],
            "classes": classes,
        }

    @router.get("/api/annotate/dataset/{dataset_id}/image/{split}/{key:path}")
    def get_image_file(dataset_id: str, split: str, key: str) -> FileResponse:
        try:
            _, entry = ann._entry(dataset_id, split, key)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(entry.image_path)

    @router.get("/api/annotate/dataset/{dataset_id}/labels/{split}/{key:path}")
    def get_image_labels(dataset_id: str, split: str, key: str) -> dict[str, Any]:
        try:
            data = ann.load_annotator_value(dataset_id, split, key)
            classes = ann.list_classes(dataset_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "split": split,
            "key": key,
            "width": data["width"],
            "height": data["height"],
            "has_label": data["has_label"],
            "boxes": data["value"]["boxes"],
            "classes": classes,
            "problems": data["problems"],
        }

    @router.post("/api/annotate/dataset/{dataset_id}/labels/{split}/{key:path}")
    def save_image_labels(dataset_id: str, split: str, key: str, payload: SaveLabelsPayload) -> dict[str, Any]:
        try:
            res = ann.save_annotator_value(
                dataset_id=dataset_id,
                split=split,
                key=key,
                value={"boxes": [b.model_dump() for b in payload.boxes]},
                only_if_changed=False,
            )
            classes = ann.list_classes(dataset_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {**res, "classes": classes}

    @router.post("/api/annotate/dataset/{dataset_id}/null/{split}/{key:path}")
    def mark_image_null(dataset_id: str, split: str, key: str) -> dict[str, Any]:
        try:
            res = ann.mark_as_null(dataset_id, split, key)
            classes = ann.list_classes(dataset_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {**res, "classes": classes}

    @router.get("/api/annotate/dataset/{dataset_id}/classes")
    def get_classes(dataset_id: str) -> dict[str, Any]:
        try:
            return {"classes": ann.list_classes(dataset_id)}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/api/annotate/dataset/{dataset_id}/classes")
    def add_class(dataset_id: str, payload: ClassNamePayload) -> dict[str, Any]:
        try:
            ann.add_class(dataset_id, payload.name)
            return {"classes": ann.list_classes(dataset_id)}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.put("/api/annotate/dataset/{dataset_id}/classes/{class_id}")
    def rename_class(dataset_id: str, class_id: int, payload: ClassNamePayload) -> dict[str, Any]:
        try:
            changed = ann.rename_classes(dataset_id, {class_id: payload.name})
            return {"changed": changed, "classes": ann.list_classes(dataset_id)}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.delete("/api/annotate/dataset/{dataset_id}/classes/{class_id}")
    def delete_class(dataset_id: str, class_id: int) -> dict[str, Any]:
        try:
            ann.remove_unused_class(dataset_id, class_id)
            return {"classes": ann.list_classes(dataset_id)}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/api/annotate/models")
    def list_models() -> dict[str, Any]:
        trained = [
            {"label": f"⭐ {c.name} ({format_bytes(c.size_bytes)})", "path": c.path}
            for c in controller.list_checkpoints()
        ]
        return {"models": trained + _imported_models(controller) + PRETRAINED_MODELS}

    @router.post("/api/annotate/models/import")
    async def import_model(file: UploadFile = File(...)) -> dict[str, str]:
        filename = _safe_pt_name(file.filename or "")
        if not (file.filename or "").lower().endswith(".pt"):
            raise HTTPException(status_code=400, detail="Choose a .pt weight file.")
        dest_dir = controller.workspace.weights_dir / "imported"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = (dest_dir / filename).resolve()
        if dest.parent != dest_dir.resolve():
            raise HTTPException(status_code=400, detail="Invalid file name.")
        total = 0
        try:
            with dest.open("wb") as out:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > _MAX_IMPORT_BYTES:
                        raise HTTPException(status_code=400, detail="Model file is larger than 512 MB.")
                    if total == len(chunk) and not (chunk.startswith(b"PK") or chunk.startswith(b"\x80")):
                        raise HTTPException(status_code=400, detail="File is not a PyTorch .pt weight.")
                    out.write(chunk)
        except HTTPException:
            dest.unlink(missing_ok=True)
            raise
        except Exception as exc:
            dest.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Could not save the model: {exc}") from exc
        if total < 1024:
            dest.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="File is too small to be a model.")
        return {"label": f"Imported {dest.name} ({format_bytes(total)})", "path": str(dest)}

    @router.post("/api/annotate/dataset/{dataset_id}/autolabel/single")
    def autolabel_single(dataset_id: str, payload: AutoLabelSinglePayload) -> dict[str, Any]:
        try:
            res = ann.run_autolabel_single(
                dataset_id=dataset_id,
                split=payload.split,
                key=payload.key,
                weights_path=payload.weights_path,
                conf=payload.conf,
                iou=payload.iou,
                imgsz=payload.imgsz,
                auto_add_classes=payload.auto_add_classes,
            )
            data = ann.load_annotator_value(dataset_id, payload.split, payload.key)
            classes = ann.list_classes(dataset_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            **res,
            "boxes": data["value"]["boxes"],
            "classes": classes,
        }

    @router.post("/api/annotate/dataset/{dataset_id}/autolabel/batch")
    def autolabel_batch(dataset_id: str, payload: AutoLabelBatchPayload) -> dict[str, Any]:
        try:
            summary = ann.run_autolabel(
                dataset_id=dataset_id,
                weights_path=payload.weights_path,
                conf=payload.conf,
                iou=payload.iou,
                imgsz=payload.imgsz,
                scope=payload.scope,
                split_filter=payload.split,
                auto_add_classes=payload.auto_add_classes,
                apply_immediately=True,
                allow_overwrite=payload.allow_overwrite,
            )
            classes = ann.list_classes(dataset_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"summary": summary, "classes": classes}

    # ------------------------------------------------------------------ live augmentation preview
    @router.get("/aug-preview", response_class=HTMLResponse)
    def aug_preview_page() -> HTMLResponse:
        html_path = STATIC_DIR / "aug_preview.html"
        if not html_path.is_file():
            raise HTTPException(status_code=500, detail="aug_preview.html not found")
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @router.get("/api/aug/sample/{dataset_id}")
    def api_aug_sample(dataset_id: str, index: int = Query(0, ge=0)) -> dict[str, Any]:
        from urllib.parse import quote
        from core.dataset.annotation_io import parse_label_file

        _, all_entries = ann._entries(dataset_id)
        entries = [e for e in all_entries if e.split == "train"]
        if not entries:
            raise HTTPException(status_code=404, detail="No training images in dataset")
        idx = index % len(entries)
        entry = entries[idx]
        names = {int(c["id"]): str(c["name"]) for c in ann.list_classes(dataset_id)}
        boxes = parse_label_file(entry.label_path)[0] if entry.label_path.is_file() else []
        return {
            "index": idx,
            "total": len(entries),
            "filename": entry.image_path.name,
            "image_url": f"/api/annotate/dataset/{quote(dataset_id)}/image/train/{quote(entry.key)}",
            "boxes": [
                {
                    "class_id": b.class_id,
                    "class_name": names.get(b.class_id, f"class_{b.class_id}"),
                    "cx": b.cx,
                    "cy": b.cy,
                    "w": b.w,
                    "h": b.h,
                }
                for b in boxes
            ],
        }

    @router.post("/api/aug/apply/{dataset_id}")
    def api_aug_apply(dataset_id: str, payload: ApplyAugPayload) -> dict[str, Any]:
        from core.dataset.augmentor import AugmentConfig

        cfg = AugmentConfig(
            copies_per_image=max(1, min(50, int(payload.copies_per_image))),
            seed=int(payload.seed),
            ranges=payload.ranges,
        )
        try:
            rep, counts = controller.dataset_service.create_augmented_dataset(
                dataset_id,
                cfg,
                new_name=(payload.new_name or "").strip() or None,
            )
            return {
                "dataset_id": rep.dataset_id,
                "dataset_name": rep.dataset_name,
                "counts": counts,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return router
