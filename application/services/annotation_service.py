"""AnnotationService: label editing, image management, class management, AI auto-label, label import/export.

Safety rules (from PROJECT_BLUEPRINT §8 / Phase 2):
- Existing non-empty labels are never overwritten without explicit confirmation.
- Every label write first keeps a backup in workspace/temp/label_backups (single-level undo).
- AI proposals are stored separately and only written after the user accepts them.
- Destructive image operations require an explicit confirm flag.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from config.defaults import DEFAULT_MODEL_FAMILY, DEFAULT_TASK
from core.dataset.annotation_io import (
    Box,
    boxes_to_pixel_rows,
    format_label_lines,
    parse_label_file,
    pixel_rows_to_boxes,
    render_boxes,
    write_label_file,
)
from core.dataset.diagnostics import run_diagnostics
from core.dataset.entries import SPLITS, ImageEntry, collect_entries, find_entry
from core.schemas import DatasetValidationReport, utc_now_iso
from infrastructure.process.state_store import read_json, write_json_atomic
from infrastructure.storage.archive_utils import safe_extract_zip
from infrastructure.storage.workspace_manager import WorkspaceManager
from models.registry import ModelRegistry

from application.services.dataset_service import DatasetService

_SAFE_STEM = re.compile(r"^[A-Za-z0-9._\-]+$")


class AnnotationService:
    def __init__(self, workspace: WorkspaceManager, dataset_service: DatasetService, registry: ModelRegistry) -> None:
        self.workspace = workspace
        self.datasets = dataset_service
        self.registry = registry
        self.backup_root = workspace.temp_dir / "label_backups"
        self.proposal_root = workspace.temp_dir / "proposals"
        self.preview_root = workspace.temp_dir / "annotation_previews"

    # ------------------------------------------------------------------ helpers
    def _report(self, dataset_id: str) -> DatasetValidationReport:
        report = self.datasets.get_dataset(dataset_id)
        if report is None:
            raise ValueError(f"ไม่พบ Dataset '{dataset_id}'")
        return report

    def _entries(self, dataset_id: str) -> tuple[DatasetValidationReport, list[ImageEntry]]:
        report = self._report(dataset_id)
        return report, collect_entries(report)

    def _entry(self, dataset_id: str, split: str, key: str) -> tuple[DatasetValidationReport, ImageEntry]:
        report, entries = self._entries(dataset_id)
        entry = find_entry(entries, split, key)
        if entry is None:
            raise ValueError(f"ไม่พบภาพ '{split}/{key}' ใน Dataset นี้ (อาจถูกเปลี่ยนชื่อหรือลบไปแล้ว)")
        return report, entry

    def _backup_path(self, dataset_id: str, split: str, key: str) -> Path:
        return self.backup_root / dataset_id / split / (key.replace("/", "__") + ".txt")

    def _backup_label(self, dataset_id: str, entry: ImageEntry) -> None:
        backup = self._backup_path(dataset_id, entry.split, entry.key)
        backup.parent.mkdir(parents=True, exist_ok=True)
        if entry.label_path.exists():
            shutil.copy2(entry.label_path, backup)
            absent = backup.with_name(backup.name + ".absent")
            if absent.exists():
                absent.unlink()
        else:
            if backup.exists():
                backup.unlink()
            backup.with_name(backup.name + ".absent").write_text("", encoding="utf-8")

    def _image_size(self, path: Path) -> tuple[int, int]:
        with Image.open(path) as im:
            return im.size

    def _revalidate(self, dataset_id: str) -> DatasetValidationReport:
        return self.datasets.revalidate_dataset(dataset_id)

    def _write_class_names(self, dataset_id: str, names: dict[int, str]) -> None:
        report = self._report(dataset_id)
        if not report.data_yaml_path:
            raise ValueError("Dataset นี้ไม่มี data.yaml ให้แก้ไขชื่อคลาส")
        yaml_path = Path(report.data_yaml_path)
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        data["names"] = {int(k): str(v) for k, v in sorted(names.items())}
        data["nc"] = len(names)
        yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        self._revalidate(dataset_id)

    # ------------------------------------------------------------ image browser
    def list_images(self, dataset_id: str, split: str, mode: str = "all", limit: int = 300) -> list[dict[str, Any]]:
        if split not in SPLITS:
            raise ValueError(f"Split ไม่ถูกต้อง: {split}")
        _, entries = self._entries(dataset_id)
        rows: list[dict[str, Any]] = []
        for e in entries:
            if e.split != split:
                continue
            if mode == "unlabeled" and e.has_label:
                continue
            if mode == "labeled" and not e.has_label:
                continue
            boxes, _ = parse_label_file(e.label_path) if e.has_label else ([], [])
            rows.append(
                {
                    "split": e.split,
                    "key": e.key,
                    "image_path": e.image_path,
                    "has_label": e.has_label,
                    "box_count": len(boxes),
                }
            )
            if len(rows) >= limit:
                break
        return rows

    def load_editor(self, dataset_id: str, split: str, key: str) -> dict[str, Any]:
        report, entry = self._entry(dataset_id, split, key)
        w, h = self._image_size(entry.image_path)
        boxes, problems = parse_label_file(entry.label_path)
        id_to_name = {int(k): v for k, v in report.class_names.items()}
        preview = render_boxes(entry.image_path, boxes, id_to_name)
        return {
            "split": split,
            "key": key,
            "image_path": str(entry.image_path),
            "width": w,
            "height": h,
            "rows": boxes_to_pixel_rows(boxes, id_to_name, w, h),
            "preview": preview,
            "problems": problems,
            "has_label": entry.has_label,
        }

    def load_annotator_value(self, dataset_id: str, split: str, key: str) -> dict[str, Any]:
        """Load image + boxes in gradio_image_annotation format for interactive drawing."""
        report, entry = self._entry(dataset_id, split, key)
        w, h = self._image_size(entry.image_path)
        boxes, problems = parse_label_file(entry.label_path)
        id_to_name = {int(k): str(v) for k, v in report.class_names.items()}
        label_list = [id_to_name[k] for k in sorted(id_to_name)]
        annotator_boxes = []
        for b in boxes:
            x1 = max(0, round((b.cx - b.w / 2.0) * w))
            y1 = max(0, round((b.cy - b.h / 2.0) * h))
            x2 = min(w, round((b.cx + b.w / 2.0) * w))
            y2 = min(h, round((b.cy + b.h / 2.0) * h))
            if x2 > x1 and y2 > y1:
                annotator_boxes.append(
                    {
                        "xmin": x1,
                        "ymin": y1,
                        "xmax": x2,
                        "ymax": y2,
                        "label": id_to_name.get(b.class_id, f"class_{b.class_id}"),
                    }
                )
        return {
            "value": {"image": str(entry.image_path), "boxes": annotator_boxes},
            "label_list": label_list,
            "width": w,
            "height": h,
            "has_label": entry.has_label,
            "box_count": len(annotator_boxes),
            "problems": problems,
        }

    def save_annotator_value(
        self,
        dataset_id: str,
        split: str,
        key: str,
        value: dict[str, Any] | None,
        only_if_changed: bool = False,
    ) -> dict[str, Any]:
        """Save boxes directly from gradio_image_annotation. Auto-adds new class names if typed by user."""
        if not value or not isinstance(value, dict):
            raise ValueError("ไม่พบข้อมูลภาพ/กรอบที่จะบันทึก")
        report, entry = self._entry(dataset_id, split, key)
        w, h = self._image_size(entry.image_path)
        names = {int(k): str(v) for k, v in report.class_names.items()}
        name_to_id = {v: k for k, v in names.items()}

        raw_boxes = value.get("boxes") or []
        added_classes: list[str] = []
        parsed_boxes: list[Box] = []
        for b in raw_boxes:
            if not isinstance(b, dict):
                continue
            label = str(b.get("label") or "").strip()
            if not label:
                label = names[min(names)] if names else "object"
            if label not in name_to_id:
                new_id = max(names.keys(), default=-1) + 1
                names[new_id] = label
                name_to_id[label] = new_id
                added_classes.append(label)
            x1 = min(max(float(b["xmin"]), 0.0), float(w))
            y1 = min(max(float(b["ymin"]), 0.0), float(h))
            x2 = min(max(float(b["xmax"]), 0.0), float(w))
            y2 = min(max(float(b["ymax"]), 0.0), float(h))
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue
            parsed_boxes.append(
                Box(
                    class_id=name_to_id[label],
                    cx=((x1 + x2) / 2.0) / w,
                    cy=((y1 + y2) / 2.0) / h,
                    w=(x2 - x1) / w,
                    h=(y2 - y1) / h,
                )
            )

        if added_classes:
            self._write_class_names(dataset_id, names)

        existing, _ = parse_label_file(entry.label_path)
        changed = format_label_lines(existing) != format_label_lines(parsed_boxes)
        if only_if_changed and not changed:
            return {"box_count": len(parsed_boxes), "changed": False, "added_classes": added_classes}
        if changed or not entry.has_label:
            self._backup_label(dataset_id, entry)
            write_label_file(entry.label_path, parsed_boxes)
        return {"box_count": len(parsed_boxes), "changed": changed, "added_classes": added_classes}

    def mark_as_null(self, dataset_id: str, split: str, key: str) -> dict[str, Any]:
        """Mark an image as a background/null image (writes an empty .txt label file)."""
        _, entry = self._entry(dataset_id, split, key)
        self._backup_label(dataset_id, entry)
        write_label_file(entry.label_path, [])
        return {"split": split, "key": key, "box_count": 0}

    def run_autolabel_single(
        self,
        dataset_id: str,
        split: str,
        key: str,
        weights_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        auto_add_classes: bool = True,
    ) -> dict[str, Any]:
        """Run AI detection on a single image and write its label immediately (with backup)."""
        weights_str = str(weights_path).strip() if weights_path else ""
        known_pretrained = {"yolo11n.pt", "yolo11s.pt", "yolo11m.pt"}
        if not weights_str or (weights_str not in known_pretrained and not Path(weights_str).exists()):
            raise ValueError("กรุณาเลือกโมเดล (.pt) ที่ต้องการใช้")

        report, entry = self._entry(dataset_id, split, key)
        adapter = self.registry.get(DEFAULT_MODEL_FAMILY, DEFAULT_TASK)
        raw = adapter.predict_boxes_batch(
            weights_path=Path(weights_str),
            image_paths=[entry.image_path],
            conf=conf,
            iou=iou,
            imgsz=imgsz,
        )
        names = {int(k): str(v) for k, v in report.class_names.items()}
        name_to_id = {v: k for k, v in names.items()}
        added_classes: list[str] = []
        boxes: list[Box] = []

        if raw:
            _, w, h, dets = raw[0]
            for d in dets:
                if d.class_name not in name_to_id:
                    if auto_add_classes:
                        new_id = max(names.keys(), default=-1) + 1
                        names[new_id] = d.class_name
                        name_to_id[d.class_name] = new_id
                        added_classes.append(d.class_name)
                    else:
                        continue
                x1 = min(max(d.x1, 0.0), float(w))
                y1 = min(max(d.y1, 0.0), float(h))
                x2 = min(max(d.x2, 0.0), float(w))
                y2 = min(max(d.y2, 0.0), float(h))
                if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                    continue
                boxes.append(
                    Box(
                        class_id=name_to_id[d.class_name],
                        cx=((x1 + x2) / 2.0) / w,
                        cy=((y1 + y2) / 2.0) / h,
                        w=(x2 - x1) / w,
                        h=(y2 - y1) / h,
                    )
                )

        if added_classes:
            self._write_class_names(dataset_id, names)
        self._backup_label(dataset_id, entry)
        write_label_file(entry.label_path, boxes)
        return {"box_count": len(boxes), "added_classes": added_classes}

    def save_boxes(
        self,
        dataset_id: str,
        split: str,
        key: str,
        rows: list[list],
        allow_overwrite: bool,
    ) -> dict[str, Any]:
        report, entry = self._entry(dataset_id, split, key)
        w, h = self._image_size(entry.image_path)
        name_to_id = {str(v): int(k) for k, v in report.class_names.items()}
        boxes = pixel_rows_to_boxes(rows, w, h, name_to_id)

        existing, _ = parse_label_file(entry.label_path)
        changed = format_label_lines(existing) != format_label_lines(boxes)
        if entry.has_label and existing and changed and not allow_overwrite:
            raise ValueError(
                "ภาพนี้มี Label เดิมอยู่แล้ว — ติ๊ก 'ยืนยันการเขียนทับ Label เดิม' ก่อนบันทึก "
                "(ระบบจะเก็บสำรองไว้ให้กดย้อนกลับได้)"
            )
        if changed or not entry.has_label:
            self._backup_label(dataset_id, entry)
            write_label_file(entry.label_path, boxes)
        return {"box_count": len(boxes), "changed": changed}

    def undo_last_save(self, dataset_id: str, split: str, key: str) -> str:
        _, entry = self._entry(dataset_id, split, key)
        backup = self._backup_path(dataset_id, split, key)
        absent = backup.with_name(backup.name + ".absent")
        if absent.exists():
            if entry.label_path.exists():
                entry.label_path.unlink()
            absent.unlink()
            return "ย้อนกลับแล้ว: ลบ Label ที่เพิ่งสร้าง"
        if backup.exists():
            shutil.copy2(backup, entry.label_path)
            backup.unlink()
            return "ย้อนกลับแล้ว: คืน Label ก่อนหน้า"
        raise ValueError("ไม่มีประวัติการบันทึกให้ย้อนกลับ (ย้อนกลับได้หนึ่งครั้งต่อการบันทึก)")

    def rename_image(self, dataset_id: str, split: str, key: str, new_stem: str) -> str:
        new_stem = new_stem.strip()
        if not _SAFE_STEM.match(new_stem) or ".." in new_stem:
            raise ValueError("ชื่อใหม่ใช้ได้เฉพาะ a-z, A-Z, 0-9, จุด, ขีดกลาง และขีดล่าง")
        report, entry = self._entry(dataset_id, split, key)
        new_img = entry.image_path.with_name(new_stem + entry.image_path.suffix)
        new_lbl = entry.label_path.with_name(new_stem + ".txt")
        if new_img.exists() or (entry.has_label and new_lbl.exists()):
            raise ValueError(f"มีไฟล์ชื่อ '{new_stem}' อยู่แล้วในโฟลเดอร์เดียวกัน")
        entry.image_path.rename(new_img)
        if entry.has_label:
            entry.label_path.rename(new_lbl)
        self._revalidate(dataset_id)
        new_key = (key.rsplit("/", 1)[0] + "/" if "/" in key else "") + new_stem
        return new_key

    def delete_image(self, dataset_id: str, split: str, key: str, confirm: bool) -> None:
        if not confirm:
            raise ValueError("ติ๊กยืนยันการลบก่อน (การลบภาพและ Label ไม่สามารถย้อนกลับได้)")
        _, entry = self._entry(dataset_id, split, key)
        self._backup_label(dataset_id, entry)  # keep a copy of the label for undo
        entry.image_path.unlink()
        if entry.label_path.exists():
            entry.label_path.unlink()
        self._revalidate(dataset_id)

    # ------------------------------------------------------------------ classes
    def list_classes(self, dataset_id: str) -> list[dict[str, Any]]:
        report, entries = self._entries(dataset_id)
        counts: dict[int, int] = {}
        for e in entries:
            if not e.has_label:
                continue
            boxes, _ = parse_label_file(e.label_path)
            for b in boxes:
                counts[b.class_id] = counts.get(b.class_id, 0) + 1
        return [
            {"id": int(cid), "name": str(name), "count": counts.get(int(cid), 0)}
            for cid, name in sorted(report.class_names.items(), key=lambda x: int(x[0]))
        ]

    def add_class(self, dataset_id: str, name: str) -> None:
        name = (name or "").strip()
        if not name:
            raise ValueError("กรุณาระบุชื่อคลาส")
        report = self._report(dataset_id)
        names = {int(k): str(v) for k, v in report.class_names.items()}
        if name in names.values():
            raise ValueError(f"มีคลาส '{name}' อยู่แล้ว")
        new_id = max(names.keys(), default=-1) + 1
        names[new_id] = name
        self._write_class_names(dataset_id, names)

    def rename_classes(self, dataset_id: str, new_names: dict[int, str]) -> int:
        report = self._report(dataset_id)
        names = {int(k): str(v) for k, v in report.class_names.items()}
        changed = 0
        for cid, new in new_names.items():
            new = (new or "").strip()
            if int(cid) not in names:
                raise ValueError(f"ไม่พบคลาส id {cid}")
            if not new:
                raise ValueError("ชื่อคลาสต้องไม่ว่าง")
            if names[int(cid)] != new:
                names[int(cid)] = new
                changed += 1
        if len(set(names.values())) != len(names):
            raise ValueError("ชื่อคลาสซ้ำกัน")
        if changed:
            self._write_class_names(dataset_id, names)
        return changed

    def remove_last_class(self, dataset_id: str, class_id: int) -> None:
        """Only the highest-id class can be removed, and only when no box uses it (keeps other ids stable)."""
        report = self._report(dataset_id)
        names = {int(k): str(v) for k, v in report.class_names.items()}
        if class_id not in names:
            raise ValueError(f"ไม่พบคลาส id {class_id}")
        if class_id != max(names):
            raise ValueError("ลบได้เฉพาะคลาสที่มี id สูงสุดเท่านั้น (เพื่อไม่ให้ id ของคลาสอื่นเปลี่ยน)")
        used = {row["id"]: row["count"] for row in self.list_classes(dataset_id)}
        if used.get(class_id, 0) > 0:
            raise ValueError(f"คลาส '{names[class_id]}' ยังมี Box ใช้งานอยู่ {used[class_id]} กรอบ")
        del names[class_id]
        self._write_class_names(dataset_id, names)

    def remove_unused_class(self, dataset_id: str, class_id: int) -> None:
        """Delete ANY unused class (count == 0) and safely re-index remaining classes + label files."""
        report, entries = self._entries(dataset_id)
        names = {int(k): str(v) for k, v in report.class_names.items()}
        if class_id not in names:
            raise ValueError(f"ไม่พบคลาส id {class_id}")
        used = {row["id"]: row["count"] for row in self.list_classes(dataset_id)}
        if used.get(class_id, 0) > 0:
            raise ValueError(f"คลาส '{names[class_id]}' ยังมี Box ใช้งานอยู่ {used[class_id]} กรอบ (ต้องลบกรอบก่อน)")

        remaining_ids = [cid for cid in sorted(names) if cid != class_id]
        id_map = {old_id: new_id for new_id, old_id in enumerate(remaining_ids)}
        new_names = {id_map[old_id]: names[old_id] for old_id in remaining_ids}

        # Remap class IDs in all label files when any id shifted
        if any(old_id != new_id for old_id, new_id in id_map.items()):
            for e in entries:
                if not e.has_label:
                    continue
                boxes, _ = parse_label_file(e.label_path)
                if not boxes:
                    continue
                updated = [
                    Box(class_id=id_map.get(b.class_id, b.class_id), cx=b.cx, cy=b.cy, w=b.w, h=b.h)
                    for b in boxes
                ]
                write_label_file(e.label_path, updated)

        self._write_class_names(dataset_id, new_names)

    # ----------------------------------------------------------------- AI label
    def run_autolabel(
        self,
        dataset_id: str,
        weights_path: str | Path,
        conf: float,
        iou: float,
        imgsz: int,
        scope: str = "unlabeled",
        split_filter: str = "all",
        auto_add_classes: bool = True,
        apply_immediately: bool = False,
        allow_overwrite: bool = False,
    ) -> dict[str, Any]:
        weights_str = str(weights_path).strip() if weights_path else ""
        known_pretrained = {"yolo11n.pt", "yolo11s.pt", "yolo11m.pt"}
        if not weights_str or (weights_str not in known_pretrained and not Path(weights_str).exists()):
            raise ValueError("กรุณาเลือกโมเดล (.pt) ที่ต้องการใช้ทำ Auto-Label")
        if scope not in {"unlabeled", "all"}:
            raise ValueError("scope ไม่ถูกต้อง")
        report, entries = self._entries(dataset_id)

        targets = [
            e
            for e in entries
            if (scope == "all" or not e.has_label) and (split_filter == "all" or e.split == split_filter)
        ]
        if not targets:
            raise ValueError("ไม่มีภาพที่ตรงกับเงื่อนไขสำหรับ Auto-Label")

        adapter = self.registry.get(DEFAULT_MODEL_FAMILY, DEFAULT_TASK)
        raw = adapter.predict_boxes_batch(
            weights_path=Path(weights_str),
            image_paths=[e.image_path for e in targets],
            conf=conf,
            iou=iou,
            imgsz=imgsz,
        )
        by_path = {str(e.image_path): e for e in targets}
        names = {int(k): str(v) for k, v in report.class_names.items()}
        name_to_id = {v: k for k, v in names.items()}
        added_classes: list[str] = []

        items: list[dict[str, Any]] = []
        total_boxes = 0
        unknown_boxes = 0
        with_detections = 0
        for path_str, w, h, dets in raw:
            entry = by_path.get(path_str)
            if entry is None:
                continue
            boxes: list[dict[str, Any]] = []
            for d in dets:
                if d.class_name not in name_to_id:
                    if auto_add_classes:
                        new_id = max(names.keys(), default=-1) + 1
                        names[new_id] = d.class_name
                        name_to_id[d.class_name] = new_id
                        added_classes.append(d.class_name)
                    else:
                        unknown_boxes += 1
                        continue
                x1, y1, x2, y2 = (
                    min(max(d.x1, 0.0), w),
                    min(max(d.y1, 0.0), h),
                    min(max(d.x2, 0.0), w),
                    min(max(d.y2, 0.0), h),
                )
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                boxes.append(
                    {
                        "class_id": name_to_id[d.class_name],
                        "cx": ((x1 + x2) / 2.0) / w,
                        "cy": ((y1 + y2) / 2.0) / h,
                        "w": (x2 - x1) / w,
                        "h": (y2 - y1) / h,
                        "conf": round(d.confidence, 4),
                    }
                )
            if boxes:
                with_detections += 1
            total_boxes += len(boxes)
            items.append(
                {
                    "split": entry.split,
                    "key": entry.key,
                    "image_path": str(entry.image_path),
                    "has_label": entry.has_label,
                    "width": w,
                    "height": h,
                    "boxes": boxes,
                }
            )

        if added_classes:
            self._write_class_names(dataset_id, names)

        proposal = {
            "dataset_id": dataset_id,
            "weights": weights_str,
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "scope": scope,
            "created_at": utc_now_iso(),
            "items": items,
            "summary": {
                "images_scanned": len(items),
                "images_with_detections": with_detections,
                "total_boxes": total_boxes,
                "unknown_class_boxes": unknown_boxes,
                "added_classes": added_classes,
            },
        }
        write_json_atomic(self.proposal_root / f"{dataset_id}.json", proposal)

        if apply_immediately:
            applied = self.accept_proposals(dataset_id, allow_overwrite=allow_overwrite)
            proposal["summary"].update(applied)
        return proposal["summary"]

    def get_proposals(self, dataset_id: str) -> dict[str, Any] | None:
        return read_json(self.proposal_root / f"{dataset_id}.json")

    def build_proposal_previews(self, dataset_id: str, max_items: int = 24) -> list[tuple[str, str]]:
        proposal = self.get_proposals(dataset_id)
        if not proposal:
            return []
        report = self._report(dataset_id)
        id_to_name = {int(k): str(v) for k, v in report.class_names.items()}
        gallery: list[tuple[str, str]] = []
        out_dir = self.preview_root / dataset_id
        for item in [i for i in proposal["items"] if i["boxes"]][:max_items]:
            boxes = [Box(b["class_id"], b["cx"], b["cy"], b["w"], b["h"], b["conf"]) for b in item["boxes"]]
            out = out_dir / f"{item['split']}__{item['key'].replace('/', '__')}.jpg"
            render_boxes(Path(item["image_path"]), boxes, id_to_name, out_path=out)
            gallery.append((str(out), f"{item['split']}/{item['key']} ({len(boxes)} กรอบ)"))
        return gallery

    def accept_proposals(self, dataset_id: str, allow_overwrite: bool) -> dict[str, int]:
        proposal = self.get_proposals(dataset_id)
        if not proposal:
            raise ValueError("ไม่มีผล Auto-Label ที่รอการยืนยัน")
        _, entries = self._entries(dataset_id)
        written = skipped_existing = skipped_empty = missing = 0
        for item in proposal["items"]:
            entry = find_entry(entries, item["split"], item["key"])
            if entry is None:
                missing += 1
                continue
            if not item["boxes"]:
                skipped_empty += 1
                continue
            existing, _ = parse_label_file(entry.label_path)
            if existing and not allow_overwrite:
                skipped_existing += 1
                continue
            boxes = [Box(b["class_id"], b["cx"], b["cy"], b["w"], b["h"]) for b in item["boxes"]]
            self._backup_label(dataset_id, entry)
            write_label_file(entry.label_path, boxes)
            written += 1
        self._revalidate(dataset_id)
        (self.proposal_root / f"{dataset_id}.json").unlink(missing_ok=True)
        return {
            "written": written,
            "skipped_existing": skipped_existing,
            "skipped_no_detection": skipped_empty,
            "missing_image": missing,
        }

    def discard_proposals(self, dataset_id: str) -> None:
        (self.proposal_root / f"{dataset_id}.json").unlink(missing_ok=True)

    # ------------------------------------------------------- diagnostics & split
    def diagnostics(self, dataset_id: str) -> dict[str, Any]:
        report, entries = self._entries(dataset_id)
        return run_diagnostics(entries, {int(k): v for k, v in report.class_names.items()})

    def create_split_dataset(
        self,
        dataset_id: str,
        train_pct: float,
        val_pct: float,
        test_pct: float,
        seed: int,
        drop_duplicates: bool,
        new_name: str | None,
    ) -> tuple[DatasetValidationReport, dict[str, Any]]:
        report = self._report(dataset_id)
        new_report, counts = self.datasets.create_split_dataset(
            dataset_id=dataset_id,
            train_pct=train_pct,
            val_pct=val_pct,
            test_pct=test_pct,
            seed=seed,
            drop_duplicates=drop_duplicates,
            new_name=new_name,
        )
        return new_report, counts

    # ------------------------------------------------------ import / export
    def export_labels(self, dataset_id: str) -> Path:
        report, entries = self._entries(dataset_id)
        out_dir = self.workspace.exports_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = out_dir / f"{dataset_id}_labels_{stamp}.zip"
        import zipfile

        names = {int(k): str(v) for k, v in report.class_names.items()}
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            if report.data_yaml_path and Path(report.data_yaml_path).exists():
                zf.write(report.data_yaml_path, arcname="data.yaml")
            zf.writestr("classes.txt", "\n".join(names[k] for k in sorted(names)) + "\n")
            for e in entries:
                if e.has_label:
                    zf.write(e.label_path, arcname=f"labels/{e.split}/{e.key}.txt")
        return zip_path

    def import_labels(self, dataset_id: str, zip_path: str | Path, allow_overwrite: bool) -> dict[str, int]:
        if not zip_path:
            raise ValueError("กรุณาเลือกไฟล์ ZIP ของ Label")
        _, entries = self._entries(dataset_id)
        lookup = {(e.split, e.key): e for e in entries}
        tmp = self.workspace.temp_dir / "import_labels" / uuid.uuid4().hex[:10]
        result = {"imported": 0, "unchanged": 0, "skipped_existing": 0, "invalid": 0, "unmatched": 0}
        try:
            root = safe_extract_zip(Path(zip_path), tmp)
            for txt in sorted(root.rglob("*.txt")):
                if txt.name == "classes.txt":
                    continue
                parts = txt.relative_to(root).parts
                split = None
                for idx, part in enumerate(parts[:-1]):
                    if part.lower() == "labels" and idx + 1 < len(parts) - 1 and parts[idx + 1] in SPLITS:
                        split = parts[idx + 1]
                        key = Path(*parts[idx + 2 :]).with_suffix("").as_posix()
                        break
                if split is None:
                    result["unmatched"] += 1
                    continue
                entry = lookup.get((split, key))
                if entry is None:
                    result["unmatched"] += 1
                    continue
                boxes, problems = parse_label_file(txt)
                if problems:
                    result["invalid"] += 1
                    continue
                existing, _ = parse_label_file(entry.label_path)
                if existing and format_label_lines(existing) == format_label_lines(boxes):
                    result["unchanged"] += 1
                    continue
                if existing and not allow_overwrite:
                    result["skipped_existing"] += 1
                    continue
                self._backup_label(dataset_id, entry)
                write_label_file(entry.label_path, boxes)
                result["imported"] += 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self._revalidate(dataset_id)
        return result
