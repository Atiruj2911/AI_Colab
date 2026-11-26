"""DatasetService: upload, safe extraction, inspection, validation, and demo dataset creation."""

from __future__ import annotations

import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from config.app_config import AppConfig
from config.defaults import ALLOWED_IMAGE_EXTENSIONS
from core.dataset.demo_generator import generate_demo_yolo_dataset
from core.dataset.inspector import render_sample_previews
from core.dataset.validator import validate_yolo_dataset
from core.schemas import DatasetValidationReport
from infrastructure.process.state_store import read_json, write_json_atomic
from infrastructure.storage.archive_utils import copy_uploaded_files_into_dir, safe_extract_zip
from infrastructure.storage.workspace_manager import WorkspaceManager
from utils.validation import sanitize_identifier

UNSUPPORTED_ARCHIVE_EXTENSIONS = {".rar", ".7z", ".tar", ".gz", ".tgz", ".bz2", ".xz"}


class DatasetService:
    """Manages uploaded and validated YOLO datasets inside workspace/datasets."""

    def __init__(self, config: AppConfig, workspace: WorkspaceManager) -> None:
        self.config = config
        self.workspace = workspace

    def _meta_path(self, dataset_id: str) -> Path:
        return self.workspace.datasets_dir / dataset_id / "dataset_meta.json"

    def ingest_upload(self, uploaded_files: list[str | Path], dataset_name: str | None = None) -> DatasetValidationReport:
        """Accept either a single ZIP archive or a set of YOLO files, extract safely, and validate."""
        paths = [Path(p) for p in uploaded_files if p]
        if not paths:
            raise ValueError("กรุณาเลือกไฟล์ ZIP หรือไฟล์ Dataset ก่อนอัปโหลด")

        unsupported = [p for p in paths if p.suffix.lower() in UNSUPPORTED_ARCHIVE_EXTENSIONS]
        if unsupported:
            names = ", ".join(p.name for p in unsupported)
            raise ValueError(
                f"ไม่รองรับไฟล์ {names} กรุณาแตกไฟล์ แล้วบีบอัดใหม่เป็น .zip ก่อนอัปโหลด"
            )

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        inferred_name = dataset_name.strip() if dataset_name and dataset_name.strip() else paths[0].stem
        safe_name = sanitize_identifier(inferred_name, fallback="dataset")
        dataset_id = f"{safe_name}_{stamp}"

        upload_backup_dir = self.workspace.uploads_dir / dataset_id
        upload_backup_dir.mkdir(parents=True, exist_ok=True)
        dataset_container = self.workspace.datasets_dir / dataset_id
        extracted_dir = dataset_container / "raw"
        extracted_dir.mkdir(parents=True, exist_ok=True)

        if len(paths) == 1 and paths[0].suffix.lower() == ".zip":
            # Preserve original uploaded archive in uploads/
            saved_zip = upload_backup_dir / paths[0].name
            shutil.copy2(paths[0], saved_zip)
            dataset_root = safe_extract_zip(
                zip_path=saved_zip,
                dest_dir=extracted_dir,
                max_archive_mb=self.config.max_upload_mb,
                max_extracted_mb=self.config.max_extracted_mb,
                max_files=self.config.max_archive_files,
            )
        else:
            dataset_root = copy_uploaded_files_into_dir(paths, extracted_dir)

        report = validate_yolo_dataset(
            dataset_id=dataset_id,
            dataset_name=safe_name,
            dataset_root=dataset_root,
            resolved_yaml_dir=dataset_container,
        )

        # Reject uploads that cannot possibly be a YOLO dataset. The original archive
        # stays in uploads/ (never silently discarded); only the extracted copy is removed.
        fatal_codes = {"MISSING_DATA_YAML", "INVALID_YAML", "NO_IMAGES_FOUND"}
        fatal = [e for e in report.errors if e.code in fatal_codes]
        if not fatal and report.total_images == 0:
            fatal = [e for e in report.errors if e.severity == "error"]
        if fatal:
            shutil.rmtree(dataset_container, ignore_errors=True)
            details = "\n".join(f"• {e.message}" + (f" ({e.hint})" if e.hint else "") for e in fatal)
            raise ValueError(
                f"ไม่สามารถนำเข้า '{paths[0].name}' ได้ — โครงสร้างไม่ใช่ YOLO Dataset:\n{details}\n"
                f"ไฟล์ต้นฉบับเก็บไว้ที่ uploads/{dataset_id}/"
            )

        write_json_atomic(self._meta_path(dataset_id), report.to_dict())
        return report

    def create_demo_dataset(self, train_count: int = 12, val_count: int = 4) -> DatasetValidationReport:
        """Generate a lightweight synthetic YOLO Detection dataset for immediate testing."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_id = f"demo_shapes_{stamp}"
        dataset_container = self.workspace.datasets_dir / dataset_id
        raw_dir = dataset_container / "raw"
        zip_path = self.workspace.uploads_dir / f"{dataset_id}.zip"

        generate_demo_yolo_dataset(
            output_dir=raw_dir,
            train_count=train_count,
            val_count=val_count,
            also_create_zip=zip_path,
        )
        report = validate_yolo_dataset(
            dataset_id=dataset_id,
            dataset_name="demo_shapes",
            dataset_root=raw_dir,
            resolved_yaml_dir=dataset_container,
        )
        write_json_atomic(self._meta_path(dataset_id), report.to_dict())
        return report

    def list_datasets(self) -> list[DatasetValidationReport]:
        reports: list[DatasetValidationReport] = []
        for meta_file in self.workspace.datasets_dir.glob("*/dataset_meta.json"):
            data = read_json(meta_file)
            if not data:
                continue
            try:
                reports.append(DatasetValidationReport.from_dict(data))
            except Exception:
                continue
        reports.sort(key=lambda r: r.validated_at, reverse=True)
        return reports

    def get_dataset(self, dataset_id: str) -> DatasetValidationReport | None:
        data = read_json(self._meta_path(dataset_id))
        if not data:
            return None
        return DatasetValidationReport.from_dict(data)

    def revalidate_dataset(self, dataset_id: str) -> DatasetValidationReport:
        existing = self.get_dataset(dataset_id)
        if existing is None:
            raise ValueError(f"ไม่พบ Dataset '{dataset_id}'")
        dataset_container = self.workspace.datasets_dir / dataset_id
        report = validate_yolo_dataset(
            dataset_id=existing.dataset_id,
            dataset_name=existing.dataset_name,
            dataset_root=Path(existing.dataset_root),
            resolved_yaml_dir=dataset_container,
        )
        write_json_atomic(self._meta_path(dataset_id), report.to_dict())
        return report

    def create_split_dataset(
        self,
        dataset_id: str,
        train_pct: float,
        val_pct: float,
        test_pct: float,
        seed: int,
        drop_duplicates: bool = True,
        new_name: str | None = None,
    ) -> tuple[DatasetValidationReport, dict[str, int]]:
        """Create a NEW dataset folder re-split with a seed. Corrupt images are excluded; duplicates optionally."""
        from core.dataset.diagnostics import find_corrupt, find_duplicate_groups
        from core.dataset.entries import collect_entries
        from core.dataset.splitter import split_entries

        source = self.get_dataset(dataset_id)
        if source is None:
            raise ValueError(f"ไม่พบ Dataset '{dataset_id}'")
        entries = collect_entries(source)
        if not entries:
            raise ValueError("Dataset ต้นทางไม่มีภาพ")

        corrupt = set(find_corrupt(entries))
        excluded_dup: set[str] = set()
        if drop_duplicates:
            for group in find_duplicate_groups([e for e in entries if e.display_id not in corrupt]):
                excluded_dup.update(group[1:])  # keep the first occurrence of each group
        kept = [e for e in entries if e.display_id not in corrupt and e.display_id not in excluded_dup]

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = sanitize_identifier(new_name or f"{source.dataset_name}_split", fallback="split")
        new_id = f"{base}_{stamp}"
        container = self.workspace.datasets_dir / new_id
        out_root = container / "raw"
        counts = split_entries(
            entries=kept,
            class_names={int(k): v for k, v in source.class_names.items()},
            out_root=out_root,
            train_pct=train_pct,
            val_pct=val_pct,
            test_pct=test_pct,
            seed=seed,
            source_id=dataset_id,
        )
        report = validate_yolo_dataset(
            dataset_id=new_id,
            dataset_name=base,
            dataset_root=out_root,
            resolved_yaml_dir=container,
        )
        write_json_atomic(self._meta_path(new_id), report.to_dict())
        counts["excluded_corrupt"] = len(corrupt)
        counts["excluded_duplicate"] = len(excluded_dup)
        return report, counts

    # ------------------------------------------------- Mode 1: Images-only upload
    def ingest_images_only(
        self,
        uploaded_files: list[str | Path],
        dataset_name: str | None = None,
        class_names: list[str] | None = None,
    ) -> DatasetValidationReport:
        """Create a new YOLO dataset from unlabeled images (ZIP or image files), ready for annotation."""
        paths = [Path(p) for p in uploaded_files if p]
        if not paths:
            raise ValueError("กรุณาเลือกไฟล์ภาพหรือไฟล์ ZIP ที่มีภาพก่อน")

        unsupported = [p for p in paths if p.suffix.lower() in UNSUPPORTED_ARCHIVE_EXTENSIONS]
        if unsupported:
            names = ", ".join(p.name for p in unsupported)
            raise ValueError(f"ไม่รองรับไฟล์ {names} กรุณาบีบอัดเป็น .zip หรืออัปโหลดไฟล์ภาพโดยตรง")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        inferred = dataset_name.strip() if dataset_name and dataset_name.strip() else paths[0].stem
        safe_name = sanitize_identifier(inferred, fallback="images_dataset")
        dataset_id = f"{safe_name}_{stamp}"

        tmp_extract = self.workspace.temp_dir / "img_only_extract" / uuid.uuid4().hex[:10]
        tmp_extract.mkdir(parents=True, exist_ok=True)
        try:
            image_sources: list[Path] = []
            for p in paths:
                if p.suffix.lower() == ".zip":
                    extracted = safe_extract_zip(
                        zip_path=p,
                        dest_dir=tmp_extract / p.stem,
                        max_archive_mb=self.config.max_upload_mb,
                        max_extracted_mb=self.config.max_extracted_mb,
                        max_files=self.config.max_archive_files,
                    )
                    image_sources.extend(
                        f for f in sorted(extracted.rglob("*")) if f.is_file() and f.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
                    )
                elif p.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
                    image_sources.append(p)

            if not image_sources:
                raise ValueError("ไม่พบไฟล์ภาพ (.jpg, .png, .webp, .bmp) ในไฟล์ที่อัปโหลด")

            container = self.workspace.datasets_dir / dataset_id
            raw_dir = container / "raw"
            img_train = raw_dir / "images" / "train"
            lbl_train = raw_dir / "labels" / "train"
            img_val = raw_dir / "images" / "val"
            lbl_val = raw_dir / "labels" / "val"
            for d in (img_train, lbl_train, img_val, lbl_val):
                d.mkdir(parents=True, exist_ok=True)

            used_stems: set[str] = set()
            for src in image_sources:
                stem = sanitize_identifier(src.stem, fallback="img")
                candidate = stem
                idx = 1
                while candidate.lower() in used_stems:
                    candidate = f"{stem}_{idx}"
                    idx += 1
                used_stems.add(candidate.lower())
                shutil.copy2(src, img_train / f"{candidate}{src.suffix.lower()}")

            cleaned_classes = [c.strip() for c in (class_names or []) if c and c.strip()]
            # deduplicate preserving order (allow empty — user adds first class when drawing!)
            unique_classes = list(dict.fromkeys(cleaned_classes))
            yaml_data = {
                "path": ".",
                "train": "images/train",
                "val": "images/val",
                "nc": len(unique_classes),
                "names": {i: name for i, name in enumerate(unique_classes)},
            }
            (raw_dir / "data.yaml").write_text(
                yaml.safe_dump(yaml_data, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
        finally:
            shutil.rmtree(tmp_extract, ignore_errors=True)

        report = validate_yolo_dataset(
            dataset_id=dataset_id,
            dataset_name=safe_name,
            dataset_root=raw_dir,
            resolved_yaml_dir=container,
        )
        write_json_atomic(self._meta_path(dataset_id), report.to_dict())
        return report

    # ------------------------------------------------- Mode 2: Custom folder mapping
    def _stage_dir(self, stage_id: str) -> Path:
        return self.workspace.temp_dir / "staged_uploads" / stage_id

    def stage_custom_upload(self, uploaded_files: list[str | Path]) -> dict[str, Any]:
        """Extract uploaded archive/files to a staging folder and inspect subdirectories for manual mapping."""
        paths = [Path(p) for p in uploaded_files if p]
        if not paths:
            raise ValueError("กรุณาเลือกไฟล์ ZIP หรือไฟล์ Dataset ก่อน")
        unsupported = [p for p in paths if p.suffix.lower() in UNSUPPORTED_ARCHIVE_EXTENSIONS]
        if unsupported:
            names = ", ".join(p.name for p in unsupported)
            raise ValueError(f"ไม่รองรับไฟล์ {names} กรุณาบีบอัดเป็น .zip ก่อน")

        stage_id = f"stage_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        stage_root = self._stage_dir(stage_id)
        stage_root.mkdir(parents=True, exist_ok=True)

        if len(paths) == 1 and paths[0].suffix.lower() == ".zip":
            root = safe_extract_zip(
                zip_path=paths[0],
                dest_dir=stage_root,
                max_archive_mb=self.config.max_upload_mb,
                max_extracted_mb=self.config.max_extracted_mb,
                max_files=self.config.max_archive_files,
            )
        else:
            root = copy_uploaded_files_into_dir(paths, stage_root)

        image_counts: dict[str, int] = {}
        label_counts: dict[str, int] = {}
        suggested_classes: list[str] = []

        for f in sorted(root.rglob("*")):
            if not f.is_file():
                continue
            rel_dir = f.parent.relative_to(root).as_posix()
            ext = f.suffix.lower()
            if ext in ALLOWED_IMAGE_EXTENSIONS:
                image_counts[rel_dir] = image_counts.get(rel_dir, 0) + 1
            elif ext == ".txt" and f.name.lower() != "classes.txt":
                label_counts[rel_dir] = label_counts.get(rel_dir, 0) + 1
            elif f.name.lower() == "classes.txt" and not suggested_classes:
                lines = [ln.strip() for ln in f.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
                suggested_classes = lines
            elif ext in {".yaml", ".yml"} and not suggested_classes:
                try:
                    parsed = yaml.safe_load(f.read_text(encoding="utf-8", errors="replace"))
                    if isinstance(parsed, dict) and "names" in parsed:
                        raw_names = parsed["names"]
                        if isinstance(raw_names, list):
                            suggested_classes = [str(x).strip() for x in raw_names if str(x).strip()]
                        elif isinstance(raw_names, dict):
                            suggested_classes = [str(v).strip() for _, v in sorted(raw_names.items(), key=lambda x: int(x[0]))]
                except Exception:
                    pass

        return {
            "stage_id": stage_id,
            "root_dir": str(root),
            "default_name": sanitize_identifier(paths[0].stem, fallback="custom_dataset"),
            "image_dirs": sorted(image_counts.items(), key=lambda x: (-x[1], x[0])),
            "label_dirs": sorted(label_counts.items(), key=lambda x: (-x[1], x[0])),
            "suggested_classes": suggested_classes,
        }

    def preview_custom_mapping(
        self, stage_id: str, root_dir: str | Path, image_rel_dir: str, label_rel_dir: str | None
    ) -> dict[str, Any]:
        """Preview how many images and YOLO .txt labels match by filename stem."""
        root = Path(root_dir)
        img_dir = root if image_rel_dir in ("", ".") else root / image_rel_dir
        if not img_dir.exists():
            raise ValueError(f"ไม่พบโฟลเดอร์ภาพ: {image_rel_dir}")

        imgs = [f for f in sorted(img_dir.iterdir()) if f.is_file() and f.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS]
        lbl_by_stem: dict[str, Path] = {}
        if label_rel_dir and label_rel_dir != "(ไม่มี Label)":
            lbl_dir = root if label_rel_dir in ("", ".") else root / label_rel_dir
            if lbl_dir.exists():
                for f in sorted(lbl_dir.iterdir()):
                    if f.is_file() and f.suffix.lower() == ".txt" and f.name.lower() != "classes.txt":
                        lbl_by_stem[f.stem] = f

        matched = [f.name for f in imgs if f.stem in lbl_by_stem]
        unmatched_imgs = [f.name for f in imgs if f.stem not in lbl_by_stem]
        img_stems = {f.stem for f in imgs}
        orphan_lbls = [p.name for stem, p in lbl_by_stem.items() if stem not in img_stems]

        return {
            "total_images": len(imgs),
            "total_labels": len(lbl_by_stem),
            "matched": len(matched),
            "unmatched_images": len(unmatched_imgs),
            "orphan_labels": len(orphan_lbls),
            "sample_matched": matched[:5],
        }

    def commit_custom_mapping(
        self,
        stage_id: str,
        root_dir: str | Path,
        dataset_name: str,
        image_rel_dir: str,
        label_rel_dir: str | None,
        class_names: list[str] | None,
    ) -> DatasetValidationReport:
        """Build a standard YOLO dataset from user-selected image and label folders."""
        root = Path(root_dir)
        img_dir = root if image_rel_dir in ("", ".") else root / image_rel_dir
        if not img_dir.exists():
            raise ValueError(f"ไม่พบโฟลเดอร์ภาพ: {image_rel_dir}")

        imgs = [f for f in sorted(img_dir.iterdir()) if f.is_file() and f.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS]
        if not imgs:
            raise ValueError("โฟลเดอร์ภาพที่เลือกไม่มีไฟล์ภาพ")

        lbl_by_stem: dict[str, Path] = {}
        if label_rel_dir and label_rel_dir != "(ไม่มี Label)":
            lbl_dir = root if label_rel_dir in ("", ".") else root / label_rel_dir
            if lbl_dir.exists():
                for f in sorted(lbl_dir.iterdir()):
                    if f.is_file() and f.suffix.lower() == ".txt" and f.name.lower() != "classes.txt":
                        lbl_by_stem[f.stem] = f

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = sanitize_identifier(dataset_name or "custom_dataset", fallback="custom_dataset")
        dataset_id = f"{safe_name}_{stamp}"
        container = self.workspace.datasets_dir / dataset_id
        raw_dir = container / "raw"
        img_train = raw_dir / "images" / "train"
        lbl_train = raw_dir / "labels" / "train"
        img_val = raw_dir / "images" / "val"
        lbl_val = raw_dir / "labels" / "val"
        for d in (img_train, lbl_train, img_val, lbl_val):
            d.mkdir(parents=True, exist_ok=True)

        max_class_id = -1
        for img in imgs:
            shutil.copy2(img, img_train / img.name)
            lbl = lbl_by_stem.get(img.stem)
            if lbl is not None:
                shutil.copy2(lbl, lbl_train / f"{img.stem}.txt")
                # infer max class id in case user didn't type class names
                for line in lbl.read_text(encoding="utf-8", errors="replace").splitlines():
                    parts = line.strip().split()
                    if parts:
                        try:
                            max_class_id = max(max_class_id, int(float(parts[0])))
                        except ValueError:
                            pass

        cleaned_classes = [c.strip() for c in (class_names or []) if c and c.strip()]
        unique_classes = list(dict.fromkeys(cleaned_classes))
        needed_nc = max(len(unique_classes), max_class_id + 1, 1)
        while len(unique_classes) < needed_nc:
            unique_classes.append(f"class_{len(unique_classes)}")

        yaml_data = {
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            "nc": len(unique_classes),
            "names": {i: name for i, name in enumerate(unique_classes)},
        }
        (raw_dir / "data.yaml").write_text(
            yaml.safe_dump(yaml_data, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

        # Clean up staged temp directory
        shutil.rmtree(self._stage_dir(stage_id), ignore_errors=True)

        report = validate_yolo_dataset(
            dataset_id=dataset_id,
            dataset_name=safe_name,
            dataset_root=raw_dir,
            resolved_yaml_dir=container,
        )
        write_json_atomic(self._meta_path(dataset_id), report.to_dict())
        return report

    def rename_dataset(self, dataset_id: str, new_name: str) -> DatasetValidationReport:
        report = self.get_dataset(dataset_id)
        if report is None:
            raise ValueError(f"ไม่พบ Dataset '{dataset_id}'")
        clean = sanitize_identifier(new_name, fallback="")
        if not clean:
            raise ValueError("กรุณาระบุชื่อ Dataset ใหม่ให้ถูกต้อง")
        report.dataset_name = clean
        write_json_atomic(self._meta_path(dataset_id), report.to_dict())
        return report

    def preview_split(
        self,
        dataset_id: str,
        train_pct: float,
        val_pct: float,
        test_pct: float,
        drop_duplicates: bool = True,
    ) -> dict[str, int]:
        """Compute exact image counts per split before applying split."""
        from core.dataset.diagnostics import find_corrupt, find_duplicate_groups
        from core.dataset.entries import collect_entries

        if min(train_pct, val_pct, test_pct) < 0:
            raise ValueError("เปอร์เซ็นต์ต้องไม่ติดลบ")
        if abs((train_pct + val_pct + test_pct) - 100.0) > 0.01:
            raise ValueError(f"ผลรวม Train + Val + Test ต้องเท่ากับ 100% (ตอนนี้ {train_pct + val_pct + test_pct:g}%)")
        source = self.get_dataset(dataset_id)
        if source is None:
            raise ValueError(f"ไม่พบ Dataset '{dataset_id}'")
        entries = collect_entries(source)
        corrupt = set(find_corrupt(entries))
        excluded_dup: set[str] = set()
        if drop_duplicates:
            for group in find_duplicate_groups([e for e in entries if e.display_id not in corrupt]):
                excluded_dup.update(group[1:])
        n = len([e for e in entries if e.display_id not in corrupt and e.display_id not in excluded_dup])
        n_train = round(n * train_pct / 100.0)
        n_val = n - n_train if test_pct == 0 else min(round(n * val_pct / 100.0), n - n_train)
        n_test = n - n_train - n_val
        return {
            "total": n,
            "train": n_train,
            "val": n_val,
            "test": n_test,
            "excluded_corrupt": len(corrupt),
            "excluded_duplicate": len(excluded_dup),
        }

    def preview_augmentation(self, dataset_id: str, cfg: Any, sample_index: int = 0) -> tuple[Any, int, int]:
        """Preview offline augmentation on a train image. Returns (PIL.Image, box_count, total_train_images)."""
        from core.dataset.augmentor import preview_augmented_sample
        from core.dataset.entries import collect_entries

        source = self.get_dataset(dataset_id)
        if source is None:
            raise ValueError(f"ไม่พบ Dataset '{dataset_id}'")
        train_entries = [e for e in collect_entries(source) if e.split == "train"]
        if not train_entries:
            raise ValueError("Dataset นี้ไม่มีภาพในชุด Train สำหรับ Preview Augmentation")
        entry = train_entries[sample_index % len(train_entries)]
        id_to_name = {int(k): str(v) for k, v in source.class_names.items()}
        img, box_cnt = preview_augmented_sample(entry, id_to_name, cfg, seed=cfg.seed + sample_index)
        return img, box_cnt, len(train_entries)

    def create_augmented_dataset(
        self,
        dataset_id: str,
        cfg: Any,
        new_name: str | None = None,
    ) -> tuple[DatasetValidationReport, dict[str, Any]]:
        """Generate a NEW dataset with offline augmented copies added to Train split ONLY (Val/Test untouched)."""
        from core.dataset.augmentor import generate_offline_augmented_dataset
        from core.dataset.entries import collect_entries

        source = self.get_dataset(dataset_id)
        if source is None:
            raise ValueError(f"ไม่พบ Dataset '{dataset_id}'")
        entries = collect_entries(source)
        if not any(e.split == "train" for e in entries):
            raise ValueError("Dataset นี้ไม่มีภาพในชุด Train")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = sanitize_identifier(new_name or f"{source.dataset_name}_aug", fallback="augmented")
        new_id = f"{base}_{stamp}"
        container = self.workspace.datasets_dir / new_id
        out_root = container / "raw"
        counts = generate_offline_augmented_dataset(
            entries=entries,
            class_names={int(k): v for k, v in source.class_names.items()},
            out_root=out_root,
            cfg=cfg,
            source_id=dataset_id,
        )
        report = validate_yolo_dataset(
            dataset_id=new_id,
            dataset_name=base,
            dataset_root=out_root,
            resolved_yaml_dir=container,
        )
        write_json_atomic(self._meta_path(new_id), report.to_dict())
        return report, counts

    def get_preview_gallery(self, dataset_id: str, draw_boxes: bool = True) -> list[tuple[str, str]]:
        report = self.get_dataset(dataset_id)
        if report is None:
            return []
        preview_dir = self.workspace.temp_dir / "previews" / dataset_id / ("boxes" if draw_boxes else "raw")
        return render_sample_previews(report, preview_dir=preview_dir, draw_boxes=draw_boxes)
