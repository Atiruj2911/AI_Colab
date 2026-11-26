"""YOLO Detection dataset validator with actionable errors, warnings, and safe YAML resolution."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from config.defaults import ALLOWED_IMAGE_EXTENSIONS, SAMPLE_PREVIEW_COUNT
from core.schemas import DatasetValidationReport, SplitStats, ValidationIssue


def _count_image_files(dataset_root: Path) -> int:
    return sum(
        1
        for p in dataset_root.rglob("*")
        if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS and "__MACOSX" not in p.parts
    )


def _find_data_yaml(dataset_root: Path) -> Path | None:
    """Locate data.yaml (or dataset.yaml / first .yaml) near the dataset root."""
    for candidate_name in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
        direct = dataset_root / candidate_name
        if direct.is_file():
            return direct
    yamls = sorted(
        p for p in dataset_root.rglob("*.yaml")
        if p.name != "resolved_data.yaml" and "__MACOSX" not in p.parts
    )
    if not yamls:
        yamls = sorted(
            p for p in dataset_root.rglob("*.yml")
            if p.name != "resolved_data.yml" and "__MACOSX" not in p.parts
        )
    return yamls[0] if yamls else None


def _parse_class_names(raw_names: Any) -> tuple[dict[int, str], list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    class_names: dict[int, str] = {}

    if raw_names is None:
        issues.append(
            ValidationIssue(
                severity="error",
                code="MISSING_NAMES",
                message="ไม่พบคีย์ 'names' ใน data.yaml",
                hint="เพิ่มรายชื่อคลาส เช่น names: [person, helmet] หรือ names: {0: person, 1: helmet}",
            )
        )
        return class_names, issues

    if isinstance(raw_names, list):
        if len(raw_names) == 0:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="EMPTY_NAMES",
                    message="รายการ 'names' ใน data.yaml ว่างเปล่า",
                    hint="ระบุอย่างน้อย 1 คลาสใน data.yaml",
                )
            )
        for idx, item in enumerate(raw_names):
            class_names[idx] = str(item).strip()
    elif isinstance(raw_names, dict):
        if len(raw_names) == 0:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="EMPTY_NAMES",
                    message="รายการ 'names' ใน data.yaml ว่างเปล่า",
                )
            )
        for k, v in raw_names.items():
            try:
                cid = int(k)
            except (TypeError, ValueError):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="INVALID_CLASS_ID",
                        message=f"รหัสคลาส '{k}' ใน data.yaml ไม่ใช่จำนวนเต็ม",
                    )
                )
                continue
            class_names[cid] = str(v).strip()
    else:
        issues.append(
            ValidationIssue(
                severity="error",
                code="INVALID_NAMES_TYPE",
                message="รูปแบบของ 'names' ใน data.yaml ไม่ถูกต้อง (ต้องเป็น list หรือ dict)",
            )
        )

    return class_names, issues


def _resolve_split_dir(base_dir: Path, dataset_root: Path, raw_entry: Any, split_name: str) -> Path | None:
    """Resolve train/val/test image directory from data.yaml entry or standard YOLO conventions."""
    candidates: list[Path] = []

    if isinstance(raw_entry, str) and raw_entry.strip():
        entry_str = raw_entry.strip()
        p = Path(entry_str)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append((base_dir / p).resolve())
            candidates.append((dataset_root / p).resolve())
            # Common Roboflow pattern: "../train/images" when yaml is at dataset root
            stripped = entry_str.lstrip("./\\")
            while stripped.startswith("../") or stripped.startswith("..\\"):
                stripped = stripped[3:]
            candidates.append((dataset_root / stripped).resolve())

    # Standard fallback conventions if YAML path doesn't resolve directly
    aliases = [split_name]
    if split_name == "val":
        aliases.append("valid")
    for alias in aliases:
        candidates.extend(
            [
                dataset_root / "images" / alias,
                dataset_root / alias / "images",
                dataset_root / alias,
            ]
        )

    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return cand.resolve()
    return None


def _infer_labels_dir(images_dir: Path) -> Path | None:
    """Follow Ultralytics YOLO convention: replace 'images' path segment with 'labels'."""
    parts = list(images_dir.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx].lower() == "images":
            candidate = Path(*parts[:idx], "labels", *parts[idx + 1 :])
            if candidate.exists() and candidate.is_dir():
                return candidate
    # Alternative sibling folder: <split>/labels next to <split>/images
    sibling = images_dir.parent / "labels"
    if sibling.exists() and sibling.is_dir():
        return sibling
    # Labels alongside images
    if any(images_dir.glob("*.txt")):
        return images_dir
    return None


def _build_structure_tree(root: Path, max_depth: int = 3, max_entries_per_dir: int = 10) -> str:
    lines = [f"{root.name}/"]

    def _walk(directory: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            children = sorted(
                [p for p in directory.iterdir() if p.name != "__MACOSX" and not p.name.startswith(".")],
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except OSError:
            return

        shown = children[:max_entries_per_dir]
        hidden_count = len(children) - len(shown)

        for idx, child in enumerate(shown):
            is_last = (idx == len(shown) - 1) and (hidden_count == 0)
            connector = "└── " if is_last else "├── "
            if child.is_dir():
                file_count = sum(1 for _ in child.glob("*") if _.is_file())
                suffix = f"/ ({file_count} files)" if file_count else "/"
                lines.append(f"{prefix}{connector}{child.name}{suffix}")
                extension = "    " if is_last else "│   "
                _walk(child, prefix + extension, depth + 1)
            else:
                lines.append(f"{prefix}{connector}{child.name}")

        if hidden_count > 0:
            lines.append(f"{prefix}└── ... (+{hidden_count} more)")

    _walk(root, "", 1)
    return "\n".join(lines)


def _validate_label_file(
    label_path: Path,
    valid_class_ids: set[int],
) -> tuple[int, list[int], list[str]]:
    """Validate YOLO detection label file lines: `class cx cy w h` in [0, 1]."""
    problems: list[str] = []
    class_ids_found: list[int] = []
    box_count = 0

    try:
        content = label_path.read_text(encoding="utf-8", errors="strict")
    except Exception as exc:
        return 0, [], [f"อ่านไฟล์ label ไม่ได้ ({exc})"]

    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    if not lines:
        return 0, [], []  # Empty file = background image (valid warning, not fatal)

    for line_no, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            problems.append(
                f"บรรทัด {line_no}: ต้องมี 5 ค่า (class x_center y_center width height) แต่พบ {len(parts)} ค่า"
            )
            continue

        try:
            cid = int(float(parts[0]))
            if float(parts[0]) != cid:
                raise ValueError("class id not integer")
            coords = [float(x) for x in parts[1:5]]
        except ValueError:
            problems.append(f"บรรทัด {line_no}: ค่าตัวเลขไม่ถูกต้อง ('{line}')")
            continue

        if valid_class_ids and cid not in valid_class_ids:
            problems.append(
                f"บรรทัด {line_no}: class_id={cid} ไม่อยู่ใน names ของ data.yaml ({sorted(valid_class_ids)})"
            )
            continue

        cx, cy, w, h = coords
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            problems.append(
                f"บรรทัด {line_no}: พิกัด normalized ออกนอกช่วง (cx={cx:.3f}, cy={cy:.3f}, w={w:.3f}, h={h:.3f})"
            )
            continue

        box_count += 1
        class_ids_found.append(cid)

    return box_count, class_ids_found, problems


def validate_yolo_dataset(
    dataset_id: str,
    dataset_name: str,
    dataset_root: Path,
    resolved_yaml_dir: Path | None = None,
) -> DatasetValidationReport:
    """Inspect and validate a YOLO Detection dataset thoroughly."""
    dataset_root = Path(dataset_root).resolve()
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    tree_str = _build_structure_tree(dataset_root)

    yaml_path = _find_data_yaml(dataset_root)
    if yaml_path is None:
        errors.append(
            ValidationIssue(
                severity="error",
                code="MISSING_DATA_YAML",
                message="ไม่พบไฟล์ data.yaml ในชุดข้อมูล",
                hint="ตรวจสอบว่าไฟล์ ZIP มี data.yaml ที่กำหนด path ของ train, val และ names",
            )
        )
        if _count_image_files(dataset_root) == 0:
            errors.append(
                ValidationIssue(
                    severity="error",
                    code="NO_IMAGES_FOUND",
                    message="ไม่พบไฟล์ภาพ (.jpg/.png ฯลฯ) ในชุดข้อมูลเลย",
                    hint="ตรวจสอบว่าโฟลเดอร์ images/train และ images/val มีไฟล์ภาพจริงก่อนบีบอัดเป็น ZIP",
                )
            )
        return DatasetValidationReport(
            is_valid=False,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            dataset_root=str(dataset_root),
            errors=errors,
            warnings=warnings,
            structure_tree=tree_str,
        )

    try:
        raw_yaml = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not isinstance(raw_yaml, dict):
            raise ValueError("เนื้อหา data.yaml ต้องเป็น YAML mapping (key: value)")
    except Exception as exc:
        errors.append(
            ValidationIssue(
                severity="error",
                code="INVALID_YAML",
                message=f"ไม่สามารถอ่าน data.yaml ได้: {exc}",
                path=str(yaml_path),
                hint="ตรวจสอบ syntax ของไฟล์ YAML (การย่อหน้าและเครื่องหมาย :)",
            )
        )
        return DatasetValidationReport(
            is_valid=False,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            dataset_root=str(dataset_root),
            data_yaml_path=str(yaml_path),
            errors=errors,
            warnings=warnings,
            structure_tree=tree_str,
        )

    class_names, name_issues = _parse_class_names(raw_yaml.get("names"))
    errors.extend([i for i in name_issues if i.severity == "error"])
    warnings.extend([i for i in name_issues if i.severity == "warning"])

    nc_declared = raw_yaml.get("nc")
    if nc_declared is not None:
        try:
            if int(nc_declared) != len(class_names):
                warnings.append(
                    ValidationIssue(
                        severity="warning",
                        code="NC_MISMATCH",
                        message=f"ค่า nc={nc_declared} ไม่ตรงกับจำนวนชื่อคลาสใน names ({len(class_names)})",
                        path=str(yaml_path),
                    )
                )
        except (TypeError, ValueError):
            warnings.append(
                ValidationIssue(
                    severity="warning",
                    code="INVALID_NC",
                    message=f"ค่า nc ใน data.yaml ไม่ใช่ตัวเลข: {nc_declared}",
                )
            )

    yaml_base = yaml_path.parent
    if isinstance(raw_yaml.get("path"), str) and raw_yaml["path"].strip():
        candidate_base = Path(raw_yaml["path"].strip())
        if not candidate_base.is_absolute():
            candidate_base = (yaml_path.parent / candidate_base).resolve()
        if candidate_base.exists() and candidate_base.is_dir():
            yaml_base = candidate_base

    valid_class_ids = set(class_names.keys())
    splits_result: dict[str, SplitStats] = {}
    class_counter: Counter[int] = Counter()
    sample_images: list[str] = []
    resolved_split_paths: dict[str, str] = {}

    for split_name in ("train", "val", "test"):
        raw_entry = raw_yaml.get(split_name)
        if split_name == "val" and raw_entry is None:
            raw_entry = raw_yaml.get("valid")

        images_dir = _resolve_split_dir(yaml_base, dataset_root, raw_entry, split_name)
        if images_dir is None:
            if split_name in ("train", "val"):
                errors.append(
                    ValidationIssue(
                        severity="error",
                        code=f"MISSING_{split_name.upper()}_SPLIT",
                        message=f"ไม่พบโฟลเดอร์ภาพของชุด '{split_name}' (ตรวจสอบคีย์ '{split_name}' ใน data.yaml)",
                        hint=f"จัดโครงสร้างเป็น images/{split_name}/ หรือ {split_name}/images/ และระบุใน data.yaml",
                    )
                )
            continue

        resolved_split_paths[split_name] = str(images_dir)
        labels_dir = _infer_labels_dir(images_dir)

        image_files = sorted(
            p for p in images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
        )
        label_files = (
            sorted(p for p in labels_dir.rglob("*.txt") if p.is_file())
            if labels_dir is not None
            else []
        )

        # Map relative stem -> file path for robust pairing
        image_map = {p.relative_to(images_dir).with_suffix(""): p for p in image_files}
        label_map = (
            {p.relative_to(labels_dir).with_suffix(""): p for p in label_files}
            if labels_dir is not None
            else {}
        )

        stats = SplitStats(
            split_name=split_name,
            images_dir=str(images_dir),
            labels_dir=str(labels_dir) if labels_dir else None,
            image_count=len(image_files),
            label_count=len(label_files),
        )

        if len(image_files) == 0 and split_name in ("train", "val"):
            errors.append(
                ValidationIssue(
                    severity="error",
                    code=f"EMPTY_{split_name.upper()}_IMAGES",
                    message=f"โฟลเดอร์ '{split_name}' ไม่มีไฟล์รูปภาพที่รองรับ ({images_dir})",
                )
            )

        if labels_dir is None and split_name in ("train", "val"):
            errors.append(
                ValidationIssue(
                    severity="error",
                    code=f"MISSING_{split_name.upper()}_LABELS_DIR",
                    message=f"ไม่พบโฟลเดอร์ labels สำหรับชุด '{split_name}'",
                    hint=f"ควรมีโฟลเดอร์ labels/{split_name}/ หรือ {split_name}/labels/ คู่กับภาพ",
                )
            )

        # Check readability of images (full check for <= 500 images, sampled + all for small sets)
        unreadable_list: list[str] = []
        for img_path in image_files:
            try:
                with Image.open(img_path) as im:
                    im.verify()
                if len(sample_images) < SAMPLE_PREVIEW_COUNT:
                    sample_images.append(str(img_path))
            except Exception:
                stats.unreadable_images += 1
                if len(unreadable_list) < 5:
                    unreadable_list.append(img_path.name)

        if stats.unreadable_images > 0:
            errors.append(
                ValidationIssue(
                    severity="error",
                    code=f"UNREADABLE_IMAGES_{split_name.upper()}",
                    message=f"พบไฟล์ภาพที่เสียหาย/เปิดไม่ได้จำนวน {stats.unreadable_images} ไฟล์ในชุด '{split_name}' (เช่น {', '.join(unreadable_list)})",
                    hint="ลบหรือแทนที่ไฟล์ภาพที่เสียหายก่อนเริ่มเทรน",
                )
            )

        # Pairing & label validation
        matched_keys = set(image_map.keys()) & set(label_map.keys())
        stats.matched_pairs = len(matched_keys)
        stats.images_without_labels = len(set(image_map.keys()) - set(label_map.keys()))
        stats.orphan_labels = len(set(label_map.keys()) - set(image_map.keys()))

        malformed_examples: list[str] = []
        for key in sorted(matched_keys):
            lbl_path = label_map[key]
            box_count, found_cids, problems = _validate_label_file(lbl_path, valid_class_ids)
            if problems:
                stats.malformed_label_files += 1
                if len(malformed_examples) < 5:
                    malformed_examples.append(f"{lbl_path.name}: {problems[0]}")
            elif box_count == 0:
                stats.empty_labels += 1
            stats.total_boxes += box_count
            class_counter.update(found_cids)

        if stats.malformed_label_files > 0:
            errors.append(
                ValidationIssue(
                    severity="error",
                    code=f"MALFORMED_LABELS_{split_name.upper()}",
                    message=(
                        f"พบไฟล์ Label รูปแบบไม่ถูกต้องจำนวน {stats.malformed_label_files} ไฟล์ในชุด '{split_name}' "
                        f"({'; '.join(malformed_examples)})"
                    ),
                    hint="ตรวจสอบรูปแบบ YOLO: <class_id> <x_center> <y_center> <width> <height> โดยพิกัดอยู่ในช่วง 0–1",
                )
            )

        if stats.images_without_labels > 0:
            if stats.matched_pairs == 0 and split_name in ("train", "val"):
                errors.append(
                    ValidationIssue(
                        severity="error",
                        code=f"NO_MATCHED_LABELS_{split_name.upper()}",
                        message=f"ไม่พบไฟล์ Label (.txt) ที่ชื่อตรงกับภาพเลยในชุด '{split_name}'",
                        hint="ชื่อไฟล์ภาพและไฟล์ .txt ต้องตรงกัน เช่น img01.jpg คู่กับ img01.txt",
                    )
                )
            else:
                warnings.append(
                    ValidationIssue(
                        severity="warning",
                        code=f"MISSING_SOME_LABELS_{split_name.upper()}",
                        message=f"ชุด '{split_name}' มีภาพที่ไม่มีไฟล์ Label จำนวน {stats.images_without_labels} ภาพ (YOLO จะถือเป็นภาพ background)",
                    )
                )

        if stats.orphan_labels > 0:
            warnings.append(
                ValidationIssue(
                    severity="warning",
                    code=f"ORPHAN_LABELS_{split_name.upper()}",
                    message=f"ชุด '{split_name}' มีไฟล์ Label ที่ไม่มีภาพคู่กันจำนวน {stats.orphan_labels} ไฟล์",
                )
            )

        if stats.empty_labels > 0:
            warnings.append(
                ValidationIssue(
                    severity="warning",
                    code=f"EMPTY_LABELS_{split_name.upper()}",
                    message=f"ชุด '{split_name}' มีไฟล์ Label ว่างจำนวน {stats.empty_labels} ไฟล์ (ภาพ background)",
                )
            )

        if stats.total_boxes == 0 and split_name in ("train", "val") and stats.matched_pairs > 0:
            errors.append(
                ValidationIssue(
                    severity="error",
                    code=f"NO_BOXES_{split_name.upper()}",
                    message=f"ไม่พบ bounding box ที่ถูกต้องเลยในชุด '{split_name}'",
                )
            )

        splits_result[split_name] = stats

    total_images = sum(s.image_count for s in splits_result.values())
    total_labels = sum(s.label_count for s in splits_result.values())
    total_boxes = sum(s.total_boxes for s in splits_result.values())

    class_counts = {
        class_names.get(cid, f"class_{cid}"): int(class_counter.get(cid, 0))
        for cid in sorted(class_names.keys())
    }

    unused_classes = [name for name, cnt in class_counts.items() if cnt == 0]
    if unused_classes and not errors:
        warnings.append(
            ValidationIssue(
                severity="warning",
                code="UNUSED_CLASSES",
                message=f"คลาสที่ไม่มีตัวอย่างใน Label เลย: {', '.join(unused_classes)}",
            )
        )

    is_valid = len(errors) == 0

    resolved_yaml_path: str | None = None
    if is_valid and "train" in resolved_split_paths and "val" in resolved_split_paths:
        target_dir = Path(resolved_yaml_dir) if resolved_yaml_dir else dataset_root
        target_dir.mkdir(parents=True, exist_ok=True)
        resolved_file = target_dir / "resolved_data.yaml"
        resolved_payload: dict[str, Any] = {
            "path": str(dataset_root),
            "train": resolved_split_paths["train"],
            "val": resolved_split_paths["val"],
            "nc": len(class_names),
            "names": {int(k): str(v) for k, v in sorted(class_names.items())},
        }
        if "test" in resolved_split_paths:
            resolved_payload["test"] = resolved_split_paths["test"]
        resolved_file.write_text(
            yaml.safe_dump(resolved_payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        resolved_yaml_path = str(resolved_file)

    return DatasetValidationReport(
        is_valid=is_valid,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        dataset_root=str(dataset_root),
        data_yaml_path=str(yaml_path),
        resolved_yaml_path=resolved_yaml_path,
        class_names=class_names,
        splits=splits_result,
        total_images=total_images,
        total_labels=total_labels,
        total_boxes=total_boxes,
        class_counts=class_counts,
        errors=errors,
        warnings=warnings,
        structure_tree=tree_str,
        sample_images=sample_images,
    )
