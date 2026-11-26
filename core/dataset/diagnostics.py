"""Dataset diagnostics: duplicates, corrupt images, missing/orphan/invalid labels, class distribution."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from core.dataset.annotation_io import parse_label_file
from core.dataset.entries import ImageEntry


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def find_corrupt(entries: list[ImageEntry]) -> list[str]:
    return [e.display_id for e in entries if not is_readable_image(e.image_path)]


def find_duplicate_groups(entries: list[ImageEntry]) -> list[list[str]]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for e in entries:
        by_hash[file_sha256(e.image_path)].append(e.display_id)
    return [ids for ids in by_hash.values() if len(ids) > 1]


def run_diagnostics(entries: list[ImageEntry], class_names: dict[int, str]) -> dict[str, Any]:
    id_to_name = {int(k): str(v) for k, v in class_names.items()}

    duplicates = find_duplicate_groups(entries)
    corrupt = find_corrupt(entries)
    missing_labels = [e.display_id for e in entries if not e.has_label]

    known_labels = {e.label_path for e in entries}
    label_dirs = {(e.split, e.labels_dir) for e in entries}
    orphan_labels: list[str] = []
    for split, labels_dir in sorted(label_dirs, key=lambda x: (x[0], str(x[1]))):
        if labels_dir.exists():
            for txt in sorted(labels_dir.rglob("*.txt")):
                if txt not in known_labels:
                    orphan_labels.append(f"{split}/{txt.relative_to(labels_dir).as_posix()}")

    invalid_labels: list[tuple[str, str]] = []
    invalid_class_boxes = 0
    empty_labels = 0
    per_split: dict[str, Counter] = defaultdict(Counter)
    for e in entries:
        if not e.has_label:
            continue
        boxes, problems = parse_label_file(e.label_path)
        if problems:
            invalid_labels.append((e.display_id, problems[0]))
        if not boxes:
            empty_labels += 1
        for b in boxes:
            if b.class_id not in id_to_name:
                invalid_class_boxes += 1
                continue
            per_split[e.split][id_to_name[b.class_id]] += 1

    totals: Counter = Counter()
    for counter in per_split.values():
        totals.update(counter)
    unused_classes = [name for _, name in sorted(id_to_name.items()) if totals.get(name, 0) == 0]

    return {
        "total_images": len(entries),
        "duplicates": duplicates,
        "corrupt": corrupt,
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
        "invalid_labels": invalid_labels,
        "invalid_class_boxes": invalid_class_boxes,
        "empty_labels": empty_labels,
        "class_counts": {split: dict(counter) for split, counter in per_split.items()},
        "class_totals": dict(totals),
        "unused_classes": unused_classes,
    }
