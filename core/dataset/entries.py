"""Enumerate image/label pairs across dataset splits (shared by annotation, diagnostics, split)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.defaults import ALLOWED_IMAGE_EXTENSIONS
from core.schemas import DatasetValidationReport

SPLITS = ("train", "val", "test")


@dataclass
class ImageEntry:
    split: str
    key: str  # path relative to images dir, without suffix, posix style (e.g. "site_a/img01")
    image_path: Path
    label_path: Path  # may not exist yet
    images_dir: Path
    labels_dir: Path

    @property
    def has_label(self) -> bool:
        return self.label_path.exists()

    @property
    def display_id(self) -> str:
        return f"{self.split}/{self.key}"


def resolve_labels_dir(images_dir: Path) -> Path:
    """Ultralytics convention: replace the 'images' path segment with 'labels'."""
    parts = list(images_dir.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx].lower() == "images":
            return Path(*parts[:idx], "labels", *parts[idx + 1 :])
    return images_dir.parent / "labels"


def collect_entries(report: DatasetValidationReport) -> list[ImageEntry]:
    entries: list[ImageEntry] = []
    for split_name in SPLITS:
        stats = report.splits.get(split_name)
        if stats is None or not stats.images_dir:
            continue
        images_dir = Path(stats.images_dir)
        labels_dir = Path(stats.labels_dir) if stats.labels_dir else resolve_labels_dir(images_dir)
        for img in sorted(images_dir.rglob("*")):
            if not img.is_file() or img.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
                continue
            rel = img.relative_to(images_dir)
            entries.append(
                ImageEntry(
                    split=split_name,
                    key=rel.with_suffix("").as_posix(),
                    image_path=img,
                    label_path=labels_dir / rel.with_suffix(".txt"),
                    images_dir=images_dir,
                    labels_dir=labels_dir,
                )
            )
    return entries


def find_entry(entries: list[ImageEntry], split: str, key: str) -> ImageEntry | None:
    for entry in entries:
        if entry.split == split and entry.key == key:
            return entry
    return None
