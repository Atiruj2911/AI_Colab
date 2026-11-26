"""Dataset visual preview generator (raw or with ground-truth YOLO boxes)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from core.schemas import DatasetValidationReport

_BOX_COLORS = [
    (99, 102, 241),   # indigo
    (16, 185, 129),   # emerald
    (245, 158, 11),   # amber
    (236, 72, 153),   # pink
    (56, 189, 248),   # sky
    (168, 85, 247),   # purple
]


def _find_matching_label(image_path: Path, report: DatasetValidationReport) -> Path | None:
    for split_stats in report.splits.values():
        if not split_stats.images_dir or not split_stats.labels_dir:
            continue
        img_root = Path(split_stats.images_dir)
        lbl_root = Path(split_stats.labels_dir)
        try:
            rel = image_path.resolve().relative_to(img_root.resolve())
            cand = lbl_root / rel.with_suffix(".txt")
            if cand.exists():
                return cand
        except ValueError:
            continue
    return None


def render_sample_previews(
    report: DatasetValidationReport,
    preview_dir: Path,
    draw_boxes: bool = True,
    max_images: int = 8,
) -> list[tuple[str, str]]:
    """Return list of (image_path, caption) for Gradio Gallery preview."""
    preview_dir = Path(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    gallery_items: list[tuple[str, str]] = []

    for idx, img_str in enumerate(report.sample_images[:max_images]):
        img_path = Path(img_str)
        if not img_path.exists():
            continue

        if not draw_boxes:
            gallery_items.append((str(img_path), img_path.name))
            continue

        lbl_path = _find_matching_label(img_path, report)
        try:
            with Image.open(img_path) as im:
                rgb = im.convert("RGB")
                w, h = rgb.size
                draw = ImageDraw.Draw(rgb)
                box_count = 0

                if lbl_path and lbl_path.exists():
                    for line in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        try:
                            cid = int(float(parts[0]))
                            cx, cy, bw, bh = [float(x) for x in parts[1:5]]
                        except ValueError:
                            continue
                        x1 = max(0.0, (cx - bw / 2.0) * w)
                        y1 = max(0.0, (cy - bh / 2.0) * h)
                        x2 = min(float(w), (cx + bw / 2.0) * w)
                        y2 = min(float(h), (cy + bh / 2.0) * h)
                        color = _BOX_COLORS[cid % len(_BOX_COLORS)]
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, int(min(w, h) / 180)))
                        cname = report.class_names.get(cid, f"cls_{cid}")
                        draw.rectangle([x1, max(0, y1 - 16), min(w, x1 + len(cname) * 7 + 8), y1], fill=color)
                        draw.text((x1 + 3, max(0, y1 - 14)), cname, fill=(255, 255, 255))
                        box_count += 1

                out_file = preview_dir / f"preview_{idx:02d}_{img_path.stem}.jpg"
                rgb.save(out_file, quality=88)
                gallery_items.append((str(out_file), f"{img_path.name} ({box_count} boxes)"))
        except Exception:
            gallery_items.append((str(img_path), img_path.name))

    return gallery_items
