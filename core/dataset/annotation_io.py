"""YOLO label read/write, pixel<->normalized conversion, and box rendering."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

_BOX_COLORS = [
    (99, 102, 241),
    (16, 185, 129),
    (245, 158, 11),
    (236, 72, 153),
    (56, 189, 248),
    (168, 85, 247),
]


@dataclass
class Box:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    conf: float | None = None


def parse_label_file(path: Path) -> tuple[list[Box], list[str]]:
    """Parse a YOLO detection label file. Returns (boxes, problems)."""
    path = Path(path)
    if not path.exists():
        return [], []
    boxes: list[Box] = []
    problems: list[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            problems.append(f"บรรทัด {line_no}: ต้องมี 5 ค่า พบ {len(parts)} ค่า")
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, w, h = (float(x) for x in parts[1:5])
        except ValueError:
            problems.append(f"บรรทัด {line_no}: ค่าไม่ใช่ตัวเลข")
            continue
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            problems.append(f"บรรทัด {line_no}: พิกัด normalized อยู่นอกช่วง 0–1")
            continue
        boxes.append(Box(cid, cx, cy, w, h))
    return boxes, problems


def format_label_lines(boxes: list[Box]) -> str:
    return "".join(f"{b.class_id} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}\n" for b in boxes)


def write_label_file(path: Path, boxes: list[Box]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(format_label_lines(boxes), encoding="utf-8")
    os.replace(tmp, path)


def pixel_rows_to_boxes(
    rows: list[list] | None,
    img_w: int,
    img_h: int,
    name_to_id: dict[str, int],
) -> list[Box]:
    """Convert editor rows [class_name, x1, y1, x2, y2] (pixels) into normalized boxes."""
    boxes: list[Box] = []
    errors: list[str] = []
    for idx, row in enumerate(rows or [], start=1):
        if row is None or len(row) < 5:
            continue
        name = str(row[0] or "").strip()
        values = row[1:5]
        if not name and all(v in (None, "") for v in values):
            continue
        if name not in name_to_id:
            errors.append(f"แถว {idx}: ไม่พบคลาส '{name}' (ต้องตรงกับชื่อคลาสใน data.yaml)")
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in values)
        except (TypeError, ValueError):
            errors.append(f"แถว {idx}: พิกัดต้องเป็นตัวเลข")
            continue
        x1, x2 = (min(max(x1, 0.0), img_w), min(max(x2, 0.0), img_w))
        y1, y2 = (min(max(y1, 0.0), img_h), min(max(y2, 0.0), img_h))
        if x2 - x1 < 1 or y2 - y1 < 1:
            errors.append(f"แถว {idx}: กรอบเล็กเกินไปหรือ x2/y2 ไม่มากกว่า x1/y1")
            continue
        boxes.append(
            Box(
                class_id=name_to_id[name],
                cx=((x1 + x2) / 2.0) / img_w,
                cy=((y1 + y2) / 2.0) / img_h,
                w=(x2 - x1) / img_w,
                h=(y2 - y1) / img_h,
            )
        )
    if errors:
        raise ValueError("; ".join(errors))
    return boxes


def boxes_to_pixel_rows(boxes: list[Box], id_to_name: dict[int, str], img_w: int, img_h: int) -> list[list]:
    rows: list[list] = []
    for b in boxes:
        x1 = (b.cx - b.w / 2.0) * img_w
        x2 = (b.cx + b.w / 2.0) * img_w
        y1 = (b.cy - b.h / 2.0) * img_h
        y2 = (b.cy + b.h / 2.0) * img_h
        rows.append(
            [
                id_to_name.get(b.class_id, f"class_{b.class_id}"),
                round(x1),
                round(y1),
                round(x2),
                round(y2),
            ]
        )
    return rows


def render_boxes(
    image_path: Path,
    boxes: list[Box],
    id_to_name: dict[int, str],
    out_path: Path | None = None,
) -> Image.Image:
    """Draw boxes on a copy of the image. Saves to out_path when provided."""
    with Image.open(image_path) as im:
        rgb = im.convert("RGB")
    w, h = rgb.size
    draw = ImageDraw.Draw(rgb)
    for b in boxes:
        x1 = (b.cx - b.w / 2.0) * w
        x2 = (b.cx + b.w / 2.0) * w
        y1 = (b.cy - b.h / 2.0) * h
        y2 = (b.cy + b.h / 2.0) * h
        color = _BOX_COLORS[b.class_id % len(_BOX_COLORS)]
        label = id_to_name.get(b.class_id, f"cls_{b.class_id}")
        if b.conf is not None:
            label = f"{label} {b.conf:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, int(min(w, h) / 180)))
        draw.rectangle([x1, max(0, y1 - 14), x1 + len(label) * 6 + 6, y1], fill=color)
        draw.text((x1 + 3, max(0, y1 - 13)), label, fill=(255, 255, 255))
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(out_path, quality=90)
    return rgb
