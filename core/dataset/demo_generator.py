"""Synthetic mini YOLO Detection dataset generator for quick MVP smoke-testing without copyrighted assets."""

from __future__ import annotations

import random
import zipfile
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

DEMO_CLASSES = {0: "box", 1: "circle"}


def generate_demo_yolo_dataset(
    output_dir: Path,
    train_count: int = 12,
    val_count: int = 4,
    img_size: int = 320,
    seed: int = 42,
    also_create_zip: Path | None = None,
) -> Path:
    """Create a small, clean synthetic YOLO Detection dataset with guaranteed valid labels."""
    rng = random.Random(seed)
    output_dir = Path(output_dir).resolve()

    for split, count in (("train", train_count), ("val", val_count)):
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(count):
            bg_color = (
                rng.randint(18, 45),
                rng.randint(24, 55),
                rng.randint(40, 80),
            )
            img = Image.new("RGB", (img_size, img_size), bg_color)
            draw = ImageDraw.Draw(img)

            label_lines: list[str] = []
            num_objects = rng.randint(1, 3)
            for _ in range(num_objects):
                cid = rng.choice([0, 1])
                bw = rng.randint(int(img_size * 0.16), int(img_size * 0.34))
                bh = rng.randint(int(img_size * 0.16), int(img_size * 0.34))
                x1 = rng.randint(8, img_size - bw - 8)
                y1 = rng.randint(8, img_size - bh - 8)
                x2 = x1 + bw
                y2 = y1 + bh

                if cid == 0:
                    fill = (rng.randint(200, 250), rng.randint(120, 180), rng.randint(40, 90))
                    draw.rectangle([x1, y1, x2, y2], fill=fill, outline=(255, 255, 255), width=2)
                else:
                    fill = (rng.randint(50, 110), rng.randint(180, 240), rng.randint(160, 220))
                    draw.ellipse([x1, y1, x2, y2], fill=fill, outline=(255, 255, 255), width=2)

                cx = ((x1 + x2) / 2.0) / img_size
                cy = ((y1 + y2) / 2.0) / img_size
                nw = bw / img_size
                nh = bh / img_size
                label_lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            stem = f"demo_{split}_{idx:03d}"
            img.save(img_dir / f"{stem}.jpg", quality=92)
            (lbl_dir / f"{stem}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    data_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "nc": len(DEMO_CLASSES),
        "names": DEMO_CLASSES,
    }
    (output_dir / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    if also_create_zip is not None:
        also_create_zip = Path(also_create_zip)
        also_create_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(also_create_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(output_dir.rglob("*")):
                if file_path.is_file():
                    zf.write(file_path, arcname=file_path.relative_to(output_dir))

    return output_dir
