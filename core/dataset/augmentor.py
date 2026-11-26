"""Offline YOLO Detection Augmentation Engine with exact bounding-box alignment.

Supports:
- Geometric (updates bounding boxes): Horizontal Flip, Vertical Flip, Rotation, Scale, Translation
- Photometric (boxes unchanged): Brightness, Contrast, Saturation, Hue, Blur, Gaussian Noise

Always augments the Train split ONLY; Val and Test splits are copied untouched to prevent leakage.
"""

from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageFilter

from core.dataset.annotation_io import Box, parse_label_file, render_boxes, write_label_file
from core.dataset.entries import ImageEntry


@dataclass
class AugmentConfig:
    copies_per_image: int = 2
    seed: int = 42
    hflip_p: float = 0.5
    vflip_p: float = 0.0
    rotate_deg: float = 10.0
    rotate_steps: int = 0
    scale_pct: float = 10.0
    translate_pct: float = 5.0
    brightness_pct: float = 15.0
    contrast_pct: float = 15.0
    saturation_pct: float = 15.0
    hue_deg: float = 10.0
    blur_radius: float = 0.0
    noise_std: float = 0.0
    ranges: dict[str, dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sample_step_range(rng: random.Random, spec: dict[str, Any], default: float = 0.0) -> float:
    if not spec or not spec.get("enabled"):
        return default
    lo = float(spec.get("min", default))
    hi = float(spec.get("max", lo))
    if hi < lo:
        lo, hi = hi, lo
    step = float(spec.get("step", 0) or 0)
    if step <= 0 or hi == lo:
        return rng.uniform(lo, hi) if hi > lo else lo
    vals: list[float] = []
    cur = lo
    while cur <= hi + 1e-6 and len(vals) < 1000:
        vals.append(round(cur, 6))
        cur += step
    return rng.choice(vals) if vals else lo


def _boxes_to_xyxy(boxes: list[Box], w: int, h: int) -> list[tuple[int, float, float, float, float]]:
    out = []
    for b in boxes:
        x1 = (b.cx - b.w / 2.0) * w
        y1 = (b.cy - b.h / 2.0) * h
        x2 = (b.cx + b.w / 2.0) * w
        y2 = (b.cy + b.h / 2.0) * h
        out.append((b.class_id, x1, y1, x2, y2))
    return out


def _xyxy_to_boxes(items: list[tuple[int, float, float, float, float]], w: int, h: int) -> list[Box]:
    boxes: list[Box] = []
    for cid, x1, y1, x2, y2 in items:
        cx1 = max(0.0, min(float(w), x1))
        cy1 = max(0.0, min(float(h), y1))
        cx2 = max(0.0, min(float(w), x2))
        cy2 = max(0.0, min(float(h), y2))
        bw = cx2 - cx1
        bh = cy2 - cy1
        if bw < 2.0 or bh < 2.0:
            continue
        boxes.append(
            Box(
                class_id=int(cid),
                cx=((cx1 + cx2) / 2.0) / w,
                cy=((cy1 + cy2) / 2.0) / h,
                w=bw / w,
                h=bh / h,
            )
        )
    return boxes


def _transform_corners(
    items: list[tuple[int, float, float, float, float]],
    fn,
) -> list[tuple[int, float, float, float, float]]:
    out = []
    for cid, x1, y1, x2, y2 in items:
        pts = [fn(x1, y1), fn(x2, y1), fn(x2, y2), fn(x1, y2)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        out.append((cid, min(xs), min(ys), max(xs), max(ys)))
    return out


def augment_image_and_boxes(
    img: Image.Image,
    boxes: list[Box],
    cfg: AugmentConfig,
    rng: random.Random,
) -> tuple[Image.Image, list[Box]]:
    """Apply configured geometric & photometric transforms to one image + its YOLO boxes."""
    im = img.convert("RGB")
    w, h = im.size
    xyxy = _boxes_to_xyxy(boxes, w, h)

    # 1. Horizontal & Vertical flip
    ranges = cfg.ranges
    if ranges is not None:
        do_hflip = bool(ranges.get("hflip", {}).get("enabled")) and rng.random() < 0.5
        do_vflip = bool(ranges.get("vflip", {}).get("enabled")) and rng.random() < 0.5
    else:
        do_hflip = cfg.hflip_p > 0 and rng.random() < cfg.hflip_p
        do_vflip = cfg.vflip_p > 0 and rng.random() < cfg.vflip_p

    if do_hflip:
        im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        xyxy = [(cid, w - x2, y1, w - x1, y2) for cid, x1, y1, x2, y2 in xyxy]

    if do_vflip:
        im = im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        xyxy = [(cid, x1, h - y2, x2, h - y1) for cid, x1, y1, x2, y2 in xyxy]

    # 3. Affine: Rotation + Scale + Translation (exact forward matrix -> PIL inverse matrix)
    if ranges is not None:
        angle_deg = _sample_step_range(rng, ranges.get("rotation", {}), 0.0)
        scale = max(0.3, min(2.0, _sample_step_range(rng, ranges.get("scale", {}), 1.0)))
        tx_pct = _sample_step_range(rng, ranges.get("translation", {}), 0.0)
        ty_pct = _sample_step_range(rng, ranges.get("translation", {}), 0.0)
        tx = tx_pct / 100.0 * w
        ty = ty_pct / 100.0 * h
    else:
        if cfg.rotate_deg > 0:
            if cfg.rotate_steps and cfg.rotate_steps > 1:
                step_vals = [
                    -cfg.rotate_deg + (2.0 * cfg.rotate_deg * i) / (cfg.rotate_steps - 1)
                    for i in range(int(cfg.rotate_steps))
                ]
                angle_deg = rng.choice(step_vals)
            elif cfg.rotate_steps == 1:
                angle_deg = rng.choice([-cfg.rotate_deg, cfg.rotate_deg])
            else:
                angle_deg = rng.uniform(-cfg.rotate_deg, cfg.rotate_deg)
        else:
            angle_deg = 0.0
        scale = 1.0 + rng.uniform(-cfg.scale_pct, cfg.scale_pct) / 100.0 if cfg.scale_pct > 0 else 1.0
        scale = max(0.5, min(1.8, scale))
        tx = rng.uniform(-cfg.translate_pct, cfg.translate_pct) / 100.0 * w if cfg.translate_pct > 0 else 0.0
        ty = rng.uniform(-cfg.translate_pct, cfg.translate_pct) / 100.0 * h if cfg.translate_pct > 0 else 0.0

    if abs(angle_deg) > 0.01 or abs(scale - 1.0) > 0.005 or abs(tx) > 0.5 or abs(ty) > 0.5:
        cx, cy = w / 2.0, h / 2.0
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        # Forward transform (src -> dst): rotate & scale around center, then translate
        def fwd(x: float, y: float) -> tuple[float, float]:
            dx, dy = x - cx, y - cy
            nx = scale * (cos_a * dx - sin_a * dy) + cx + tx
            ny = scale * (sin_a * dx + cos_a * dy) + cy + ty
            return nx, ny

        # Inverse transform (dst -> src) for PIL.Image.transform(AFFINE)
        inv_s = 1.0 / scale
        a = inv_s * cos_a
        b = inv_s * sin_a
        c = cx - a * (cx + tx) - b * (cy + ty)
        d = -inv_s * sin_a
        e = inv_s * cos_a
        f = cy - d * (cx + tx) - e * (cy + ty)

        im = im.transform((w, h), Image.Transform.AFFINE, (a, b, c, d, e, f), resample=Image.Resampling.BILINEAR, fillcolor=(114, 114, 114))
        xyxy = _transform_corners(xyxy, fwd)

    # 4. Photometric transforms (boxes unchanged)
    if ranges is not None:
        if ranges.get("brightness", {}).get("enabled"):
            factor = max(0.1, _sample_step_range(rng, ranges["brightness"], 1.0))
            im = ImageEnhance.Brightness(im).enhance(factor)
        if ranges.get("contrast", {}).get("enabled"):
            factor = max(0.1, _sample_step_range(rng, ranges["contrast"], 1.0))
            im = ImageEnhance.Contrast(im).enhance(factor)
        if ranges.get("saturation", {}).get("enabled"):
            factor = max(0.0, _sample_step_range(rng, ranges["saturation"], 1.0))
            im = ImageEnhance.Color(im).enhance(factor)
        if ranges.get("hue", {}).get("enabled"):
            hue_val = _sample_step_range(rng, ranges["hue"], 0.0)
            shift = int(round(hue_val / 360.0 * 255.0))
            if shift != 0:
                hsv = np.array(im.convert("HSV"), dtype=np.uint8)
                hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + shift) % 256
                im = Image.fromarray(hsv, mode="HSV").convert("RGB")
        if ranges.get("blur", {}).get("enabled"):
            r = max(0.0, _sample_step_range(rng, ranges["blur"], 0.0))
            if r > 0.1:
                im = im.filter(ImageFilter.GaussianBlur(radius=r))
        if ranges.get("noise", {}).get("enabled"):
            n_std = max(0.0, _sample_step_range(rng, ranges["noise"], 0.0))
            if n_std > 0:
                np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
                arr = np.array(im, dtype=np.float32)
                arr += np_rng.normal(0.0, n_std, size=arr.shape).astype(np.float32)
                im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
    else:
        if cfg.brightness_pct > 0:
            factor = max(0.2, 1.0 + rng.uniform(-cfg.brightness_pct, cfg.brightness_pct) / 100.0)
            im = ImageEnhance.Brightness(im).enhance(factor)

        if cfg.contrast_pct > 0:
            factor = max(0.2, 1.0 + rng.uniform(-cfg.contrast_pct, cfg.contrast_pct) / 100.0)
            im = ImageEnhance.Contrast(im).enhance(factor)

        if cfg.saturation_pct > 0:
            factor = max(0.0, 1.0 + rng.uniform(-cfg.saturation_pct, cfg.saturation_pct) / 100.0)
            im = ImageEnhance.Color(im).enhance(factor)

        if cfg.hue_deg > 0:
            shift = int(round(rng.uniform(-cfg.hue_deg, cfg.hue_deg) / 360.0 * 255.0))
            if shift != 0:
                hsv = np.array(im.convert("HSV"), dtype=np.uint8)
                hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + shift) % 256
                im = Image.fromarray(hsv, mode="HSV").convert("RGB")

        if cfg.blur_radius > 0:
            r = rng.uniform(0.0, cfg.blur_radius)
            if r > 0.1:
                im = im.filter(ImageFilter.GaussianBlur(radius=r))

        if cfg.noise_std > 0:
            np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
            arr = np.array(im, dtype=np.float32)
            arr += np_rng.normal(0.0, float(cfg.noise_std), size=arr.shape).astype(np.float32)
            im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")

    return im, _xyxy_to_boxes(xyxy, w, h)


def preview_augmented_sample(
    entry: ImageEntry,
    id_to_name: dict[int, str],
    cfg: AugmentConfig,
    seed: int | None = None,
) -> tuple[Image.Image, int]:
    """Return a rendered preview PIL image (with transformed boxes drawn) and box count."""
    boxes, _ = parse_label_file(entry.label_path)
    rng = random.Random(cfg.seed if seed is None else seed)
    with Image.open(entry.image_path) as im:
        aug_im, aug_boxes = augment_image_and_boxes(im, boxes, cfg, rng)
    # Render boxes onto a temporary in-memory copy
    from PIL import ImageDraw

    draw = ImageDraw.Draw(aug_im)
    w, h = aug_im.size
    palette = ["#6366f1", "#22c55e", "#f59e0b", "#ec4899", "#06b6d4", "#f97316"]
    for b in aug_boxes:
        x1 = (b.cx - b.w / 2.0) * w
        y1 = (b.cy - b.h / 2.0) * h
        x2 = (b.cx + b.w / 2.0) * w
        y2 = (b.cy + b.h / 2.0) * h
        col = palette[b.class_id % len(palette)]
        draw.rectangle([x1, y1, x2, y2], outline=col, width=2)
        label = id_to_name.get(b.class_id, f"class_{b.class_id}")
        draw.text((x1 + 4, max(0, y1 - 14)), label, fill=col)
    return aug_im, len(aug_boxes)


def generate_offline_augmented_dataset(
    entries: list[ImageEntry],
    class_names: dict[int, str],
    out_root: Path,
    cfg: AugmentConfig,
    source_id: str,
) -> dict[str, Any]:
    """Create a NEW dataset folder where Train has originals + augmented copies, while Val/Test are untouched."""
    copies = max(1, min(10, int(cfg.copies_per_image)))
    out_root = Path(out_root)
    rng = random.Random(int(cfg.seed))

    counts = {"train_original": 0, "train_augmented": 0, "val": 0, "test": 0}
    has_test = False

    for entry in sorted(entries, key=lambda e: (e.split, e.key)):
        flat = entry.key.replace("/", "__")
        img_dst = out_root / "images" / entry.split / f"{flat}{entry.image_path.suffix.lower()}"
        lbl_dst = out_root / "labels" / entry.split / f"{flat}.txt"
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        lbl_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry.image_path, img_dst)
        if entry.has_label:
            shutil.copy2(entry.label_path, lbl_dst)

        if entry.split == "train":
            counts["train_original"] += 1
            boxes, _ = parse_label_file(entry.label_path)
            with Image.open(entry.image_path) as im:
                for c_idx in range(1, copies + 1):
                    aug_im, aug_boxes = augment_image_and_boxes(im, boxes, cfg, rng)
                    aug_stem = f"{flat}_aug{c_idx}"
                    aug_img_path = out_root / "images" / "train" / f"{aug_stem}.jpg"
                    aug_lbl_path = out_root / "labels" / "train" / f"{aug_stem}.txt"
                    aug_im.save(aug_img_path, quality=92)
                    if entry.has_label:
                        write_label_file(aug_lbl_path, aug_boxes)
                    counts["train_augmented"] += 1
        elif entry.split == "val":
            counts["val"] += 1
        elif entry.split == "test":
            counts["test"] += 1
            has_test = True

    yaml_data: dict[str, Any] = {"path": ".", "train": "images/train", "val": "images/val"}
    if has_test:
        yaml_data["test"] = "images/test"
    yaml_data["nc"] = len(class_names)
    yaml_data["names"] = {int(k): str(v) for k, v in sorted(class_names.items())}
    (out_root / "data.yaml").write_text(yaml.safe_dump(yaml_data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    manifest = {
        "source_dataset_id": source_id,
        "config": cfg.to_dict(),
        "counts": counts,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_root / "augment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return counts
