"""Train/Val/Test split into a NEW dataset folder (run before any offline augmentation to avoid leakage)."""

from __future__ import annotations

import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from core.dataset.entries import ImageEntry


def split_entries(
    entries: list[ImageEntry],
    class_names: dict[int, str],
    out_root: Path,
    train_pct: float,
    val_pct: float,
    test_pct: float,
    seed: int,
    source_id: str,
) -> dict[str, Any]:
    if min(train_pct, val_pct, test_pct) < 0:
        raise ValueError("เปอร์เซ็นต์ต้องไม่ติดลบ")
    if abs((train_pct + val_pct + test_pct) - 100.0) > 0.01:
        raise ValueError(f"เปอร์เซ็นต์ Train + Val + Test ต้องรวมเป็น 100 (ตอนนี้ {train_pct + val_pct + test_pct:g})")
    if not entries:
        raise ValueError("ไม่มีภาพใน Dataset ต้นทางให้แบ่ง")

    pool = sorted(entries, key=lambda e: (e.split, e.key))
    random.Random(int(seed)).shuffle(pool)

    n = len(pool)
    n_train = round(n * train_pct / 100.0)
    if test_pct == 0:
        n_val = n - n_train
    else:
        n_val = min(round(n * val_pct / 100.0), n - n_train)
    n_test = n - n_train - n_val
    if n_train == 0 or n_val == 0:
        raise ValueError(
            f"จำนวนภาพ ({n}) ไม่พอสำหรับสัดส่วนที่กำหนด — Train และ Val ต้องมีอย่างน้อย 1 ภาพ"
        )

    out_root = Path(out_root)
    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0, "background": 0}
    used: set[tuple[str, str]] = set()

    for idx, entry in enumerate(pool):
        if idx < n_train:
            dst_split = "train"
        elif idx < n_train + n_val:
            dst_split = "val"
        else:
            dst_split = "test"

        flat = entry.key.replace("/", "__")
        name = flat
        suffix_no = 1
        while (dst_split, name) in used:
            name = f"{flat}_{suffix_no}"
            suffix_no += 1
        used.add((dst_split, name))

        img_dst = out_root / "images" / dst_split / f"{name}{entry.image_path.suffix.lower()}"
        lbl_dst = out_root / "labels" / dst_split / f"{name}.txt"
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        lbl_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry.image_path, img_dst)
        if entry.has_label:
            shutil.copy2(entry.label_path, lbl_dst)
        else:
            lbl_dst.write_text("", encoding="utf-8")
            counts["background"] += 1
        counts[dst_split] += 1

    yaml_data: dict[str, Any] = {"path": ".", "train": "images/train", "val": "images/val"}
    if n_test > 0:
        yaml_data["test"] = "images/test"
    yaml_data["nc"] = len(class_names)
    yaml_data["names"] = {int(k): str(v) for k, v in sorted(class_names.items())}
    (out_root / "data.yaml").write_text(yaml.safe_dump(yaml_data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    manifest = {
        "source_dataset_id": source_id,
        "seed": int(seed),
        "ratios_pct": {"train": train_pct, "val": val_pct, "test": test_pct},
        "counts": counts,
        "source_images": n,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_root / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return counts
