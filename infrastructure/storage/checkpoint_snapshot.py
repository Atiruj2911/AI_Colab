"""Safe atomic checkpoint snapshot copy (used for Testing & Export while Training runs)."""

from __future__ import annotations

import os
import shutil
import time
import zipfile
from datetime import datetime
from pathlib import Path


def _is_valid_pt_archive(path: Path) -> bool:
    """PyTorch >=1.6 .pt checkpoints are ZIP archives; verify header & central directory are complete."""
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return len(zf.namelist()) > 0
    except zipfile.BadZipFile:
        # Older legacy pickle format fallback: non-empty file is acceptable
        return path.stat().st_size > 4096
    except OSError:
        return False


def safe_snapshot_checkpoint(
    src_path: str | Path,
    dest_dir: str | Path,
    prefix: str = "snapshot",
    retries: int = 5,
    stability_wait_sec: float = 0.2,
) -> Path:
    """Safely copy a .pt checkpoint without disturbing a live training worker.

    Verifies:
    1. Source file exists and is non-empty.
    2. File size and mtime are stable across a short window (not actively being overwritten).
    3. Atomic copy via temporary file -> os.replace.
    4. Resulting copy passes archive integrity check.
    """
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"ยังไม่พบไฟล์ Checkpoint: {src.name} (Training อาจยังไม่ถึงจุดที่บันทึกไฟล์นี้)")

    dest_root = Path(dest_dir)
    dest_root.mkdir(parents=True, exist_ok=True)

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            st1 = src.stat()
            if st1.st_size < 1024:
                time.sleep(stability_wait_sec)
                continue
            time.sleep(stability_wait_sec)
            st2 = src.stat()
            if st1.st_size != st2.st_size or st1.st_mtime_ns != st2.st_mtime_ns:
                # Worker is actively writing right now — retry briefly
                continue

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:18]
            final_path = dest_root / f"{prefix}_{stamp}.pt"
            tmp_path = dest_root / f".tmp_{prefix}_{stamp}.pt"
            shutil.copy2(src, tmp_path)
            if not _is_valid_pt_archive(tmp_path):
                tmp_path.unlink(missing_ok=True)
                time.sleep(stability_wait_sec)
                continue
            os.replace(tmp_path, final_path)
            return final_path
        except Exception as exc:
            last_err = exc
            time.sleep(stability_wait_sec)

    raise RuntimeError(
        f"ไม่สามารถคัดลอก Checkpoint '{src.name}' ได้ในขณะนี้ (ไฟล์อาจกำลังถูกเขียนโดย Training Worker — กรุณาลองกดอีกครั้ง)"
        + (f": {last_err}" if last_err else "")
    )
