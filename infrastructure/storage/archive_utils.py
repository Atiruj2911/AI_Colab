"""Safe archive extraction utilities protecting against path traversal and zip bombs."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from utils.validation import format_bytes, is_within_directory


class ArchiveSecurityError(ValueError):
    """Raised when an uploaded archive violates security or size constraints."""


def safe_extract_zip(
    zip_path: Path,
    dest_dir: Path,
    max_archive_mb: int = 2048,
    max_extracted_mb: int = 8192,
    max_files: int = 100_000,
) -> Path:
    """Safely extract a ZIP archive into dest_dir with Zip Slip & size protections."""
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ ZIP: {zip_path}")

    archive_size = zip_path.stat().st_size
    if archive_size > max_archive_mb * 1024 * 1024:
        raise ArchiveSecurityError(
            f"ไฟล์ ZIP มีขนาดใหญ่เกินกำหนด ({format_bytes(archive_size)} > {max_archive_mb} MB)"
        )

    if not zipfile.is_zipfile(zip_path):
        raise ArchiveSecurityError("ไฟล์ที่อัปโหลดไม่ใช่ไฟล์ ZIP ที่ถูกต้อง")

    max_extracted_bytes = max_extracted_mb * 1024 * 1024
    total_uncompressed = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = zf.infolist()
        if len(infos) > max_files:
            raise ArchiveSecurityError(
                f"ไฟล์ ZIP มีจำนวนไฟล์มากเกินกำหนด ({len(infos):,} > {max_files:,} ไฟล์)"
            )

        for info in infos:
            # Skip macOS metadata artifacts
            if info.filename.startswith("__MACOSX/") or "/__MACOSX/" in info.filename:
                continue

            # Guard against absolute paths or '..' traversal
            member_path = Path(info.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ArchiveSecurityError(
                    f"ตรวจพบ path ที่ไม่ปลอดภัยใน ZIP (path traversal): {info.filename}"
                )

            target_path = (dest_dir / member_path).resolve()
            if not is_within_directory(target_path, dest_dir):
                raise ArchiveSecurityError(
                    f"ตรวจพบ path ออกนอกโฟลเดอร์ปลายทาง: {info.filename}"
                )

            # Reject symlinks inside zip (Unix mode 0o120000)
            unix_mode = (info.external_attr >> 16) & 0o170000
            if unix_mode == 0o120000:
                raise ArchiveSecurityError(
                    f"ไม่อนุญาตให้มี symbolic link ภายใน ZIP: {info.filename}"
                )

            total_uncompressed += info.file_size
            if total_uncompressed > max_extracted_bytes:
                raise ArchiveSecurityError(
                    f"ขนาดข้อมูลหลังแตกไฟล์เกินกำหนด (> {max_extracted_mb} MB)"
                )

        # Extract validated entries safely chunk-by-chunk
        extracted_bytes = 0
        for info in infos:
            if info.filename.startswith("__MACOSX/") or "/__MACOSX/" in info.filename:
                continue
            target_path = (dest_dir / info.filename).resolve()
            if info.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, target_path.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    extracted_bytes += len(chunk)
                    if extracted_bytes > max_extracted_bytes:
                        raise ArchiveSecurityError(
                            f"ขนาดข้อมูลที่แตกจริงเกินกำหนด (> {max_extracted_mb} MB)"
                        )
                    dst.write(chunk)

    # Unwrap single top-level directory if the ZIP wrapped everything inside one folder
    entries = [p for p in dest_dir.iterdir() if p.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return dest_dir


def copy_uploaded_files_into_dir(file_paths: list[Path], dest_dir: Path) -> Path:
    """Copy a list of individually uploaded files (data.yaml, images, labels) into structured folders."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    images_dir = dest_dir / "images" / "train"
    labels_dir = dest_dir / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    from config.defaults import ALLOWED_IMAGE_EXTENSIONS, ALLOWED_LABEL_EXTENSIONS, ALLOWED_YAML_EXTENSIONS

    for raw in file_paths:
        src = Path(raw)
        if not src.exists() or not src.is_file():
            continue
        ext = src.suffix.lower()
        if ext in ALLOWED_YAML_EXTENSIONS:
            shutil.copy2(src, dest_dir / src.name)
        elif ext in ALLOWED_IMAGE_EXTENSIONS:
            shutil.copy2(src, images_dir / src.name)
        elif ext in ALLOWED_LABEL_EXTENSIONS:
            shutil.copy2(src, labels_dir / src.name)
        else:
            # Preserve unknown files rather than silently discarding them
            other_dir = dest_dir / "other_files"
            other_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, other_dir / src.name)

    return dest_dir
