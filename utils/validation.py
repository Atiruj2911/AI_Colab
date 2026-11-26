"""Validation and sanitization utilities."""

from __future__ import annotations

import re
from pathlib import Path

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_identifier(raw: str, fallback: str = "item", max_length: int = 64) -> str:
    """Convert user-provided text into a filesystem-safe identifier."""
    cleaned = _SAFE_NAME_RE.sub("_", (raw or "").strip()).strip("._-")
    if not cleaned:
        return fallback
    return cleaned[:max_length]


def is_within_directory(target: Path, base_dir: Path) -> bool:
    """Return True iff target resolves inside base_dir (prevents path traversal)."""
    try:
        base_resolved = base_dir.resolve()
        target_resolved = target.resolve()
        target_resolved.relative_to(base_resolved)
        return True
    except ValueError:
        return False


def format_bytes(num_bytes: int | float) -> str:
    """Format byte count into human-readable string."""
    size = float(max(0, num_bytes))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def format_duration(seconds: float | None) -> str:
    """Format elapsed/ETA seconds as HH:MM:SS."""
    if seconds is None or seconds < 0:
        return "--:--:--"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
