"""Atomic JSON / JSONL persistence helpers safe for concurrent worker writes and UI reads."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON via temp file + os.replace with brief Windows retry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)

    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(serialized)
            f.flush()
            os.fsync(f.fileno())

        last_err: Exception | None = None
        for _ in range(8):
            try:
                os.replace(tmp_path, path)
                return
            except PermissionError as exc:
                last_err = exc
                time.sleep(0.05)
        if last_err is not None:
            raise last_err
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def read_json(path: Path) -> dict[str, Any] | None:
    """Safely read JSON file, returning None if missing or momentarily unreadable."""
    path = Path(path)
    if not path.exists():
        return None
    for _ in range(5):
        try:
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                return None
            return json.loads(text)
        except (PermissionError, json.JSONDecodeError):
            time.sleep(0.04)
    return None


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON record line to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all valid JSON objects from a JSONL file."""
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                    if isinstance(obj, dict):
                        records.append(obj)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return records


def tail_text_file(path: Path, max_lines: int = 250) -> str:
    """Return the last N lines of a text log file."""
    path = Path(path)
    if not path.exists():
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:])
    except OSError:
        return ""
