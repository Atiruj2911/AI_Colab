"""Structured logging helpers keeping technical logs separate from UI messages."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(
    name: str = "ai_training_studio",
    log_file: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure a logger with console output and optional file sink."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(log_file.resolve())
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == resolved for h in logger.handlers):
            file_handler = logging.FileHandler(resolved, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
