"""Training metrics loader and summary helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from infrastructure.process.state_store import read_jsonl


def load_epoch_metrics(metrics_path: str | Path | None) -> list[dict[str, Any]]:
    """Load epoch metric history from metrics.jsonl."""
    if not metrics_path:
        return []
    return read_jsonl(Path(metrics_path))


def extract_metric_series(records: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Convert raw epoch records into plotting series."""
    epochs: list[int] = []
    train_loss: list[float | None] = []
    val_loss: list[float | None] = []
    map50: list[float | None] = []
    map50_95: list[float | None] = []
    precision: list[float | None] = []
    recall: list[float | None] = []

    for rec in records:
        ep = rec.get("epoch")
        if ep is None:
            continue
        epochs.append(int(ep))
        train_loss.append(rec.get("train_loss_total"))
        val_loss.append(rec.get("val_loss_total"))
        map50.append(rec.get("map50"))
        map50_95.append(rec.get("map50_95"))
        precision.append(rec.get("precision"))
        recall.append(rec.get("recall"))

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "map50": map50,
        "map50_95": map50_95,
        "precision": precision,
        "recall": recall,
    }
