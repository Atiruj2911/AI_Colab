"""Dark-navy Matplotlib training charts (Loss, mAP, Precision, Recall)."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _styled_figure(title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(6.8, 3.3), dpi=110)
    fig.patch.set_facecolor("#10192e")
    ax.set_facecolor("#0b1224")
    ax.set_title(title, color="#f1f5ff", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch", color="#93a4cc", fontsize=9.5)
    ax.set_ylabel(ylabel, color="#93a4cc", fontsize=9.5)
    ax.tick_params(colors="#93a4cc", labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color("#25365e")
    ax.grid(True, color="#1d2c4f", linestyle="--", linewidth=0.7, alpha=0.8)
    return fig, ax


def _plot_series(
    epochs: list[int],
    lines: list[tuple[str, list[float | None], str]],
    title: str,
    ylabel: str,
    y_zero_one: bool = False,
):
    fig, ax = _styled_figure(title, ylabel)
    plotted_any = False

    for label, values, color in lines:
        xs = [e for e, v in zip(epochs, values) if v is not None]
        ys = [float(v) for v in values if v is not None]
        if xs and ys:
            ax.plot(xs, ys, label=label, color=color, linewidth=2.2, marker="o", markersize=3.5)
            plotted_any = True

    if not plotted_any:
        ax.text(
            0.5,
            0.5,
            "No epoch metrics yet\n(Charts appear after Epoch 1)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#7d90bc",
            fontsize=10,
        )
    else:
        if y_zero_one:
            ax.set_ylim(-0.02, 1.05)
        ax.legend(
            facecolor="#131e38",
            edgecolor="#2a3d69",
            labelcolor="#e5ecff",
            fontsize=8.5,
            loc="best",
        )

    fig.tight_layout()
    return fig


def build_training_charts(series: dict[str, list[Any]]) -> tuple[Any, Any, Any, Any]:
    """Return (loss_fig, map_fig, precision_fig, recall_fig) from real epoch metrics."""
    epochs = series.get("epochs", [])

    loss_fig = _plot_series(
        epochs,
        [
            ("Train Loss (box+cls+dfl)", series.get("train_loss", []), "#38bdf8"),
            ("Val Loss (box+cls+dfl)", series.get("val_loss", []), "#fb923c"),
        ],
        title="Training & Validation Loss",
        ylabel="Total Loss",
        y_zero_one=False,
    )

    map_fig = _plot_series(
        epochs,
        [
            ("mAP50", series.get("map50", []), "#34d399"),
            ("mAP50-95", series.get("map50_95", []), "#818cf8"),
        ],
        title="Mean Average Precision (mAP)",
        ylabel="Score (0–1)",
        y_zero_one=True,
    )

    precision_fig = _plot_series(
        epochs,
        [("Precision", series.get("precision", []), "#a78bfa")],
        title="Precision",
        ylabel="Score (0–1)",
        y_zero_one=True,
    )

    recall_fig = _plot_series(
        epochs,
        [("Recall", series.get("recall", []), "#2dd4bf")],
        title="Recall",
        ylabel="Score (0–1)",
        y_zero_one=True,
    )

    return loss_fig, map_fig, precision_fig, recall_fig
