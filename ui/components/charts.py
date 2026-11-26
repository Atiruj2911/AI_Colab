"""Light-workstation Matplotlib training charts (Loss, mAP, Precision, Recall)."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _styled_figure(title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(6.8, 3.3), dpi=110)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.set_title(title, color="#18202C", fontsize=13, fontweight="medium", pad=10)
    ax.set_xlabel("Epoch", color="#566173", fontsize=10)
    ax.set_ylabel(ylabel, color="#566173", fontsize=10)
    ax.tick_params(colors="#566173", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#D8DDE4")
    ax.grid(True, color="#E6E9EE", linestyle="--", linewidth=0.7, alpha=1)
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
            color="#566173",
            fontsize=10,
        )
    else:
        if y_zero_one:
            ax.set_ylim(-0.02, 1.05)
        ax.legend(
            facecolor="#FFFFFF",
            edgecolor="#D8DDE4",
            labelcolor="#18202C",
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
            ("Train Loss (box+cls+dfl)", series.get("train_loss", []), "#1F56C9"),
            ("Val Loss (box+cls+dfl)", series.get("val_loss", []), "#A86400"),
        ],
        title="Training & Validation Loss",
        ylabel="Total Loss",
        y_zero_one=False,
    )

    map_fig = _plot_series(
        epochs,
        [
            ("mAP50", series.get("map50", []), "#0F7F63"),
            ("mAP50-95", series.get("map50_95", []), "#1F56C9"),
        ],
        title="Mean Average Precision (mAP)",
        ylabel="Score (0–1)",
        y_zero_one=True,
    )

    precision_fig = _plot_series(
        epochs,
        [("Precision", series.get("precision", []), "#1F56C9")],
        title="Precision",
        ylabel="Score (0–1)",
        y_zero_one=True,
    )

    recall_fig = _plot_series(
        epochs,
        [("Recall", series.get("recall", []), "#0F7F63")],
        title="Recall",
        ylabel="Score (0–1)",
        y_zero_one=True,
    )

    return loss_fig, map_fig, precision_fig, recall_fig
