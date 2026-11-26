"""Home: compact project overview. Values come from the live controller."""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr

from application.app_controller import AppController
from config.defaults import APP_NAME
from ui.components.common import render_kpi_cards, render_status_badge, render_temp_storage_warning
from utils.validation import format_bytes


def _short_time(value: str | None) -> str:
    if not value:
        return "—"
    text = str(value).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return text[:16]


def _dataset_mtime(root: str) -> str:
    path = Path(root)
    if not path.exists():
        return "—"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def _render_home(controller: AppController) -> str:
    ov = controller.get_system_overview()
    versions = ov["versions"]
    gpu = ov["gpu"]
    disk = ov["disk"]
    job = ov["active_job"]
    datasets = [ds for ds in controller.list_datasets() if ds.total_images > 0]
    runs = controller.list_runs()

    vram = "—"
    if gpu.get("vram_total_mb"):
        vram = f"{gpu.get('vram_used_mb', 0):.0f} / {gpu['vram_total_mb']:.0f} MB"
    elif not gpu.get("available"):
        vram = "No GPU"

    cards = render_kpi_cards(
        [
            ("Datasets", str(len(datasets)), "Available", False),
            ("Training Runs", str(len(runs)), "In this workspace", False),
            ("Models", str(ov["checkpoint_count"]), "Checkpoint files", False),
            ("Runtime", "Running" if job and job.status == "running" else (job.status.title() if job else "Idle"), "Current job", False),
        ]
    )

    run_rows = []
    for item in runs[:5]:
        run_rows.append(
            "<tr>"
            f"<td>{html.escape(item.run_name)}</td>"
            f"<td>{html.escape(item.status)}</td>"
            f"<td>{item.current_epoch}/{item.total_epochs}</td>"
            f"<td>{_short_time(item.updated_at)}</td>"
            "</tr>"
        )
    ds_rows = []
    ordered = sorted(datasets, key=lambda d: Path(d.dataset_root).stat().st_mtime if Path(d.dataset_root).exists() else 0, reverse=True)
    for ds in ordered[:5]:
        ds_rows.append(
            "<tr>"
            f"<td>{html.escape(ds.dataset_name)}</td>"
            f"<td>{ds.total_images:,}</td>"
            f"<td>{len(ds.class_names)}</td>"
            f"<td>{_dataset_mtime(ds.dataset_root)}</td>"
            "</tr>"
        )

    def _table(headers: list[str], rows: list[str], empty: str) -> str:
        head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
        body = "".join(rows) if rows else f'<tr><td colspan="{len(headers)}" class="empty-note">{html.escape(empty)}</td></tr>'
        return f'<table class="data-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'

    activity = (
        '<div class="studio-card">'
        '<div style="font-size:15px;font-weight:600;margin-bottom:8px;">Recent training runs</div>'
        + _table(["Run", "Status", "Epoch", "Updated"], run_rows, "No training runs yet. Start one from Training.")
        + '<div style="font-size:15px;font-weight:600;margin:16px 0 8px;">Latest datasets</div>'
        + _table(["Dataset", "Images", "Classes", "Modified"], ds_rows, "No datasets yet. Import one from Datasets.")
        + "</div>"
    )

    runtime = (
        '<div class="studio-card">'
        '<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px;">'
        '<div style="font-size:15px;font-weight:600;">Runtime environment</div>'
        + render_status_badge(job.status if job else None)
        + "</div>"
        '<div class="kv-list">'
        f"<div><span>GPU</span><br>{html.escape(str(gpu.get('name', '—')))}</div>"
        f"<div><span>VRAM</span><br>{html.escape(vram)}</div>"
        f"<div><span>Python</span><br>{html.escape(str(versions.get('python', '—')))}</div>"
        f"<div><span>PyTorch</span><br>{html.escape(str(versions.get('pytorch', '—')))}</div>"
        f"<div><span>Disk free</span><br>{html.escape(format_bytes(int(disk.get('free_bytes') or 0)))}</div>"
        "</div></div>"
    )

    warning = render_temp_storage_warning(compact=True) if ov.get("is_colab") else ""
    header = (
        '<div class="page-header">'
        "<div>"
        f'<h2 class="page-title">{html.escape(APP_NAME)}</h2>'
        '<div class="page-subtitle">YOLO11 workstation for datasets, annotation, preparation, and training.</div>'
        "</div>"
        f"<div>{render_status_badge(job.status if job else None)}</div>"
        "</div>"
    )
    return header + cards + warning + f'<div class="dash-grid">{activity}{runtime}</div>'


def build_home_page(controller: AppController) -> dict[str, Any]:
    with gr.Column(visible=True) as container:
        overview_html = gr.HTML(value=_render_home(controller))

    return {
        "container": container,
        "refresh_fn": lambda: _render_home(controller),
        "refresh_outputs": [overview_html],
    }
