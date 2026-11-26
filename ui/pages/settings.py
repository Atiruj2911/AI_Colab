"""Settings: runtime facts and workspace cleanup. Only controls with a real effect are editable."""

from __future__ import annotations

import html
from typing import Any

import gradio as gr

from application.app_controller import AppController
from config.defaults import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_IMG_SIZE,
    DEFAULT_MODEL_VARIANT_LABEL,
    DEFAULT_MONITOR_REFRESH_SECONDS,
)
from ui.components.common import render_page_header
from utils.validation import format_bytes


def _runtime_html(controller: AppController) -> str:
    ov = controller.get_system_overview()
    versions = ov["versions"]
    gpu = ov["gpu"]
    rows = [
        ("Python", versions.get("python", "—")),
        ("Platform", versions.get("platform", "—")),
        ("PyTorch", versions.get("pytorch", "—")),
        ("CUDA", versions.get("cuda_version", "—")),
        ("GPU", gpu.get("summary", "—")),
        ("Ultralytics", versions.get("ultralytics", "—")),
        ("Gradio", versions.get("gradio", "—")),
        ("Workspace", ov["workspace_root"]),
        ("Colab", "Yes — temporary disk" if ov["is_colab"] else "No"),
        ("Monitor refresh", f"{DEFAULT_MONITOR_REFRESH_SECONDS:g} s"),
    ]
    body = "".join(
        "<tr>"
        f"<td>{html.escape(label)}</td>"
        f"<td><code>{html.escape(str(value))}</code></td>"
        "</tr>"
        for label, value in rows
    )
    return (
        '<div class="studio-card">'
        '<div style="font-size:15px;font-weight:600;margin-bottom:8px;">Runtime</div>'
        f'<table class="data-table"><tbody>{body}</tbody></table>'
        "</div>"
    )


def _storage_html(controller: AppController) -> str:
    ov = controller.get_system_overview()
    usage = ov["workspace_usage"]
    labels = [
        ("uploads", "Uploads"),
        ("datasets", "Datasets"),
        ("weights", "Weights cache"),
        ("runs", "Training runs"),
        ("exports", "Exports"),
        ("temp", "Temporary files"),
        ("sessions", "Sessions"),
        ("logs", "Logs"),
    ]
    rows = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{html.escape(format_bytes(usage.get(key, 0)))}</td></tr>"
        for key, label in labels
    )
    return (
        '<div class="studio-card">'
        '<div style="font-size:15px;font-weight:600;margin-bottom:8px;">Storage</div>'
        f'<div class="empty-note" style="margin-bottom:8px;">Disk {html.escape(str(ov["disk"].get("summary", "—")))} · '
        f'Workspace {html.escape(format_bytes(usage.get("total", 0)))}</div>'
        f'<table class="data-table"><thead><tr><th>Folder</th><th>Size</th></tr></thead><tbody>{rows}</tbody></table>'
        "</div>"
    )


def _defaults_html() -> str:
    return (
        '<div class="studio-card">'
        '<div style="font-size:15px;font-weight:600;margin-bottom:8px;">Default training parameters</div>'
        '<div class="kv-list">'
        f"<div><span>Model</span> · {html.escape(DEFAULT_MODEL_VARIANT_LABEL)}</div>"
        f"<div><span>Epochs</span> · {DEFAULT_EPOCHS}</div>"
        f"<div><span>Batch size</span> · {DEFAULT_BATCH_SIZE}</div>"
        f"<div><span>Image size</span> · {DEFAULT_IMG_SIZE}</div>"
        "</div>"
        '<div class="empty-note" style="margin-top:8px;">Change these on the Training page for each run. They are not stored as global preferences.</div>'
        "</div>"
    )


def build_settings_page(controller: AppController) -> dict[str, Any]:
    with gr.Column(visible=False) as container:
        gr.HTML(render_page_header("Settings", "Runtime, storage, and the defaults used when a run starts."))
        runtime_html = gr.HTML(value=_runtime_html(controller))
        gr.HTML(value=_defaults_html())
        storage_html = gr.HTML(value=_storage_html(controller))
        with gr.Group(elem_classes=["studio-card"]):
            gr.Markdown("**Clear a workspace folder.** This deletes files and cannot be undone.")
            cleanup_target = gr.Dropdown(
                label="Folder",
                choices=[
                    ("temp/ — preview and inference files", "temp"),
                    ("exports/ — exported checkpoints", "exports"),
                    ("uploads/ — original uploads", "uploads"),
                    ("datasets/ — all datasets", "datasets"),
                    ("runs/ — training history and checkpoints", "runs"),
                ],
                value="temp",
            )
            confirm_cleanup = gr.Checkbox(label="I understand this permanently deletes the selected folder", value=False)
            btn_cleanup = gr.Button("Clear folder", variant="stop")
            cleanup_feedback = gr.Markdown("")

        def _do_cleanup(subdir: str, confirmed: bool):
            ok, msg = controller.clear_workspace_folder(subdir, confirmed)
            prefix = "Cleared. " if ok else "Not cleared. "
            return f"{prefix}{msg}", False, _storage_html(controller)

        def _refresh():
            return _runtime_html(controller), _storage_html(controller)

        btn_cleanup.click(
            fn=_do_cleanup,
            inputs=[cleanup_target, confirm_cleanup],
            outputs=[cleanup_feedback, confirm_cleanup, storage_html],
        )

    return {
        "container": container,
        "refresh_fn": _refresh,
        "refresh_outputs": [runtime_html, storage_html],
    }
