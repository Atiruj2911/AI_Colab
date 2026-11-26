"""Models: read-only catalog of training runs and checkpoint files.

Download and Test use the existing checkpoint export and Training workspace.
Delete and architecture search are not implemented in the backend.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr

from application.app_controller import AppController
from core.schemas import JobState
from ui.components.common import render_page_header
from utils.validation import format_bytes


def _short_time(value: str | None) -> str:
    if not value:
        return "—"
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return text[:16]


def _dataset_name(controller: AppController, dataset_id: str) -> str:
    report = controller.get_dataset(dataset_id)
    return report.dataset_name if report else dataset_id or "—"


def _file_size(path: str | None) -> str:
    if not path:
        return "—"
    file_path = Path(path)
    if not file_path.is_file():
        return "—"
    return format_bytes(file_path.stat().st_size)


def _runs(controller: AppController) -> list[JobState]:
    return controller.list_runs()


def _choices(controller: AppController) -> list[tuple[str, str]]:
    return [(f"{job.run_name} ({job.model_variant})", job.run_id) for job in _runs(controller)]


def _table(controller: AppController, query: str) -> str:
    needle = (query or "").strip().lower()
    rows: list[str] = []
    for job in _runs(controller):
        ds_name = _dataset_name(controller, job.dataset_id)
        blob = " ".join([job.run_name, job.model_variant, ds_name, job.status]).lower()
        if needle and needle not in blob:
            continue
        mp = f"{job.best_map50:.3f}" if job.best_map50 is not None else "—"
        rows.append(
            "<tr>"
            f"<td>{html.escape(job.run_name)}</td>"
            f"<td>{html.escape(job.model_variant)}</td>"
            f"<td>{html.escape(ds_name)}</td>"
            f"<td>{html.escape(job.status)}</td>"
            f"<td>{mp}</td>"
            f"<td>{_short_time(job.started_at)}</td>"
            f"<td>{html.escape(_file_size(job.best_checkpoint_path))}</td>"
            "</tr>"
        )
    if not rows:
        body = '<tr><td colspan="7" class="empty-note">No models yet. A row appears after a training run creates a checkpoint.</td></tr>'
    else:
        body = "".join(rows)
    return (
        '<div class="studio-card" style="padding:0;overflow:auto;">'
        '<table class="data-table"><thead><tr>'
        "<th>Model</th><th>Architecture</th><th>Dataset</th><th>Run status</th>"
        "<th>Best mAP50</th><th>Created</th><th>best.pt</th>"
        f"</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _details(controller: AppController, run_id: str | None) -> str:
    job = next((item for item in _runs(controller) if item.run_id == run_id), None)
    if job is None:
        return '<div class="studio-card"><div class="empty-note">Select a model to see its training configuration and checkpoint files.</div></div>'
    artifacts = controller.list_run_artifacts(job.run_id)
    files = "".join(
        f"<tr><td>{html.escape(item.kind)}</td><td>{html.escape(item.name)}</td><td>{html.escape(format_bytes(item.size_bytes))}</td></tr>"
        for item in artifacts
    ) or '<tr><td colspan="3" class="empty-note">No checkpoint files recorded.</td></tr>'
    mp50 = f"{job.best_map50:.4f}" if job.best_map50 is not None else "—"
    mp95 = f"{job.best_map50_95:.4f}" if job.best_map50_95 is not None else "—"
    return (
        '<div class="studio-card">'
        f'<div style="font-size:16px;font-weight:600;margin-bottom:8px;">{html.escape(job.run_name)}</div>'
        '<div class="kv-list">'
        f"<div><span>Architecture</span> · {html.escape(job.model_family)} {html.escape(job.model_variant)} · {html.escape(job.task)}</div>"
        f"<div><span>Dataset</span> · {html.escape(_dataset_name(controller, job.dataset_id))}</div>"
        f"<div><span>Epochs</span> · {job.current_epoch} / {job.total_epochs} · best epoch {job.best_epoch or '—'}</div>"
        f"<div><span>mAP50</span> · {mp50} · <span>mAP50-95</span> · {mp95}</div>"
        f"<div><span>Started</span> · {_short_time(job.started_at)}</div>"
        "</div>"
        '<div style="font-size:13px;font-weight:600;margin:12px 0 6px;">Checkpoint files</div>'
        '<table class="data-table"><thead><tr><th>Kind</th><th>File</th><th>Size</th></tr></thead>'
        f"<tbody>{files}</tbody></table>"
        '<div class="empty-note" style="margin-top:8px;">Training controls stay on the Training page. Test Model opens that workspace with this run selected.</div>'
        "</div>"
    )


def build_models_page(controller: AppController) -> dict[str, Any]:
    choices = _choices(controller)
    initial = choices[0][1] if choices else None

    with gr.Column(visible=False) as container:
        gr.HTML(render_page_header("Models", "Checkpoints produced by training runs."))
        with gr.Row(elem_classes=["compact-row"]):
            query = gr.Textbox(placeholder="Search name, dataset, or architecture", show_label=False, scale=5)
            run_dd = gr.Dropdown(label="Model", choices=choices, value=initial, scale=4)
            btn_test = gr.Button("Test Model", variant="primary", scale=2)
        table_html = gr.HTML(value=_table(controller, ""))
        detail_html = gr.HTML(value=_details(controller, initial))
        with gr.Row(elem_classes=["compact-row"]):
            btn_download = gr.Button("Download best.pt", variant="secondary", size="sm")
            download_file = gr.File(label="Download", interactive=False)
        download_msg = gr.Markdown("")

        def _on_query(text: str):
            return _table(controller, text)

        def _on_select(run_id: str | None):
            return _details(controller, run_id)

        def _refresh(current: str | None, text: str):
            fresh = _choices(controller)
            selected = current if any(item[1] == current for item in fresh) else (fresh[0][1] if fresh else None)
            return gr.update(choices=fresh, value=selected), _table(controller, text), _details(controller, selected)

        def _download(run_id: str | None):
            if not run_id:
                return None, "Select a model first."
            try:
                info = controller.export_pt_checkpoint(run_id=run_id, which="best", include_bundle=False)
                return info["file_path"], f"Ready: `{info['filename']}` ({format_bytes(info['size_bytes'])})."
            except Exception as exc:
                return None, f"Download failed: {exc}"

        query.change(fn=_on_query, inputs=[query], outputs=[table_html])
        run_dd.change(fn=_on_select, inputs=[run_dd], outputs=[detail_html])
        btn_download.click(fn=_download, inputs=[run_dd], outputs=[download_file, download_msg])

    return {
        "container": container,
        "run_dd": run_dd,
        "btn_test": btn_test,
        "refresh_fn": _refresh,
        "refresh_inputs": [run_dd, query],
        "refresh_outputs": [run_dd, table_html, detail_html],
    }
