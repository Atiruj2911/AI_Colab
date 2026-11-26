"""Annotation page shell. The canvas, tools, and save behavior live in annotator.html."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import gradio as gr

from application.app_controller import AppController


def _iframe_html(dataset_id: str | None) -> str:
    if not dataset_id:
        return (
            '<div class="studio-card">'
            '<div style="font-size:15px;font-weight:600;margin-bottom:4px;">No dataset open</div>'
            '<div class="empty-note">Import or select a dataset, then use Open on the Datasets page. '
            "The canvas stays empty until a dataset is loaded.</div>"
            "</div>"
        )
    query = f"?dataset={quote(dataset_id)}"
    return (
        f'<iframe src="/annotate{query}" title="Annotation workspace" '
        'style="width:100%;height:calc(100vh - 88px);min-height:680px;border:1px solid #E2E8F0;'
        'border-radius:8px;background:#2A2E35;display:block;"></iframe>'
    )


def _header_html(dataset_id: str | None) -> str:
    if not dataset_id:
        return ""
    url = f"/annotate?dataset={quote(dataset_id)}"
    return (
        '<div class="page-header" style="min-height:40px;margin-bottom:8px;">'
        '<div><h2 class="page-title" style="font-size:20px;">Annotation</h2>'
        '<div class="page-subtitle">Draw and edit boxes. Shortcuts are on each tool.</div></div>'
        f'<a href="{url}" target="_blank" rel="noopener" '
        'style="font-size:13px;font-weight:600;color:#2563EB;text-decoration:none;">Open fullscreen</a>'
        "</div>"
    )


def build_annotate_page(controller: AppController) -> dict[str, Any]:
    datasets = [d for d in controller.list_datasets() if d.total_images > 0]
    default_id = datasets[0].dataset_id if datasets else None

    with gr.Column(visible=False) as container:
        header_html = gr.HTML(value=_header_html(default_id))
        frame_html = gr.HTML(value=_iframe_html(default_id))

        def _refresh_page():
            sess = controller.session_service.get_or_create_session()
            active = sess.get("active_dataset_id")
            known = {d.dataset_id for d in controller.list_datasets() if d.total_images > 0}
            if active not in known:
                active = next(iter(known), None)
            return _header_html(active), _iframe_html(active)

    return {
        "container": container,
        "refresh_fn": _refresh_page,
        "refresh_inputs": [],
        "refresh_outputs": [header_html, frame_html],
    }
