"""Page 3: Annotate — Step 2 in linear workflow (Manual Bounding Box & AI Auto-Label)."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import gradio as gr

from application.app_controller import AppController


def _iframe_html(dataset_id: str | None) -> str:
    query = f"?dataset={quote(dataset_id)}" if dataset_id else ""
    url = f"/annotate{query}"
    return (
        f'<iframe src="{url}" '
        'style="width:100%;height:calc(100vh - 104px);min-height:660px;border:1px solid #233255;'
        'border-radius:12px;background:#080d1a;display:block;"></iframe>'
    )


def _header_html(dataset_id: str | None) -> str:
    query = f"?dataset={quote(dataset_id)}" if dataset_id else ""
    url = f"/annotate{query}"
    return (
        '<div style="display:flex;justify-content:space-between;align-items:center;'
        'padding:6px 10px;background:#0f172a;border:1px solid #233255;border-radius:10px;">'
        '<div style="font-size:13px;color:#cbd5f5;">'
        '<strong>Annotation Workspace</strong> · Drag on canvas to draw boxes '
        '(release to assign or create class · Enter to confirm)'
        "</div>"
        f'<a href="{url}" target="_blank" rel="noopener" '
        'style="display:inline-flex;align-items:center;gap:6px;padding:5px 12px;border-radius:7px;'
        'background:#1a2748;border:1px solid #2b3f6b;color:#e2e8f0;text-decoration:none;'
        'font-size:12px;font-weight:600;">Open Fullscreen</a>'
        "</div>"
    )


def build_annotate_page(controller: AppController) -> dict[str, Any]:
    datasets = [d for d in controller.list_datasets() if d.total_images > 0]
    default_id = datasets[0].dataset_id if datasets else None

    with gr.Column(visible=False) as container:
        with gr.Row(elem_classes=["compact-row"]):
            header_html = gr.HTML(value=_header_html(default_id), scale=9)
            btn_go_prepare = gr.Button("Next: Prepare (Split & Augment)", variant="primary", size="sm", scale=0, min_width=230)
        frame_html = gr.HTML(value=_iframe_html(default_id))

        def _refresh_page():
            sess = controller.session_service.get_or_create_session()
            active = sess.get("active_dataset_id")
            if not active:
                ds_list = [d for d in controller.list_datasets() if d.total_images > 0]
                active = ds_list[0].dataset_id if ds_list else None
            return _header_html(active), _iframe_html(active)

    return {
        "container": container,
        "btn_go_prepare": btn_go_prepare,
        "refresh_fn": _refresh_page,
        "refresh_inputs": [],
        "refresh_outputs": [header_html, frame_html],
    }
