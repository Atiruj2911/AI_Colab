"""Page 4: Prepare — Step 3 in linear workflow (Train/Val/Test Split -> Offline Augmentation)."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import gradio as gr

from application.app_controller import AppController
from core.schemas import DatasetValidationReport
from ui.components.common import (
    render_dataset_summary_card,
    render_page_header,
    render_validation_issues_markdown,
)


def _dataset_choices(controller: AppController) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for ds in controller.list_datasets():
        if ds.total_images <= 0:
            continue
        status = "Ready" if ds.is_valid else "Needs Split"
        label = f"{ds.dataset_name} ({ds.total_images} images, {len(ds.class_names)} classes — {status})"
        choices.append((label, ds.dataset_id))
    return choices


def _format_summary(report: DatasetValidationReport | None) -> tuple[str, str]:
    if report is None:
        return render_dataset_summary_card(None), render_validation_issues_markdown(None)
    return render_dataset_summary_card(report), render_validation_issues_markdown(report)


def _aug_iframe_html(ds_id: str | None) -> str:
    q = f"?dataset={quote(ds_id)}" if ds_id else ""
    return (
        f'<iframe src="/aug-preview{q}" '
        'style="width:100%;height:calc(100vh - 210px);min-height:620px;border:1px solid #233255;border-radius:10px;background:#0b1224;display:block;">'
        "</iframe>"
    )


def build_prepare_page(controller: AppController) -> dict[str, Any]:
    initial_choices = _dataset_choices(controller)
    initial_id = initial_choices[0][1] if initial_choices else None
    initial_report = controller.get_dataset(initial_id) if initial_id else None
    init_summary, init_issues = _format_summary(initial_report)

    with gr.Column(visible=False) as container:
        gr.HTML(
            render_page_header(
                "Prepare",
                "Step 1: Split annotated images into Train / Val / Test (prevents data leakage)  →  Step 2: Apply Offline Augmentation to Train set only.",
            )
        )

        with gr.Row(elem_classes=["compact-row"]):
            dataset_dropdown = gr.Dropdown(
                label="Dataset",
                choices=initial_choices,
                value=initial_id,
                interactive=True,
                scale=7,
            )
            btn_refresh_ds = gr.Button("Refresh", variant="secondary", size="sm", scale=1)
            btn_go_training = gr.Button("Next: Training", variant="primary", size="sm", scale=2)

        with gr.Tabs() as prepare_tabs:
            # ---- Tab 1: Train / Val / Test Split
            with gr.Tab("Train / Val / Test Split", id="tab_split"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        summary_html = gr.HTML(value=init_summary)
                        with gr.Row(elem_classes=["compact-row"]):
                            btn_revalidate = gr.Button("Re-validate", variant="secondary", size="sm", scale=2)
                            txt_rename_ds = gr.Textbox(placeholder="Rename dataset...", show_label=False, scale=3)
                            btn_rename_ds = gr.Button("Rename", size="sm", scale=1)
                        rename_ds_msg = gr.Markdown("")
                        issues_md = gr.Markdown(value=init_issues)

                    with gr.Column(scale=7):
                        gr.Markdown(
                            "### Train / Val / Test Split\n"
                            "Split original images **before** offline augmentation so the same image never leaks across Train and Validation."
                        )
                        with gr.Row():
                            train_pct = gr.Number(label="Train %", value=70, minimum=0, maximum=100)
                            val_pct = gr.Number(label="Val %", value=20, minimum=0, maximum=100)
                            test_pct = gr.Number(label="Test %", value=10, minimum=0, maximum=100)
                        with gr.Row():
                            seed_num = gr.Number(label="Random Seed", value=42, precision=0)
                            new_split_name = gr.Textbox(label="New Dataset Name (optional)", placeholder="e.g. board_split_v1")
                        chk_drop_dup = gr.Checkbox(label="Remove duplicate images automatically", value=True)
                        with gr.Row(elem_classes=["compact-row"]):
                            btn_preview_split = gr.Button("Generate Split", variant="secondary", size="sm")
                            btn_split = gr.Button("Apply Split", variant="primary", size="sm")
                        split_md = gr.Markdown("")

            # ---- Tab 2: Offline Augmentation (CiRA-style 60 FPS Live Studio)
            with gr.Tab("Offline Augmentation", id="tab_augment"):
                aug_preview_html = gr.HTML(value=_aug_iframe_html(initial_id))

        # ---- Handlers
        def _on_ds_change(ds_id: str | None):
            report = controller.get_dataset(ds_id) if ds_id else None
            if report:
                controller.session_service.update_session(active_dataset_id=report.dataset_id)
            s_html, i_md = _format_summary(report)
            return s_html, i_md, _aug_iframe_html(ds_id)

        def _handle_revalidate(ds_id: str | None):
            if not ds_id:
                return _format_summary(None)
            report = controller.dataset_service.revalidate_dataset(ds_id)
            return _format_summary(report)

        def _handle_rename_ds(ds_id: str | None, new_name: str):
            if not ds_id:
                return "Select a dataset first.", gr.update(), ""
            try:
                rep = controller.dataset_service.rename_dataset(ds_id, new_name)
                return f"Renamed to **`{rep.dataset_name}`**.", gr.update(choices=_dataset_choices(controller), value=ds_id), ""
            except Exception as exc:
                return f"Error: {exc}", gr.update(), new_name

        def _handle_preview_split(ds_id, tp, vp, sp, drop_dup):
            if not ds_id:
                return "Select a dataset first."
            try:
                c = controller.dataset_service.preview_split(ds_id, float(tp or 0), float(vp or 0), float(sp or 0), bool(drop_dup))
                return (
                    f"**Preview Split ({c['total']} images):** "
                    f"Train **{c['train']}** · Val **{c['val']}** · Test **{c['test']}** "
                    f"(excluded corrupt: {c['excluded_corrupt']}, duplicates: {c['excluded_duplicate']}). "
                    "Click **Apply Split** to create the new dataset."
                )
            except Exception as exc:
                return f"Error: {exc}"

        def _handle_split(ds_id, tp, vp, sp, seed, drop_dup, name):
            if not ds_id:
                return "Select a dataset first.", gr.update()
            try:
                new_report, counts = controller.annotation_service.create_split_dataset(
                    dataset_id=ds_id,
                    train_pct=float(tp or 0),
                    val_pct=float(vp or 0),
                    test_pct=float(sp or 0),
                    seed=int(seed or 0),
                    drop_duplicates=bool(drop_dup),
                    new_name=name or None,
                )
                controller.session_service.update_session(active_dataset_id=new_report.dataset_id)
                md = (
                    f"Created split dataset **`{new_report.dataset_name}`** (Ready for Augmentation or Training).\n\n"
                    f"- Train: {counts['train']} · Val: {counts['val']} · Test: {counts['test']} images\n"
                    f"- Excluded corrupt: {counts['excluded_corrupt']} · duplicates: {counts['excluded_duplicate']}"
                )
                return md, gr.update(choices=_dataset_choices(controller), value=new_report.dataset_id)
            except Exception as exc:
                return f"Split failed: {exc}", gr.update()

        def _refresh_page(current_ds_id: str | None = None):
            choices = _dataset_choices(controller)
            valid_ids = {cid for _, cid in choices}
            sess = controller.session_service.get_or_create_session()
            active = sess.get("active_dataset_id")
            if active in valid_ids:
                selected = active
            elif current_ds_id in valid_ids:
                selected = current_ds_id
            else:
                selected = choices[0][1] if choices else None
            report = controller.get_dataset(selected) if selected else None
            s_html, i_md = _format_summary(report)
            return gr.update(choices=choices, value=selected), s_html, i_md, _aug_iframe_html(selected)

        # ---- Wiring
        dataset_dropdown.change(
            fn=_on_ds_change,
            inputs=[dataset_dropdown],
            outputs=[summary_html, issues_md, aug_preview_html],
        )
        btn_refresh_ds.click(
            fn=_refresh_page,
            inputs=[dataset_dropdown],
            outputs=[dataset_dropdown, summary_html, issues_md, aug_preview_html],
        )
        btn_revalidate.click(fn=_handle_revalidate, inputs=[dataset_dropdown], outputs=[summary_html, issues_md])
        btn_rename_ds.click(
            fn=_handle_rename_ds,
            inputs=[dataset_dropdown, txt_rename_ds],
            outputs=[rename_ds_msg, dataset_dropdown, txt_rename_ds],
        )
        btn_preview_split.click(
            fn=_handle_preview_split,
            inputs=[dataset_dropdown, train_pct, val_pct, test_pct, chk_drop_dup],
            outputs=[split_md],
        )
        btn_split.click(
            fn=_handle_split,
            inputs=[dataset_dropdown, train_pct, val_pct, test_pct, seed_num, chk_drop_dup, new_split_name],
            outputs=[split_md, dataset_dropdown],
        )

    return {
        "container": container,
        "dataset_dropdown": dataset_dropdown,
        "btn_go_training": btn_go_training,
        "refresh_fn": _refresh_page,
        "refresh_inputs": [dataset_dropdown],
        "refresh_outputs": [dataset_dropdown, summary_html, issues_md, aug_preview_html],
    }
