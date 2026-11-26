"""Unified Training Workspace (Brief v2 §3.4, §3.5, §6 & §7):
- Top: Dataset Selector / Run Selector / Reconnect
- Left (~58%): Training Setup (with Conditional Fields by Training Mode) + Training Monitor & 4 Charts
- Right (~42%): Model Testing (Manual Only: Update Best Model -> Upload Images -> Previous/Next -> Detect)
- Bottom: Checkpoint & Export (.pt only, safe snapshot copy while Training runs)
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import gradio as gr

from application.app_controller import AppController
from config.defaults import (
    ALLOWED_IMG_SIZES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_EPOCHS,
    DEFAULT_IMG_SIZE,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MODEL_VARIANT_LABEL,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_WORKERS,
    YOLO11_MODEL_VARIANTS,
)
from core.schemas import JobState
from infrastructure.monitoring.system_monitor import get_available_devices
from ui.components.charts import build_training_charts
from ui.components.common import (
    render_page_header,
    render_status_badge,
    render_temp_storage_warning,
)
from utils.validation import format_bytes, format_duration

TRAINING_MODES = [
    ("Train New", "train"),
    ("Fine-tune", "fine_tune"),
    ("Resume", "resume"),
]
_STOPPABLE_STATUSES = {"queued", "starting", "running"}


def _stop_enabled(job: JobState | None) -> bool:
    """Stop is available only while a job is actually running."""
    return job is not None and job.status in _STOPPABLE_STATUSES


def reconnect_needed(snap: dict[str, Any]) -> bool:
    """Show Reconnect only when a live job looks disconnected or its heartbeat is stale."""
    job = snap.get("job")
    if job is None:
        return False
    if snap.get("heartbeat_stale"):
        return True
    return job.status in {"queued", "starting", "running", "stopping"} and not snap.get("alive")


def _dataset_choices(controller: AppController) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for ds in controller.list_datasets():
        if ds.total_images <= 0:
            continue
        status = "Ready" if ds.is_valid else "Needs Review"
        choices.append((f"{ds.dataset_name} ({ds.total_images} images, {len(ds.class_names)} classes — {status})", ds.dataset_id))
    return choices


def _dataset_readiness_notice(controller: AppController, dataset_id: str | None) -> str:
    if not dataset_id:
        return "No dataset selected. Import or select a dataset before starting training."
    ds = controller.get_dataset(dataset_id)
    if ds is None:
        return "Selected dataset was not found."
    if not ds.is_valid:
        val_split = ds.splits.get("val")
        if val_split is None or val_split.image_count == 0:
            return "Dataset is missing a Validation split. Go to Prepare to generate and apply a Train/Val split before training."
        return "Dataset has validation errors. Open Prepare or Dataset > Manage & Diagnostics to resolve them."
    return ""


def _run_choices(controller: AppController) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for j in controller.list_runs():
        ep = f"{j.current_epoch}/{j.total_epochs}"
        out.append((f"[{j.status.upper()}] {j.run_name} (Epoch {ep})", j.run_id))
    return out


def _resumable_checkpoint_choices(controller: AppController) -> list[tuple[str, str]]:
    """Return existing last.pt checkpoints across runs for Resume mode."""
    out: list[tuple[str, str]] = []
    for j in controller.list_runs():
        if j.last_checkpoint_path and Path(j.last_checkpoint_path).is_file():
            sz = format_bytes(Path(j.last_checkpoint_path).stat().st_size)
            label = f"{j.run_name} — last.pt (Epoch {j.current_epoch}/{j.total_epochs}, {sz})"
            out.append((label, j.last_checkpoint_path))
    return out


def _render_monitor_html(snap: dict[str, Any]) -> str:
    job: JobState | None = snap.get("job")
    gpu: dict[str, Any] = snap.get("gpu") or {}
    gpu_txt = f"{gpu.get('name', 'CPU')}"
    if gpu.get("vram_total_mb"):
        gpu_txt += f" · VRAM {gpu.get('vram_used_mb', 0):.0f}/{gpu['vram_total_mb']:.0f} MB"

    if job is None:
        return (
            '<div class="studio-card">'
            '<div style="display:flex;justify-content:space-between;align-items:center;">'
            "<strong>Idle</strong>"
            f'<span style="font-size:12px;color:#64748B;">{html.escape(gpu_txt)}</span>'
            "</div>"
            '<div class="empty-note" style="margin-top:6px;">Select a dataset and click Start Training.</div>'
            "</div>"
        )

    latest = job.latest_metrics or {}
    pct = max(0.0, min(100.0, float(job.progress_pct or 0.0)))
    fmt = lambda v: f"{float(v):.4f}" if v is not None else "—"
    best_ep = f"Epoch {job.best_epoch}" if job.best_epoch else "—"

    err_html = ""
    if job.error_message:
        err_html = (
            '<div style="margin-top:8px;padding:8px 10px;border-radius:8px;background:rgba(239,68,68,0.12);'
            f'border:1px solid rgba(239,68,68,0.4);color:#fca5a5;font-size:12px;">Error: {html.escape(job.error_message)}</div>'
        )

    return f"""
    <div class="studio-card">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
        <div>
          <strong style="font-size:14px;">{html.escape(job.run_name)}</strong>
          <span style="color:#64748B;font-size:12px;margin-left:8px;">{html.escape(job.model_variant.upper())} · {html.escape(gpu_txt)}</span>
        </div>
        {render_status_badge(job.status)}
      </div>

      <div style="margin-top:10px;display:flex;justify-content:space-between;font-size:12px;color:#64748B;">
        <span>Epoch <strong>{job.current_epoch} / {job.total_epochs}</strong> ({pct:.1f}%)</span>
        <span>Elapsed: <strong>{format_duration(job.elapsed_seconds)}</strong> · ETA: <strong>{format_duration(job.eta_seconds)}</strong></span>
      </div>
      <div style="height:8px;background:#E2E8F0;border-radius:99px;overflow:hidden;margin:6px 0 10px 0;">
        <div style="height:100%;width:{pct:.1f}%;background:#2563EB;"></div>
      </div>

      <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:6px;text-align:center;">
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:6px;">
          <div style="font-size:11px;color:#64748B;">Train Loss</div>
          <div style="font-weight:650;">{fmt(latest.get('train_loss_total'))}</div>
        </div>
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:6px;">
          <div style="font-size:11px;color:#64748B;">Val Loss</div>
          <div style="font-weight:650;">{fmt(latest.get('val_loss_total'))}</div>
        </div>
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:6px;">
          <div style="font-size:11px;color:#64748B;">Precision</div>
          <div style="font-weight:650;">{fmt(latest.get('precision'))}</div>
        </div>
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:6px;">
          <div style="font-size:11px;color:#64748B;">Recall</div>
          <div style="font-weight:650;">{fmt(latest.get('recall'))}</div>
        </div>
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:6px;">
          <div style="font-size:11px;color:#64748B;">mAP50</div>
          <div style="font-weight:650;">{fmt(job.best_map50)}</div>
        </div>
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:6px;">
          <div style="font-size:11px;color:#64748B;">mAP50-95</div>
          <div style="font-weight:650;">{fmt(job.best_map50_95)}</div>
          <div style="font-size:10px;color:#64748B;">{best_ep}</div>
        </div>
      </div>
      {err_html}
    </div>
    """


def _render_checkpoint_status(job: JobState | None) -> str:
    if job is None:
        return "No checkpoints available for the selected run."
    parts = [f"**Run:** `{job.run_name}`"]
    best_p = Path(job.best_checkpoint_path) if job.best_checkpoint_path else None
    last_p = Path(job.last_checkpoint_path) if job.last_checkpoint_path else None
    if best_p and best_p.is_file():
        ep = f"Epoch {job.best_epoch}" if job.best_epoch else "—"
        mp = f"mAP50-95 {job.best_map50_95:.4f}" if job.best_map50_95 is not None else ""
        parts.append(f"**best.pt:** Ready ({format_bytes(best_p.stat().st_size)}, {ep} {mp})")
    else:
        parts.append("**best.pt:** Pending")
    if last_p and last_p.is_file():
        parts.append(f"**last.pt:** Ready ({format_bytes(last_p.stat().st_size)})")
    else:
        parts.append("**last.pt:** Pending")
    return " · ".join(parts)


def build_training_workspace_page(controller: AppController) -> dict[str, Any]:
    ds_choices = _dataset_choices(controller)
    init_ds = ds_choices[0][1] if ds_choices else None
    run_choices = _run_choices(controller)
    resume_choices = _resumable_checkpoint_choices(controller)
    devices = get_available_devices()
    init_snap = controller.get_monitor_snapshot(run_id=None)
    init_job: JobState | None = init_snap.get("job")
    init_run_id = init_job.run_id if init_job else (run_choices[0][1] if run_choices else None)
    init_ds_notice = _dataset_readiness_notice(controller, init_ds)

    with gr.Column(visible=False) as container:
        gr.HTML(
            render_page_header(
                "Training",
                "Set up a run, watch metrics, test a checkpoint manually, and export .pt weights.",
            )
        )

        # ---------------- TOP BAR: Dataset / Run Selector / Reconnect
        with gr.Row(elem_classes=["compact-row"]):
            dataset_dd = gr.Dropdown(
                label="Dataset",
                choices=ds_choices,
                value=init_ds,
                scale=4,
            )
            run_selector = gr.Dropdown(
                label="Run",
                choices=run_choices,
                value=init_run_id,
                scale=4,
            )
            btn_reconnect = gr.Button(
                "Reconnect",
                variant="secondary",
                size="sm",
                scale=1,
                visible=reconnect_needed(init_snap),
            )

        # ---------------- MAIN SPLIT LAYOUT (Left ~58% Training | Right ~42% Model Testing)
        with gr.Row(equal_height=False):
            # ==================== LEFT: TRAINING SETUP + TRAINING MONITOR ====================
            with gr.Column(scale=6, min_width=460):
                gr.Markdown("### Training Setup")

                ds_notice_md = gr.Markdown(
                    value=(f"**Notice:** {init_ds_notice}" if init_ds_notice else ""),
                    visible=bool(init_ds_notice),
                )

                mode_rd = gr.Radio(
                    label="Training Mode",
                    choices=TRAINING_MODES,
                    value="train",
                )

                # --- Mode 1: Train New (visible by default)
                with gr.Column(visible=True) as train_new_group:
                    with gr.Row():
                        train_model_dd = gr.Dropdown(
                            label="Model",
                            choices=list(YOLO11_MODEL_VARIANTS.keys()),
                            value=DEFAULT_MODEL_VARIANT_LABEL,
                            scale=5,
                        )
                        train_weights_tb = gr.Textbox(
                            label="Pretrained Weight",
                            value=YOLO11_MODEL_VARIANTS[DEFAULT_MODEL_VARIANT_LABEL]["pretrained_weights"],
                            scale=5,
                        )

                # --- Mode 2: Fine-tune (hidden until selected; preserves values)
                with gr.Column(visible=False) as finetune_group:
                    gr.Markdown("_Starts a new training run initialized from your selected weights._")
                    with gr.Row():
                        ft_model_dd = gr.Dropdown(
                            label="Model",
                            choices=list(YOLO11_MODEL_VARIANTS.keys()),
                            value=DEFAULT_MODEL_VARIANT_LABEL,
                            scale=4,
                        )
                        ft_weights_tb = gr.Textbox(
                            label="Pretrained Weight",
                            placeholder="Path to .pt weight or upload below",
                            scale=6,
                        )
                    ft_weight_file = gr.File(
                        label="Upload Weight (.pt)",
                        file_types=[".pt"],
                        type="filepath",
                    )

                # --- Mode 3: Resume (hidden until selected; hides irrelevant Model/Pretrained Weight)
                with gr.Column(visible=False) as resume_group:
                    gr.Markdown("_Resume continues training from an existing run's `last.pt` checkpoint._")
                    resume_ckpt_dd = gr.Dropdown(
                        label="Checkpoint",
                        choices=resume_choices,
                        value=(resume_choices[0][1] if resume_choices else None),
                    )
                    resume_upload_file = gr.File(
                        label="Or Upload Checkpoint (last.pt)",
                        file_types=[".pt"],
                        type="filepath",
                    )
                    resume_empty_md = gr.Markdown(
                        value="**Notice:** No resumable `last.pt` checkpoint found. Complete at least 1 epoch or upload a `last.pt` file.",
                        visible=(len(resume_choices) == 0),
                    )

                # --- Core Parameters (Batch Size & Image Size hidden in Resume mode)
                with gr.Row():
                    epochs_num = gr.Number(label="Epochs", value=DEFAULT_EPOCHS, minimum=1, maximum=1000, precision=0)
                    batch_num = gr.Number(label="Batch Size", value=DEFAULT_BATCH_SIZE, minimum=1, maximum=256, precision=0, visible=True)
                    imgsz_dd = gr.Dropdown(label="Image Size", choices=ALLOWED_IMG_SIZES, value=DEFAULT_IMG_SIZE, visible=True)
                    device_dd = gr.Dropdown(label="Device", choices=devices, value=devices[0][1])

                with gr.Accordion("Advanced Parameters", open=False):
                    with gr.Row():
                        run_name_tb = gr.Textbox(label="Run Name (optional)", placeholder="e.g. yolo11s_v1")
                        workers_num = gr.Number(label="Workers", value=DEFAULT_WORKERS, minimum=0, maximum=16, precision=0)
                        seed_num = gr.Number(label="Seed", value=DEFAULT_SEED, precision=0)
                    with gr.Row():
                        optimizer_dd = gr.Dropdown(
                            label="Optimizer",
                            choices=["auto", "SGD", "Adam", "AdamW"],
                            value="auto",
                        )
                        lr0_num = gr.Number(label="Learning Rate", value=0.0, minimum=0.0, maximum=1.0, info="0 uses framework default")
                        patience_num = gr.Number(label="Patience", value=DEFAULT_PATIENCE, minimum=0, precision=0)
                    with gr.Row():
                        chk_save_best = gr.Checkbox(label="Save Best Checkpoint (mAP50-95)", value=True)
                        chk_save_last = gr.Checkbox(label="Save Last Checkpoint every epoch", value=True)

                with gr.Row(elem_classes=["compact-row"]):
                    btn_start = gr.Button("Start Training", variant="primary", scale=3)
                    btn_stop = gr.Button(
                        "Stop Training",
                        variant="stop",
                        scale=2,
                        interactive=_stop_enabled(init_job),
                    )
                train_feedback = gr.Markdown("")

                # --- Training Monitor & 4 Charts
                gr.Markdown("### Training Monitor")
                init_loss_fig, init_map_fig, init_prec_fig, init_rec_fig = build_training_charts(init_snap.get("series") or {})
                monitor_html = gr.HTML(value=_render_monitor_html(init_snap))
                with gr.Row():
                    loss_plot = gr.Plot(value=init_loss_fig, label="Loss")
                    map_plot = gr.Plot(value=init_map_fig, label="mAP")
                with gr.Row():
                    prec_plot = gr.Plot(value=init_prec_fig, label="Precision")
                    rec_plot = gr.Plot(value=init_rec_fig, label="Recall")

                with gr.Accordion("Training Log & Storage Notice", open=False):
                    gr.HTML(render_temp_storage_warning())
                    log_box = gr.Textbox(
                        label="Worker Log",
                        value=init_snap.get("log_tail", ""),
                        lines=10,
                        interactive=False,
                    )

            # ==================== RIGHT: MODEL TESTING (MANUAL ONLY) ====================
            with gr.Column(scale=5, min_width=380):
                gr.Markdown("### Model Testing")

                test_snapshot_state = gr.State(None)
                test_images_state = gr.State([])
                test_idx_state = gr.State(0)
                test_result_state = gr.State(None)

                with gr.Row(elem_classes=["compact-row"]):
                    btn_update_best = gr.Button("Update Best Model", variant="primary", scale=3)
                snapshot_info_md = gr.Markdown(
                    "_No test model loaded. Click **Update Best Model** to load a snapshot of the latest `best.pt`._"
                )

                test_upload = gr.File(
                    label="Upload Images",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                )

                test_image_view = gr.Image(
                    label="Test Image",
                    type="filepath",
                    height=340,
                    interactive=False,
                )

                with gr.Row(elem_classes=["compact-row"]):
                    btn_test_prev = gr.Button("Previous", size="sm", scale=2)
                    test_counter_md = gr.Markdown("**0 / 0**")
                    btn_test_next = gr.Button("Next", size="sm", scale=2)
                    view_mode_rd = gr.Radio(
                        choices=[("Detection", "annotated"), ("Original", "original")],
                        value="annotated",
                        show_label=False,
                        scale=4,
                    )

                test_conf = gr.Slider(label="Confidence", minimum=0.05, maximum=0.95, step=0.05, value=DEFAULT_CONF_THRESHOLD)
                test_iou = gr.Slider(label="IoU Threshold", minimum=0.05, maximum=0.95, step=0.05, value=DEFAULT_IOU_THRESHOLD)

                btn_detect = gr.Button("Detect", variant="primary")
                test_msg = gr.Markdown("")

                det_table = gr.Dataframe(
                    headers=["#", "Class", "Confidence", "x1, y1, x2, y2"],
                    datatype=["number", "str", "str", "str"],
                    type="array",
                    column_count=4,
                    row_count=0,
                    interactive=False,
                    label="Detection Results",
                )

        # ---------------- BOTTOM: CHECKPOINT & EXPORT (.pt ONLY)
        gr.Markdown("---")
        gr.Markdown("### Checkpoint & Export")
        ckpt_status_md = gr.Markdown(_render_checkpoint_status(init_job))
        with gr.Row(elem_classes=["compact-row"]):
            btn_dl_best = gr.Button("Download Best.pt", variant="primary", scale=2)
            btn_dl_last = gr.Button("Download Last.pt", variant="secondary", scale=2)
            export_which_dd = gr.Dropdown(
                choices=[("best.pt", "best"), ("last.pt", "last")],
                value="best",
                show_label=False,
                scale=2,
            )
            btn_export_bundle = gr.Button("Export Selected Checkpoint", variant="secondary", scale=3)
        export_file = gr.File(label="Snapshot Download", interactive=False)
        export_msg = gr.Markdown("")

        # ================================================================ HANDLERS
        def _on_dataset_change(ds_id: str | None):
            notice = _dataset_readiness_notice(controller, ds_id)
            return gr.update(value=(f"**Notice:** {notice}" if notice else ""), visible=bool(notice))

        def _on_train_model_change(label: str):
            spec = YOLO11_MODEL_VARIANTS.get(label)
            return spec["pretrained_weights"] if spec else "yolo11s.pt"

        def _on_ft_upload(fpath: str | None):
            return fpath if fpath else gr.update()

        def _on_mode_change(mode: str, resume_ckpt: str | None, resume_upload: str | None):
            is_train = mode == "train"
            is_ft = mode == "fine_tune"
            is_resume = mode == "resume"

            res_choices = _resumable_checkpoint_choices(controller)
            valid_ckpt = resume_ckpt if any(c[1] == resume_ckpt for c in res_choices) else (res_choices[0][1] if res_choices else None)
            has_resume_ckpt = bool(valid_ckpt or resume_upload)
            can_start = (not is_resume) or has_resume_ckpt

            return (
                gr.update(visible=is_train),
                gr.update(visible=is_ft),
                gr.update(visible=is_resume),
                gr.update(choices=res_choices, value=valid_ckpt),
                gr.update(visible=(is_resume and not has_resume_ckpt)),
                gr.update(visible=(not is_resume)),  # Batch Size
                gr.update(visible=(not is_resume)),  # Image Size
                gr.update(interactive=can_start),    # Start Training button
            )

        def _on_resume_ckpt_change(mode: str, resume_ckpt: str | None, resume_upload: str | None):
            if mode != "resume":
                return gr.update(), gr.update()
            has_ckpt = bool(resume_ckpt or resume_upload)
            return gr.update(visible=(not has_ckpt)), gr.update(interactive=has_ckpt)

        def _handle_start(
            ds_id, mode,
            train_model_label, train_weights,
            ft_model_label, ft_weights, ft_uploaded,
            resume_ckpt, resume_uploaded,
            epochs, batch, imgsz, device, run_name, workers, seed,
            optimizer, lr0, patience, save_best, save_last,
        ):
            no_mon = (gr.update(),) * 9
            if not ds_id:
                return ("Error: Select a dataset before starting training.", *no_mon)

            if mode == "train":
                spec = YOLO11_MODEL_VARIANTS.get(train_model_label)
                if not spec:
                    return ("Error: Select a valid YOLO11 model.", *no_mon)
                model_id = spec["model_id"]
                eff_weights = (train_weights or "").strip() or spec["pretrained_weights"]
            elif mode == "fine_tune":
                spec = YOLO11_MODEL_VARIANTS.get(ft_model_label)
                if not spec:
                    return ("Error: Select a valid YOLO11 model.", *no_mon)
                model_id = spec["model_id"]
                eff_weights = ft_uploaded if ft_uploaded else (ft_weights or "").strip()
                if not eff_weights:
                    return ("Error: Specify or upload a .pt weight for Fine-tune mode.", *no_mon)
            else:  # resume
                eff_weights = resume_uploaded if resume_uploaded else (resume_ckpt or "").strip()
                if not eff_weights or not Path(eff_weights).is_file():
                    return ("Error: Select or upload a valid last.pt checkpoint to resume.", *no_mon)
                model_id = "yolo11s"

            try:
                job = controller.start_training(
                    dataset_id=ds_id,
                    model_variant=model_id,
                    pretrained_weights=eff_weights,
                    epochs=int(epochs),
                    batch_size=int(batch),
                    imgsz=int(imgsz),
                    device=device,
                    seed=int(seed),
                    patience=int(patience),
                    workers=int(workers),
                    run_name=run_name or None,
                    mode=mode,
                    optimizer=optimizer,
                    lr0=(float(lr0) if lr0 and float(lr0) > 0 else None),
                    save_best=bool(save_best),
                    save_last=bool(save_last),
                )
                snap = controller.get_monitor_snapshot(run_id=job.run_id)
                l_fig, m_fig, p_fig, r_fig = build_training_charts(snap.get("series") or {})
                return (
                    f"Training started: **`{job.run_name}`** (PID {job.pid}).",
                    gr.update(choices=_run_choices(controller), value=job.run_id),
                    _render_monitor_html(snap),
                    l_fig,
                    m_fig,
                    p_fig,
                    r_fig,
                    snap.get("log_tail", ""),
                    _render_checkpoint_status(snap.get("job")),
                    gr.update(interactive=True),
                )
            except Exception as exc:
                return (f"Error: {exc}", *no_mon)

        def _handle_stop(selected_run: str | None):
            try:
                job = controller.stop_training(run_id=selected_run or None)
                if job is None:
                    return "No active training job to stop.", gr.update(interactive=False)
                return f"Graceful stop requested for **`{job.run_name}`**.", gr.update(interactive=False)
            except Exception as exc:
                return f"Error stopping training: {exc}", gr.update()

        def _refresh_monitor(selected_run: str | None):
            snap = controller.get_monitor_snapshot(run_id=selected_run or None)
            job = snap.get("job")
            runs = _run_choices(controller)
            run_val = selected_run if any(r[1] == selected_run for r in runs) else (job.run_id if job else None)
            l_fig, m_fig, p_fig, r_fig = build_training_charts(snap.get("series") or {})
            return (
                gr.update(choices=runs, value=run_val),
                _render_monitor_html(snap),
                l_fig,
                m_fig,
                p_fig,
                r_fig,
                snap.get("log_tail", ""),
                _render_checkpoint_status(job),
                gr.update(interactive=_stop_enabled(job)),
            )

        def _refresh_full(selected_ds: str | None, selected_run: str | None):
            ds_list = _dataset_choices(controller)
            ds_val = selected_ds if any(d[1] == selected_ds for d in ds_list) else (ds_list[0][1] if ds_list else None)
            notice = _dataset_readiness_notice(controller, ds_val)
            mon = _refresh_monitor(selected_run)
            snap = controller.get_monitor_snapshot(run_id=selected_run or None)
            return (
                gr.update(choices=ds_list, value=ds_val),
                gr.update(value=(f"**Notice:** {notice}" if notice else ""), visible=bool(notice)),
                *mon,
                gr.update(visible=reconnect_needed(snap)),
            )

        # ---- Manual Testing handlers (NEVER auto-update, NEVER auto-detect)
        def _handle_update_best(selected_run: str | None):
            try:
                info = controller.create_testing_snapshot(run_id=selected_run or None)
                mp = f"{info['best_map50_95']:.4f}" if info["best_map50_95"] is not None else "—"
                md = (
                    f"**Loaded Snapshot:** `{info['run_name']}` · "
                    f"Epoch **{info['best_epoch']}** · mAP50-95 **{mp}** · "
                    f"Updated **{info['updated_at']}** ({format_bytes(info['size_bytes'])})"
                )
                return info, md
            except Exception as exc:
                return gr.update(), f"Error: {exc}"

        def _handle_upload_test_images(files: list[str] | None):
            paths = [str(f) for f in (files or []) if f]
            if not paths:
                return [], 0, None, None, "**0 / 0**", [], "No test images loaded."
            return paths, 0, None, paths[0], f"**1 / {len(paths)}**", [], f"Loaded {len(paths)} image(s). Click Detect to run inference."

        def _handle_test_nav(paths: list[str], cur_idx: int, delta: int):
            if not paths:
                return 0, None, None, "**0 / 0**", [], "Upload test images first."
            new_idx = (cur_idx + delta) % len(paths)
            return (
                new_idx,
                None,
                paths[new_idx],
                f"**{new_idx + 1} / {len(paths)}**",
                [],
                f"Image {new_idx + 1}/{len(paths)}: `{Path(paths[new_idx]).name}` (click Detect to run inference)",
            )

        def _handle_detect(snapshot: dict[str, Any] | None, paths: list[str], cur_idx: int, conf: float, iou: float, view_mode: str):
            if not snapshot or not snapshot.get("snapshot_path"):
                return None, gr.update(), [], "No test model loaded. Click Update Best Model first."
            if not paths:
                return None, gr.update(), [], "Upload test images first."
            idx = max(0, min(cur_idx, len(paths) - 1))
            img_path = paths[idx]
            try:
                res = controller.run_inference(
                    weights_path=snapshot["snapshot_path"],
                    image_path=img_path,
                    conf=float(conf),
                    iou=float(iou),
                )
                rows = [
                    [
                        i + 1,
                        d.class_name,
                        f"{d.confidence:.3f}",
                        f"{int(round(d.x1))}, {int(round(d.y1))}, {int(round(d.x2))}, {int(round(d.y2))}",
                    ]
                    for i, d in enumerate(res.detections)
                ]
                saved_res = {"index": idx, "annotated": res.annotated_image_path, "original": img_path, "rows": rows}
                shown = res.annotated_image_path if view_mode == "annotated" else img_path
                msg = f"Detected **{len(rows)} object(s)** ({res.inference_ms:.1f} ms) using Snapshot Epoch {snapshot.get('best_epoch', '—')}."
                return saved_res, shown, rows, msg
            except Exception as exc:
                return None, gr.update(), [], f"Detection error: {exc}"

        def _handle_view_mode(view_mode: str, paths: list[str], cur_idx: int, last_res: dict[str, Any] | None):
            if not paths:
                return None
            idx = max(0, min(cur_idx, len(paths) - 1))
            if view_mode == "annotated" and last_res and last_res.get("index") == idx:
                return last_res.get("annotated")
            return paths[idx]

        # ---- Export handlers (.pt only, safe snapshot while training)
        def _handle_export(selected_run: str | None, which: str, bundle: bool):
            try:
                info = controller.export_pt_checkpoint(run_id=selected_run or None, which=which, include_bundle=bundle)
                msg = (
                    f"Snapshot ready: **`{info['filename']}`** "
                    f"({format_bytes(info['size_bytes'])}) · {info['exported_at']}"
                )
                return info["file_path"], msg
            except Exception as exc:
                return None, f"Export error: {exc}"

        # ---- wiring
        dataset_dd.change(fn=_on_dataset_change, inputs=[dataset_dd], outputs=[ds_notice_md])
        train_model_dd.change(fn=_on_train_model_change, inputs=[train_model_dd], outputs=[train_weights_tb])
        ft_weight_file.change(fn=_on_ft_upload, inputs=[ft_weight_file], outputs=[ft_weights_tb])

        mode_rd.change(
            fn=_on_mode_change,
            inputs=[mode_rd, resume_ckpt_dd, resume_upload_file],
            outputs=[
                train_new_group,
                finetune_group,
                resume_group,
                resume_ckpt_dd,
                resume_empty_md,
                batch_num,
                imgsz_dd,
                btn_start,
            ],
        )
        resume_ckpt_dd.change(
            fn=_on_resume_ckpt_change,
            inputs=[mode_rd, resume_ckpt_dd, resume_upload_file],
            outputs=[resume_empty_md, btn_start],
        )
        resume_upload_file.change(
            fn=_on_resume_ckpt_change,
            inputs=[mode_rd, resume_ckpt_dd, resume_upload_file],
            outputs=[resume_empty_md, btn_start],
        )

        mon_outputs = [run_selector, monitor_html, loss_plot, map_plot, prec_plot, rec_plot, log_box, ckpt_status_md, btn_stop]
        btn_start.click(
            fn=_handle_start,
            inputs=[
                dataset_dd, mode_rd,
                train_model_dd, train_weights_tb,
                ft_model_dd, ft_weights_tb, ft_weight_file,
                resume_ckpt_dd, resume_upload_file,
                epochs_num, batch_num, imgsz_dd, device_dd, run_name_tb, workers_num, seed_num,
                optimizer_dd, lr0_num, patience_num, chk_save_best, chk_save_last,
            ],
            outputs=[train_feedback, *mon_outputs],
        )
        btn_stop.click(fn=_handle_stop, inputs=[run_selector], outputs=[train_feedback, btn_stop])
        btn_reconnect.click(fn=_refresh_monitor, inputs=[run_selector], outputs=mon_outputs)
        run_selector.change(fn=_refresh_monitor, inputs=[run_selector], outputs=mon_outputs)

        btn_update_best.click(fn=_handle_update_best, inputs=[run_selector], outputs=[test_snapshot_state, snapshot_info_md])
        test_upload.change(
            fn=_handle_upload_test_images,
            inputs=[test_upload],
            outputs=[test_images_state, test_idx_state, test_result_state, test_image_view, test_counter_md, det_table, test_msg],
        )
        btn_test_prev.click(
            fn=lambda p, i: _handle_test_nav(p, i, -1),
            inputs=[test_images_state, test_idx_state],
            outputs=[test_idx_state, test_result_state, test_image_view, test_counter_md, det_table, test_msg],
        )
        btn_test_next.click(
            fn=lambda p, i: _handle_test_nav(p, i, 1),
            inputs=[test_images_state, test_idx_state],
            outputs=[test_idx_state, test_result_state, test_image_view, test_counter_md, det_table, test_msg],
        )
        btn_detect.click(
            fn=_handle_detect,
            inputs=[test_snapshot_state, test_images_state, test_idx_state, test_conf, test_iou, view_mode_rd],
            outputs=[test_result_state, test_image_view, det_table, test_msg],
        )
        view_mode_rd.change(
            fn=_handle_view_mode,
            inputs=[view_mode_rd, test_images_state, test_idx_state, test_result_state],
            outputs=[test_image_view],
        )

        btn_dl_best.click(fn=lambda r: _handle_export(r, "best", False), inputs=[run_selector], outputs=[export_file, export_msg])
        btn_dl_last.click(fn=lambda r: _handle_export(r, "last", False), inputs=[run_selector], outputs=[export_file, export_msg])
        btn_export_bundle.click(
            fn=lambda r, w: _handle_export(r, w, True),
            inputs=[run_selector, export_which_dd],
            outputs=[export_file, export_msg],
        )

    return {
        "container": container,
        "run_selector": run_selector,
        "monitor_refresh_fn": _refresh_monitor,
        "btn_reconnect": btn_reconnect,
        "monitor_refresh_outputs": mon_outputs,
        "refresh_fn": _refresh_full,
        "refresh_inputs": [dataset_dd, run_selector],
        "refresh_outputs": [dataset_dd, ds_notice_md, *mon_outputs, btn_reconnect],
    }
