"""Page 4: Training Monitor (live epoch metrics, charts, checkpoints, GPU/VRAM, logs, stop/reconnect)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from application.app_controller import AppController
from infrastructure.monitoring.system_monitor import get_disk_status
from ui.components.charts import build_training_charts
from ui.components.common import (
    render_kpi_cards,
    render_page_header,
    render_runtime_status_panel,
    render_status_badge,
)
from utils.validation import format_duration


def _run_choices(controller: AppController) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for run in controller.list_runs():
        label = f"{run.run_name} [{run.status}] — Ep {run.current_epoch}/{run.total_epochs}"
        choices.append((label, run.run_id))
    return choices


def render_monitor_outputs(controller: AppController, selected_run_id: str | None = None):
    snap = controller.get_monitor_snapshot(run_id=selected_run_id or None)
    job = snap["job"]
    gpu = snap["gpu"]
    series = snap["series"]
    disk = get_disk_status(controller.workspace.root)

    header_html = render_page_header(
        "Training Monitor",
        "ติดตามการเทรนแบบเรียลไทม์จาก Background Worker (Reconnect ได้เมื่อเบราว์เซอร์หลุด)",
        right_html=render_status_badge(job.status if job else None),
    )

    if job is None:
        kpi_html = render_kpi_cards(
            [
                ("Epoch", "0 / 0", "ยังไม่มีงานเทรน", False),
                ("เวลา (Elapsed)", "--:--:--", "ระยะเวลาที่ผ่านไป", False),
                ("คาดว่าจะเสร็จ (ETA)", "--:--:--", "ประมาณการ", False),
                ("Best mAP50-95", "-", "mAP50: -", True),
                ("Train / Val Loss", "-", "Precision / Recall: -", False),
            ]
        )
        progress_html = (
            '<div class="progress-track"><div class="progress-fill" style="width:0%;"></div></div>'
            "<div style='font-size:12px;color:#93a4cc;'>ความคืบหน้า: 0%</div>"
        )
        download_files: list[str] = []
    else:
        latest = job.latest_metrics or {}
        t_loss = latest.get("train_loss_total")
        v_loss = latest.get("val_loss_total")
        prec = latest.get("precision")
        rec = latest.get("recall")
        loss_val = f"{t_loss:.4f}" if isinstance(t_loss, (int, float)) else "-"
        loss_sub = (
            f"Val: {v_loss:.4f}" if isinstance(v_loss, (int, float)) else ""
        )
        if isinstance(prec, (int, float)) and isinstance(rec, (int, float)):
            loss_sub = f"{loss_sub} | P:{prec:.3f} R:{rec:.3f}".strip(" |")

        best_95 = f"{job.best_map50_95:.4f}" if job.best_map50_95 is not None else "-"
        best_50 = f"mAP50: {job.best_map50:.4f}" if job.best_map50 is not None else "รอผล Validation"

        kpi_html = render_kpi_cards(
            [
                ("Epoch", f"{job.current_epoch} / {job.total_epochs}", f"{job.progress_pct:.1f}%", False),
                ("เวลา (Elapsed)", format_duration(job.elapsed_seconds), f"PID: {job.pid or '-'}", False),
                ("คาดว่าจะเสร็จ (ETA)", format_duration(job.eta_seconds), "คำนวณจากเวลาเฉลี่ยต่อ Epoch", False),
                ("Best mAP50-95", best_95, best_50, True),
                ("Latest Loss", loss_val, loss_sub or "รอจบ Epoch แรก", False),
            ]
        )
        pct = max(0.0, min(100.0, float(job.progress_pct)))
        progress_html = (
            f'<div class="progress-track"><div class="progress-fill" style="width:{pct:.1f}%;"></div></div>'
            f"<div style='font-size:12px;color:#93a4cc;'>ความคืบหน้า: {pct:.1f}% ({job.current_epoch}/{job.total_epochs} epochs)</div>"
        )

        download_files = []
        for candidate in (job.best_checkpoint_path, job.last_checkpoint_path):
            if candidate and Path(candidate).exists():
                download_files.append(candidate)

    status_panel_html = render_runtime_status_panel(
        job=job,
        gpu=gpu,
        disk=disk,
        heartbeat_stale=snap["heartbeat_stale"],
    )
    loss_fig, map_fig, prec_fig, rec_fig = build_training_charts(series)
    log_text = snap["log_tail"] or "ยังไม่มีบันทึก Training Log"

    return (
        header_html,
        kpi_html,
        progress_html,
        loss_fig,
        map_fig,
        prec_fig,
        rec_fig,
        status_panel_html,
        download_files if download_files else None,
        log_text,
    )


def build_monitor_page(controller: AppController) -> dict[str, Any]:
    choices = _run_choices(controller)
    initial_run = choices[0][1] if choices else None
    init_vals = render_monitor_outputs(controller, initial_run)

    with gr.Column(visible=False) as container:
        header_html = gr.HTML(value=init_vals[0])

        with gr.Row():
            run_selector = gr.Dropdown(
                label="เลือก Run ที่ต้องการติดตาม",
                choices=choices,
                value=initial_run,
                scale=4,
                interactive=True,
            )
            btn_reconnect = gr.Button("🔄 Reconnect / รีเฟรช", variant="secondary", scale=2)
            btn_pause_planned = gr.Button("⏸ หยุดชั่วคราว (Phase 4)", interactive=False, scale=1)
            btn_stop = gr.Button("🛑 หยุดเทรน (Graceful Stop)", elem_classes=["danger-btn"], scale=2)

        action_feedback = gr.Markdown(value="")
        kpi_html = gr.HTML(value=init_vals[1])
        progress_html = gr.HTML(value=init_vals[2])

        with gr.Row():
            with gr.Column(scale=7):
                with gr.Tabs():
                    with gr.Tab("Loss"):
                        plot_loss = gr.Plot(value=init_vals[3], label="Train / Val Loss")
                    with gr.Tab("mAP"):
                        plot_map = gr.Plot(value=init_vals[4], label="mAP50 & mAP50-95")
                    with gr.Tab("Precision"):
                        plot_prec = gr.Plot(value=init_vals[5], label="Precision")
                    with gr.Tab("Recall"):
                        plot_rec = gr.Plot(value=init_vals[6], label="Recall")

            with gr.Column(scale=5):
                status_panel_html = gr.HTML(value=init_vals[7])
                checkpoint_downloads = gr.File(
                    label="ดาวน์โหลด Checkpoints ล่าสุด (best.pt / last.pt)",
                    value=init_vals[8],
                    interactive=False,
                )

        with gr.Accordion("📜 Training Log (เรียลไทม์)", open=True):
            log_box = gr.Textbox(
                label="Log ล่าสุดจาก Training Worker",
                value=init_vals[9],
                lines=12,
                max_lines=18,
                interactive=False,
            )

        def _refresh_monitor(selected_run: str | None):
            new_choices = _run_choices(controller)
            ids = {rid for _, rid in new_choices}
            active = controller.training_service.get_current_job()
            if selected_run in ids:
                target = selected_run
            elif active and active.run_id in ids:
                target = active.run_id
            else:
                target = new_choices[0][1] if new_choices else None
            outputs = render_monitor_outputs(controller, target)
            return (gr.update(choices=new_choices, value=target), *outputs)

        def _handle_stop(selected_run: str | None):
            job = controller.stop_training(run_id=selected_run or None)
            if job is None:
                msg = "ℹ️ ไม่มีงานเทรนที่กำลังทำงานอยู่"
            else:
                msg = (
                    f"🛑 ส่งสัญญาณ **หยุดเทรนอย่างปลอดภัย (Graceful Stop)** ไปยัง `{job.run_name}` แล้ว "
                    "(Worker จะบันทึก `last.pt` และหยุดในจังหวะปลอดภัยถัดไป)"
                )
            refreshed = _refresh_monitor(selected_run)
            return (msg, *refreshed)

        def _handle_reconnect(selected_run: str | None):
            snap = controller.get_monitor_snapshot(run_id=selected_run or None)
            job = snap["job"]
            if job is None:
                msg = "ℹ️ ยังไม่มีงานเทรนให้เชื่อมต่อใน Workspace นี้"
            elif snap["alive"]:
                msg = f"🔄 เชื่อมต่อกลับไปยังงานที่กำลังรันอยู่แล้ว: `{job.run_name}` (PID: `{job.pid}`, Epoch {job.current_epoch}/{job.total_epochs}) โดยไม่เริ่มเทรนซ้ำ"
            else:
                msg = f"ℹ️ โหลดสถานะล่าสุดของ `{job.run_name}` เรียบร้อยแล้ว (สถานะ: **{job.status}**)"
            refreshed = _refresh_monitor(selected_run)
            return (msg, *refreshed)

        monitor_outputs = [
            run_selector,
            header_html,
            kpi_html,
            progress_html,
            plot_loss,
            plot_map,
            plot_prec,
            plot_rec,
            status_panel_html,
            checkpoint_downloads,
            log_box,
        ]

        run_selector.change(fn=_refresh_monitor, inputs=[run_selector], outputs=monitor_outputs)
        btn_reconnect.click(fn=_handle_reconnect, inputs=[run_selector], outputs=[action_feedback, *monitor_outputs])
        btn_stop.click(fn=_handle_stop, inputs=[run_selector], outputs=[action_feedback, *monitor_outputs])

    return {
        "container": container,
        "run_selector": run_selector,
        "refresh_fn": _refresh_monitor,
        "refresh_inputs": [run_selector],
        "refresh_outputs": monitor_outputs,
    }
