"""Page 1: Home / Dashboard (overview, quick start, runtime summary, concise help)."""

from __future__ import annotations

import html
from typing import Any

import gradio as gr

from application.app_controller import AppController
from config.defaults import APP_NAME, APP_PHASE, APP_VERSION
from ui.components.common import (
    render_kpi_cards,
    render_status_badge,
    render_temp_storage_warning,
)


def _render_overview_html(controller: AppController) -> str:
    ov = controller.get_system_overview()
    versions = ov["versions"]
    gpu = ov["gpu"]
    disk = ov["disk"]
    session = ov["session"]
    active_job = ov["active_job"]

    job_label = "ยังไม่มีงานเทรน"
    job_sub = "ไปที่หน้า Dataset → Training"
    if active_job is not None:
        job_label = f"Ep {active_job.current_epoch}/{active_job.total_epochs}"
        job_sub = f"{active_job.run_name} ({active_job.status})"

    kpi_html = render_kpi_cards(
        [
            ("Session ID", str(session.get("session_id", "-")), "Temporary Workspace", False),
            ("Dataset พร้อมใช้", str(ov["dataset_count"]), "อัปโหลดในเซสชันนี้", False),
            ("งานเทรน (Runs)", str(ov["run_count"]), f"Checkpoints: {ov['checkpoint_count']}", False),
            ("งานล่าสุด", job_label, job_sub, active_job is not None),
        ]
    )

    env_html = (
        '<div class="studio-card" style="margin-top:10px;">'
        "<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:8px;'>"
        '<strong style="font-size:15px;">Runtime Environment</strong>'
        + render_status_badge(active_job.status if active_job else None)
        + "</div>"
        "<div style='font-size:13px;line-height:1.85;color:#cbd5f5;'>"
        f"• <strong>GPU / VRAM:</strong> {html.escape(str(gpu.get('summary', '-')))}<br>"
        f"• <strong>Stack:</strong> Python {html.escape(versions.get('python', '-'))} | "
        f"PyTorch {html.escape(versions.get('pytorch', '-'))} (CUDA {html.escape(versions.get('cuda_version', '-'))}) | "
        f"Ultralytics {html.escape(versions.get('ultralytics', '-'))} | "
        f"Gradio {html.escape(versions.get('gradio', '-'))}<br>"
        f"• <strong>Disk:</strong> {html.escape(str(disk.get('summary', '-')))} | "
        f"<strong>Workspace:</strong> <code>{html.escape(ov['workspace_root'])}</code>"
        "</div>"
        "</div>"
    )
    return kpi_html + env_html


def build_home_page(controller: AppController) -> dict[str, Any]:
    with gr.Column(visible=True) as container:
        gr.HTML(
            f"""
            <div class="hero-box">
                <div class="hero-title">{html.escape(APP_NAME)}</div>
                <div class="hero-sub">End-to-End YOLO11 Object Detection — Dataset, Annotation, Augmentation, Training, Testing & Export (.pt)</div>
                <div class="pill-row">
                    <span class="feature-pill feature-pill-accent">YOLO11 Detection</span>
                    <span class="feature-pill">End-to-End Workflow</span>
                    <span class="feature-pill">Temporary Storage</span>
                    <span class="feature-pill">Disconnect-Safe Worker</span>
                </div>
            </div>
            """
        )

        gr.Markdown("### Workflow")
        with gr.Row():
            btn_go_dataset = gr.Button("1. Dataset", variant="secondary")
            btn_go_annotate = gr.Button("2. Annotate", variant="secondary")
            btn_go_prepare = gr.Button("3. Prepare", variant="secondary")
            btn_go_training = gr.Button("4. Training", variant="primary")

        overview_html = gr.HTML(value=_render_overview_html(controller))
        btn_refresh_home = gr.Button("Refresh Status", size="sm")

        gr.HTML(render_temp_storage_warning(compact=True))

        with gr.Accordion("Quick Guide", open=False):
            gr.Markdown(
                """
                **ขั้นตอนใช้งานครบวงจร (Phase 1):**
                1. **Dataset:** นำเข้าภาพเปล่า (Mode 1) หรือชุดภาพ+Label `.txt` (Mode 2) หรือสร้าง Demo Dataset → แบ่ง Train/Val/Test → ทำ Offline Augmentation (เฉพาะชุด Train)
                2. **Annotate:** วาด/ย้าย/Resize/ลบกรอบบน Canvas เต็มพื้นที่ พร้อม Class Manager และ AI Assist Panel
                3. **Training Workspace:** ตั้งค่าและเริ่มเทรน YOLO11 (`n/s/m/l/x`, โหมด `Train New / Fine-tune / Resume`), ดูผล Real-time ฝั่งซ้าย, กด **Update Best Model** เพื่อทดสอบรูปเองฝั่งขวา และดาวน์โหลด `best.pt` / `last.pt` ด้านล่างโดยไม่ต้องหยุดเทรน
                """
            )

        btn_refresh_home.click(fn=lambda: _render_overview_html(controller), outputs=[overview_html])

    return {
        "container": container,
        "refresh_fn": lambda: _render_overview_html(controller),
        "refresh_outputs": [overview_html],
        "btn_go_dataset": btn_go_dataset,
        "btn_go_annotate": btn_go_annotate,
        "btn_go_prepare": btn_go_prepare,
        "btn_go_training": btn_go_training,
    }
