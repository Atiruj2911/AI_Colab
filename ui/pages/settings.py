"""Page 8: Settings (runtime versions, workspace storage cleanup with confirmation, auto-save preview, UI refresh)."""

from __future__ import annotations

import html
from typing import Any

import gradio as gr

from application.app_controller import AppController
from config.defaults import (
    DEFAULT_MONITOR_REFRESH_SECONDS,
    MAX_MONITOR_REFRESH_SECONDS,
    MIN_MONITOR_REFRESH_SECONDS,
)
from ui.components.common import render_page_header, render_planned_notice
from utils.validation import format_bytes


def _render_general_info(controller: AppController) -> str:
    ov = controller.get_system_overview()
    v = ov["versions"]
    gpu = ov["gpu"]
    rows = [
        ("Python", v.get("python", "-")),
        ("Platform", v.get("platform", "-")),
        ("PyTorch", v.get("pytorch", "-")),
        ("CUDA", v.get("cuda_version", "-")),
        ("GPU", gpu.get("summary", "-")),
        ("Ultralytics", v.get("ultralytics", "-")),
        ("Gradio", v.get("gradio", "-")),
        ("ONNX / ONNXRuntime", f"{v.get('onnx', '-')} / {v.get('onnxruntime', '-')}"),
        ("Workspace Root", ov["workspace_root"]),
        ("Google Colab Detected", "ใช่ (Temporary Storage)" if ov["is_colab"] else "ไม่ใช่ (Local Environment)"),
    ]
    tr_html = "".join(
        f"<tr style='border-bottom:1px solid #1e2d50;'>"
        f"<td style='padding:8px 10px;color:#93a4cc;width:34%;'><strong>{html.escape(k)}</strong></td>"
        f"<td style='padding:8px 10px;color:#e5ecff;'><code>{html.escape(str(val))}</code></td>"
        f"</tr>"
        for k, val in rows
    )
    return (
        '<div class="studio-card">'
        "<h4 style='margin-top:0;'>ข้อมูลสภาพแวดล้อมและเวอร์ชันแพ็กเกจ (Runtime Diagnostics)</h4>"
        f"<table style='width:100%;border-collapse:collapse;font-size:13px;'>{tr_html}</table>"
        "</div>"
    )


def _render_storage_html(controller: AppController) -> str:
    ov = controller.get_system_overview()
    disk = ov["disk"]
    usage = ov["workspace_usage"]
    labels = {
        "uploads": "uploads/ (ไฟล์อัปโหลดต้นฉบับ)",
        "datasets": "datasets/ (ชุดข้อมูลที่แตกและตรวจแล้ว)",
        "weights": "weights/ (Pretrained weights cache)",
        "runs": "runs/ (ผลเทรน, checkpoints, logs)",
        "exports": "exports/ (โมเดล .pt / .onnx ที่ส่งออก)",
        "temp": "temp/ (ภาพ preview / inference ชั่วคราว)",
        "sessions": "sessions/ (สถานะเซสชัน)",
        "logs": "logs/ (System logs)",
    }
    rows = "".join(
        f"<tr style='border-bottom:1px solid #1e2d50;'>"
        f"<td style='padding:7px 10px;color:#cbd5f5;'>{html.escape(desc)}</td>"
        f"<td style='padding:7px 10px;text-align:right;color:#f8faff;'><strong>{format_bytes(usage.get(key, 0))}</strong></td>"
        f"</tr>"
        for key, desc in labels.items()
    )
    return (
        '<div class="studio-card">'
        "<h4 style='margin-top:0;'>การใช้งานพื้นที่จัดเก็บ (Workspace Storage)</h4>"
        f"<div style='font-size:13px;color:#cbd5f5;margin-bottom:8px;'>"
        f"<strong>ดิสก์ทั้งหมด:</strong> {html.escape(str(disk['summary']))} | "
        f"<strong>Workspace รวม:</strong> <code>{format_bytes(usage.get('total', 0))}</code>"
        "</div>"
        f"<table style='width:100%;border-collapse:collapse;font-size:13px;'>{rows}</table>"
        "</div>"
    )


def build_settings_page(controller: AppController) -> dict[str, Any]:
    with gr.Column(visible=False) as container:
        gr.HTML(
            render_page_header(
                "Settings",
                "ตรวจสอบสภาพแวดล้อม จัดการพื้นที่จัดเก็บ และตั้งค่าการรีเฟรชหน้า Monitor",
            )
        )

        with gr.Tabs():
            with gr.Tab("ทั่วไป"):
                general_html = gr.HTML(value=_render_general_info(controller))
                btn_refresh_general = gr.Button("🔄 รีเฟรชข้อมูลสภาพแวดล้อม", size="sm")

            with gr.Tab("Auto Save"):
                gr.HTML(
                    render_planned_notice(
                        "Phase 4 — Configurable Auto-Save Policy",
                        "ใน Phase 1 ระบบบันทึก best.pt และ last.pt อัตโนมัติทุกการเทรน การปรับเงื่อนไข Auto-Save แบบละเอียด (ทุกการดีขึ้น >= X / ทุก N Epoch / ทุก N นาที) ระหว่างเทรนจะเปิดใช้งานใน Phase 4",
                    )
                )
                with gr.Group(elem_classes=["studio-card"]):
                    gr.Checkbox(label="เปิดใช้งาน Auto Save (best.pt + last.pt — ค่าเริ่มต้น MVP)", value=True, interactive=False)
                    gr.Radio(
                        label="เงื่อนไขการบันทึกเพิ่มเติม (Phase 4)",
                        choices=[
                            "เมื่อ Best Model ดีขึ้น >= 0.02 (Phase 4)",
                            "บันทึกทุก ๆ 5 Epoch (Phase 4)",
                            "บันทึกทุก ๆ 10 นาที (Phase 4)",
                        ],
                        value="เมื่อ Best Model ดีขึ้น >= 0.02 (Phase 4)",
                        interactive=False,
                    )

            with gr.Tab("พื้นที่จัดเก็บ"):
                storage_html = gr.HTML(value=_render_storage_html(controller))
                with gr.Group(elem_classes=["studio-card"]):
                    gr.Markdown("#### ล้างข้อมูลใน Workspace (ต้องยืนยันก่อนลบ)")
                    cleanup_target = gr.Dropdown(
                        label="เลือกโฟลเดอร์ที่ต้องการล้าง",
                        choices=[
                            ("temp/ (ไฟล์ภาพ preview & inference ชั่วคราว — ปลอดภัย)", "temp"),
                            ("exports/ (ไฟล์ Export เก่า)", "exports"),
                            ("uploads/ (ไฟล์ ZIP ต้นฉบับที่อัปโหลด)", "uploads"),
                            ("datasets/ (Dataset ทั้งหมดใน Workspace)", "datasets"),
                            ("runs/ (ประวัติและ Checkpoints ของทุก Run)", "runs"),
                        ],
                        value="temp",
                    )
                    confirm_cleanup = gr.Checkbox(
                        label="ยืนยันว่าต้องการลบไฟล์ทั้งหมดในโฟลเดอร์ที่เลือก (ไม่สามารถกู้คืนได้)",
                        value=False,
                    )
                    with gr.Row():
                        btn_cleanup = gr.Button("🗑️ ล้างโฟลเดอร์ที่เลือก", elem_classes=["danger-btn"])
                        btn_refresh_storage = gr.Button("🔄 รีเฟรชพื้นที่จัดเก็บ", variant="secondary")
                    cleanup_feedback = gr.Markdown(value="")

            with gr.Tab("UI"):
                with gr.Group(elem_classes=["studio-card"]):
                    gr.Markdown("#### การตั้งค่าการอัปเดตหน้า Monitor")
                    refresh_interval_slider = gr.Slider(
                        label="ความถี่ในการรีเฟรชหน้า Monitor อัตโนมัติ (วินาที)",
                        minimum=MIN_MONITOR_REFRESH_SECONDS,
                        maximum=MAX_MONITOR_REFRESH_SECONDS,
                        step=0.5,
                        value=DEFAULT_MONITOR_REFRESH_SECONDS,
                    )
                    auto_refresh_toggle = gr.Checkbox(
                        label="เปิดการรีเฟรชหน้า Monitor อัตโนมัติ",
                        value=True,
                    )
                    ui_settings_feedback = gr.Markdown(
                        value="ปรับความถี่ได้ทันทีเพื่อให้เหมาะกับความเร็วเน็ต/Colab"
                    )

        def _do_cleanup(subdir: str, confirmed: bool):
            ok, msg = controller.clear_workspace_folder(subdir, confirmed)
            prefix = "✅ " if ok else "⚠️ "
            return f"{prefix}{msg}", False, _render_storage_html(controller)

        def _refresh_all():
            return _render_general_info(controller), _render_storage_html(controller)

        btn_refresh_general.click(fn=lambda: _render_general_info(controller), outputs=[general_html])
        btn_refresh_storage.click(fn=lambda: _render_storage_html(controller), outputs=[storage_html])
        btn_cleanup.click(
            fn=_do_cleanup,
            inputs=[cleanup_target, confirm_cleanup],
            outputs=[cleanup_feedback, confirm_cleanup, storage_html],
        )

    return {
        "container": container,
        "refresh_interval_slider": refresh_interval_slider,
        "auto_refresh_toggle": auto_refresh_toggle,
        "ui_settings_feedback": ui_settings_feedback,
        "refresh_fn": _refresh_all,
        "refresh_outputs": [general_html, storage_html],
    }
