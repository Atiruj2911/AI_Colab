"""Page 6: Model Export (.pt and .onnx in Phase 1, planned formats disabled)."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import gradio as gr

from application.app_controller import AppController
from config.defaults import ALLOWED_IMG_SIZES, DEFAULT_IMG_SIZE
from core.schemas import ExportArtifact
from ui.components.common import render_page_header
from utils.validation import format_bytes


def _export_model_choices(controller: AppController) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for ckpt in controller.list_checkpoints():
        map_txt = f" (mAP50-95: {ckpt.best_map50_95:.3f})" if ckpt.best_map50_95 is not None else ""
        label = f"{ckpt.name}{map_txt} — {format_bytes(ckpt.size_bytes)}"
        choices.append((label, ckpt.path))
    return choices


def _render_export_status_html(artifacts: list[ExportArtifact], bundle_dir: Path | None = None) -> str:
    if not artifacts:
        return (
            '<div class="studio-card">'
            "<h4 style='margin-top:0;'>สถานะการ Export</h4>"
            "<div style='color:#93a4cc;font-size:13px;'>เลือกโมเดลและรูปแบบที่ต้องการ แล้วกด <strong>เริ่ม Export</strong></div>"
            "</div>"
        )

    ok_count = sum(1 for a in artifacts if a.success)
    fail_count = len(artifacts) - ok_count
    if fail_count == 0:
        banner = '<span class="status-badge badge-success">✅ Export สำเร็จทั้งหมด</span>'
    elif ok_count > 0:
        banner = f'<span class="status-badge badge-warning">⚠️ สำเร็จ {ok_count} / ล้มเหลว {fail_count}</span>'
    else:
        banner = '<span class="status-badge badge-danger">❌ Export ไม่สำเร็จ</span>'

    rows_html: list[str] = []
    for art in artifacts:
        if art.success:
            rows_html.append(
                "<div style='padding:10px 12px;margin-top:8px;border-radius:10px;background:#0b1224;border:1px solid #24365e;'>"
                f"<strong>✅ {html.escape(art.filename)}</strong> "
                f"<span style='color:#93a4cc;'>({format_bytes(art.size_bytes)})</span>"
                f"<div style='font-size:12px;color:#86efac;margin-top:2px;'>{html.escape(art.message)}</div>"
                "</div>"
            )
        else:
            rows_html.append(
                "<div style='padding:10px 12px;margin-top:8px;border-radius:10px;background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.4);'>"
                f"<strong>❌ {html.escape(art.format.upper())}</strong>"
                f"<div style='font-size:12px;color:#fecaca;margin-top:2px;'>{html.escape(art.message)}</div>"
                "</div>"
            )

    dir_note = (
        f"<div style='font-size:12px;color:#93a4cc;margin-top:10px;'>โฟลเดอร์ผลลัพธ์: <code>{html.escape(str(bundle_dir))}</code></div>"
        if bundle_dir
        else ""
    )

    return (
        '<div class="studio-card">'
        "<div style='display:flex;justify-content:space-between;align-items:center;gap:8px;'>"
        "<h4 style='margin:0;'>สถานะการ Export</h4>"
        f"{banner}"
        "</div>"
        + "".join(rows_html)
        + dir_note
        + "</div>"
    )


def build_export_page(controller: AppController) -> dict[str, Any]:
    choices = _export_model_choices(controller)
    default_ckpt = choices[0][1] if choices else None

    with gr.Column(visible=False) as container:
        gr.HTML(
            render_page_header(
                "Export Model",
                "ส่งออกและดาวน์โหลดโมเดลที่เทรนแล้ว (.pt และ .onnx) พร้อมไฟล์ config และ data.yaml",
            )
        )

        with gr.Row():
            with gr.Column(scale=6):
                with gr.Group(elem_classes=["studio-card"]):
                    export_model_dd = gr.Dropdown(
                        label="เลือกโมเดลสำหรับ Export",
                        choices=choices,
                        value=default_ckpt,
                        interactive=True,
                    )
                    upload_pt_export = gr.File(
                        label="หรืออัปโหลดไฟล์ .pt เพื่อ Export (ไม่บังคับ)",
                        file_types=[".pt"],
                        type="filepath",
                    )
                    export_imgsz = gr.Dropdown(
                        label="Image Size สำหรับ ONNX Export",
                        choices=ALLOWED_IMG_SIZES,
                        value=DEFAULT_IMG_SIZE,
                        interactive=True,
                    )
                    btn_refresh_export = gr.Button("🔄 รีเฟรชรายการโมเดล", size="sm")

                    gr.Markdown("#### รูปแบบการ Export")
                    formats_active = gr.CheckboxGroup(
                        label="รูปแบบที่รองรับใน Phase 1",
                        choices=[
                            ("PyTorch (.pt)", "pt"),
                            ("ONNX (.onnx)", "onnx"),
                        ],
                        value=["pt", "onnx"],
                    )
                    gr.CheckboxGroup(
                        label="รูปแบบเพิ่มเติมในแผน (Disabled ใน Phase 1)",
                        choices=[
                            "TorchScript (.torchscript) — เพิ่มเติมใน Phase 5",
                            "OpenVINO — เพิ่มเติมใน Phase 5",
                            "TensorRT (.engine) — เพิ่มเติมใน Phase 5",
                        ],
                        value=[],
                        interactive=False,
                    )
                    btn_export = gr.Button("▶ เริ่ม Export", variant="primary")

            with gr.Column(scale=6):
                export_status_html = gr.HTML(value=_render_export_status_html([]))
                exported_files_download = gr.File(
                    label="ดาวน์โหลดไฟล์ที่ Export แล้ว (รวม config / data.yaml)",
                    interactive=False,
                )

        def _refresh_export_models(current_val: str | None):
            new_choices = _export_model_choices(controller)
            valid_paths = {p for _, p in new_choices}
            val = current_val if current_val in valid_paths else (new_choices[0][1] if new_choices else None)
            return gr.update(choices=new_choices, value=val)

        def _run_export(
            selected_model: str | None,
            uploaded_model: str | None,
            selected_formats: list[str],
            imgsz: int,
        ):
            weights = uploaded_model or selected_model
            if not weights:
                return (
                    '<div class="callout-warning">⚠️ กรุณาเลือกโมเดลจากรายการหรืออัปโหลดไฟล์ <code>.pt</code> ก่อน Export</div>',
                    None,
                )
            if not selected_formats:
                return (
                    '<div class="callout-warning">⚠️ กรุณาเลือกอย่างน้อย 1 รูปแบบการ Export (เช่น PyTorch หรือ ONNX)</div>',
                    None,
                )
            try:
                bundle_dir, artifacts = controller.export_model(
                    weights_path=weights,
                    formats=selected_formats,
                    imgsz=int(imgsz),
                )
                download_paths: list[str] = [a.path for a in artifacts if a.success and a.path and Path(a.path).exists()]
                for extra_name in ("train_config.json", "data.yaml", "export_manifest.json"):
                    extra_file = bundle_dir / extra_name
                    if extra_file.exists():
                        download_paths.append(str(extra_file))
                return (
                    _render_export_status_html(artifacts, bundle_dir=bundle_dir),
                    download_paths if download_paths else None,
                )
            except Exception as exc:
                return (
                    f'<div class="callout-danger">❌ Export ไม่สำเร็จ: {html.escape(str(exc))}</div>',
                    None,
                )

        btn_refresh_export.click(fn=_refresh_export_models, inputs=[export_model_dd], outputs=[export_model_dd])
        btn_export.click(
            fn=_run_export,
            inputs=[export_model_dd, upload_pt_export, formats_active, export_imgsz],
            outputs=[export_status_html, exported_files_download],
        )

    return {
        "container": container,
        "refresh_fn": _refresh_export_models,
        "refresh_inputs": [export_model_dd],
        "refresh_outputs": [export_model_dd],
    }
