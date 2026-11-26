"""Page 5: Model Testing / Detect (model selection, conf/IoU controls, annotated image & table)."""

from __future__ import annotations

from typing import Any

import gradio as gr

from application.app_controller import AppController
from config.defaults import DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD
from ui.components.common import render_page_header, render_planned_notice
from utils.validation import format_bytes


def _checkpoint_choices(controller: AppController) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for ckpt in controller.list_checkpoints():
        map_txt = f" | mAP50-95: {ckpt.best_map50_95:.3f}" if ckpt.best_map50_95 is not None else ""
        label = f"{ckpt.name} ({format_bytes(ckpt.size_bytes)}{map_txt})"
        choices.append((label, ckpt.path))
    return choices


def build_testing_page(controller: AppController) -> dict[str, Any]:
    choices = _checkpoint_choices(controller)
    default_ckpt = choices[0][1] if choices else None

    with gr.Column(visible=False) as container:
        gr.HTML(
            render_page_header(
                "Testing / Detect",
                "ทดสอบโมเดลกับรูปภาพด้วย Confidence และ IoU (NMS) พร้อมแสดงกรอบตรวจจับและตารางรายละเอียด",
            )
        )

        with gr.Row():
            with gr.Column(scale=5):
                with gr.Group(elem_classes=["studio-card"]):
                    model_dd = gr.Dropdown(
                        label="เลือกโมเดลจาก Runs (.pt)",
                        choices=choices,
                        value=default_ckpt,
                        interactive=True,
                    )
                    custom_weight_file = gr.File(
                        label="หรืออัปโหลดไฟล์น้ำหนัก (.pt) จากเครื่อง (ไม่บังคับ)",
                        file_types=[".pt"],
                        type="filepath",
                    )
                    btn_refresh_models = gr.Button("🔄 รีเฟรชรายการโมเดล", size="sm")

                with gr.Group(elem_classes=["studio-card"]):
                    conf_slider = gr.Slider(
                        label="Confidence",
                        minimum=0.05,
                        maximum=0.95,
                        step=0.01,
                        value=DEFAULT_CONF_THRESHOLD,
                    )
                    iou_slider = gr.Slider(
                        label="IoU (NMS)",
                        minimum=0.05,
                        maximum=0.95,
                        step=0.01,
                        value=DEFAULT_IOU_THRESHOLD,
                    )
                    test_image = gr.Image(
                        label="อัปโหลดภาพสำหรับทดสอบ",
                        type="filepath",
                        height=240,
                    )
                    btn_detect = gr.Button("▶ ตรวจจับ (Detect)", variant="primary")
                    detect_status = gr.Markdown(value="")

            with gr.Column(scale=7):
                with gr.Tabs():
                    with gr.Tab("ผลการตรวจจับ"):
                        result_image = gr.Image(label="ภาพผลลัพธ์พร้อม Bounding Boxes", type="filepath", height=380)
                    with gr.Tab("ภาพต้นฉบับ"):
                        original_image = gr.Image(label="ภาพต้นฉบับ", type="filepath", height=380)

                detections_table = gr.Dataframe(
                    headers=["#", "Class", "Confidence", "x1", "y1", "x2", "y2"],
                    label="รายการวัตถุที่ตรวจพบ",
                    interactive=False,
                )

                gr.HTML(
                    render_planned_notice(
                        "Phase 5 — Advanced Testing",
                        "การทดสอบผ่าน Video และ Webcam แบบเรียลไทม์อยู่ในแผนพัฒนา Phase 5",
                    )
                )

        def _refresh_models(current_val: str | None):
            new_choices = _checkpoint_choices(controller)
            valid_paths = {p for _, p in new_choices}
            val = current_val if current_val in valid_paths else (new_choices[0][1] if new_choices else None)
            return gr.update(choices=new_choices, value=val)

        def _run_detect(
            selected_model: str | None,
            uploaded_weight: str | None,
            img_path: str | None,
            conf: float,
            iou: float,
        ):
            weights_path = uploaded_weight or selected_model
            if not weights_path:
                return (
                    "⚠️ กรุณาเลือกโมเดลจากการเทรน หรืออัปโหลดไฟล์ `.pt` ก่อนทดสอบ",
                    None,
                    img_path,
                    [],
                )
            if not img_path:
                return (
                    "⚠️ กรุณาอัปโหลดรูปภาพสำหรับทดสอบตรวจจับ",
                    None,
                    None,
                    [],
                )
            try:
                res = controller.run_inference(
                    weights_path=weights_path,
                    image_path=img_path,
                    conf=conf,
                    iou=iou,
                )
                rows = [
                    [
                        idx + 1,
                        d.class_name,
                        round(d.confidence, 4),
                        round(d.x1, 1),
                        round(d.y1, 1),
                        round(d.x2, 1),
                        round(d.y2, 1),
                    ]
                    for idx, d in enumerate(res.detections)
                ]
                summary = (
                    f"✅ ตรวจจับเสร็จสิ้นใน **{res.inference_ms:.1f} ms** "
                    f"({res.image_width}×{res.image_height} px) — พบวัตถุ **{len(res.detections)}** รายการ"
                )
                return summary, res.annotated_image_path, img_path, rows
            except Exception as exc:
                return f"❌ เกิดข้อผิดพลาดระหว่างตรวจจับ: {exc}", None, img_path, []

        btn_refresh_models.click(fn=_refresh_models, inputs=[model_dd], outputs=[model_dd])
        btn_detect.click(
            fn=_run_detect,
            inputs=[model_dd, custom_weight_file, test_image, conf_slider, iou_slider],
            outputs=[detect_status, result_image, original_image, detections_table],
        )

    return {
        "container": container,
        "refresh_fn": _refresh_models,
        "refresh_inputs": [model_dd],
        "refresh_outputs": [model_dd],
    }
