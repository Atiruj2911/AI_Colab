"""Page 3: Training Setup (YOLO11 Detection configuration & safe launch)."""

from __future__ import annotations

from typing import Any

import gradio as gr

from application.app_controller import AppController
from config.defaults import (
    ALLOWED_IMG_SIZES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_IMG_SIZE,
    DEFAULT_MODEL_VARIANT_LABEL,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    YOLO11_MODEL_VARIANTS,
)
from infrastructure.monitoring.system_monitor import get_available_devices
from ui.components.common import (
    render_page_header,
    render_planned_notice,
    render_temp_storage_warning,
)


def _valid_dataset_choices(controller: AppController) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for ds in controller.list_datasets():
        if ds.is_valid:
            label = f"✅ {ds.dataset_name} ({ds.total_images} images, {len(ds.class_names)} classes)"
        else:
            label = f"❌ {ds.dataset_name} (ยังไม่ผ่านการตรวจสอบ — {len(ds.errors)} errors)"
        choices.append((label, ds.dataset_id))
    return choices


def build_training_page(controller: AppController) -> dict[str, Any]:
    ds_choices = _valid_dataset_choices(controller)
    default_ds = ds_choices[0][1] if ds_choices else None
    device_choices = get_available_devices()
    default_device = device_choices[0][1] if device_choices else "auto"

    variant_labels = list(YOLO11_MODEL_VARIANTS.keys())
    default_weight = YOLO11_MODEL_VARIANTS[DEFAULT_MODEL_VARIANT_LABEL]["pretrained_weights"]

    with gr.Column(visible=False) as container:
        gr.HTML(
            render_page_header(
                "Training",
                "ตั้งค่าโมเดล YOLO11 Object Detection และเริ่มเทรนใน Background Worker แยกจากหน้าเว็บ",
            )
        )

        gr.HTML(render_temp_storage_warning(compact=True))

        with gr.Group(elem_classes=["studio-card"]):
            gr.Markdown("### ตั้งค่าการเทรน (YOLO11 Detection)")
            dataset_selector = gr.Dropdown(
                label="เลือก Dataset ที่ผ่านการตรวจสอบแล้ว",
                choices=ds_choices,
                value=default_ds,
                interactive=True,
            )

            with gr.Row():
                with gr.Column(scale=6):
                    model_variant_dd = gr.Dropdown(
                        label="Model (YOLO11)",
                        choices=variant_labels,
                        value=DEFAULT_MODEL_VARIANT_LABEL,
                        interactive=True,
                    )
                    pretrained_dd = gr.Dropdown(
                        label="Pretrained Weight",
                        choices=[
                            (f"{spec['pretrained_weights']} ({label.split(' ')[0]})", spec["pretrained_weights"])
                            for label, spec in YOLO11_MODEL_VARIANTS.items()
                        ],
                        value=default_weight,
                        interactive=True,
                    )
                    train_mode = gr.Radio(
                        label="โหมดการเทรน",
                        choices=[
                            ("Train ใหม่ (จาก Pretrained Weight — Phase 1)", "train"),
                            ("Fine-tune (ใช้ Weight เดิม — เพิ่มเติมใน Phase 4)", "fine_tune"),
                            ("Resume (เทรนต่อจาก Checkpoint — เพิ่มเติมใน Phase 4)", "resume"),
                        ],
                        value="train",
                        interactive=True,
                    )
                    run_name_input = gr.Textbox(
                        label="ชื่อ Run (ไม่บังคับ)",
                        placeholder="เช่น ppe_yolo11s_exp1",
                    )

                with gr.Column(scale=6):
                    epochs_num = gr.Number(
                        label="Epochs",
                        value=DEFAULT_EPOCHS,
                        minimum=1,
                        maximum=1000,
                        precision=0,
                    )
                    batch_num = gr.Number(
                        label="Batch Size",
                        value=DEFAULT_BATCH_SIZE,
                        minimum=1,
                        maximum=256,
                        precision=0,
                    )
                    imgsz_dd = gr.Dropdown(
                        label="Image Size (imgsz)",
                        choices=ALLOWED_IMG_SIZES,
                        value=DEFAULT_IMG_SIZE,
                        interactive=True,
                    )
                    device_dd = gr.Dropdown(
                        label="Device",
                        choices=device_choices,
                        value=default_device,
                        interactive=True,
                    )
                    with gr.Row():
                        seed_num = gr.Number(label="Random Seed", value=DEFAULT_SEED, precision=0)
                        patience_num = gr.Number(label="Early Stop Patience", value=DEFAULT_PATIENCE, precision=0)

            with gr.Accordion("⚙️ ตั้งค่าขั้นสูง (Advanced Hyperparameters — Phase 4)", open=False):
                gr.HTML(
                    render_planned_notice(
                        "Phase 4 — Model & Training Expansion",
                        "การปรับ Optimizer, Learning Rate, Scheduler, Weight Decay, Freeze Layers, AMP และโมเดล YOLOv8 / Segmentation / Classification จะเปิดใช้งานใน Phase 4",
                    )
                )

        with gr.Row():
            btn_start_train = gr.Button("▶ เริ่มการเทรน", variant="primary", scale=3)
            btn_open_monitor = gr.Button("📊 ไปที่หน้า Training Monitor", variant="secondary", scale=1)

        train_feedback = gr.Markdown(value="")

        def _sync_weight(variant_label: str):
            spec = YOLO11_MODEL_VARIANTS.get(variant_label)
            if not spec:
                return gr.update()
            return gr.update(value=spec["pretrained_weights"])

        def _refresh_training_choices(current_ds: str | None):
            choices = _valid_dataset_choices(controller)
            ids = {cid for _, cid in choices}
            session_ds = controller.session_service.get_or_create_session().get("active_dataset_id")
            if session_ds in ids:
                val = session_ds
            elif current_ds in ids:
                val = current_ds
            else:
                val = choices[0][1] if choices else None
            return gr.update(choices=choices, value=val), gr.update(choices=get_available_devices())

        def _handle_start_training(
            dataset_id: str | None,
            variant_label: str,
            pretrained_weights: str,
            mode: str,
            run_name: str,
            epochs: int,
            batch_size: int,
            imgsz: int,
            device: str,
            seed: int,
            patience: int,
        ) -> tuple[str, bool]:
            if mode != "train":
                return (
                    "⚠️ โหมด **Fine-tune** และ **Resume** อยู่ในแผนพัฒนา **Phase 4** — กรุณาเลือก **Train ใหม่** สำหรับ Phase 1 MVP",
                    False,
                )
            if not dataset_id:
                return "⚠️ กรุณาอัปโหลดและเลือก Dataset ที่หน้า **Dataset** ก่อนเริ่มการเทรน", False

            spec = YOLO11_MODEL_VARIANTS.get(variant_label)
            model_variant = spec["model_id"] if spec else "yolo11s"
            try:
                job = controller.start_training(
                    dataset_id=dataset_id,
                    model_variant=model_variant,
                    pretrained_weights=pretrained_weights,
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    imgsz=int(imgsz),
                    device=device,
                    seed=int(seed),
                    patience=int(patience),
                    run_name=run_name or None,
                    mode="train",
                )
                return (
                    f"🚀 **เริ่มการเทรนแล้ว!** Run: `{job.run_name}` (PID: `{job.pid}`, {job.total_epochs} epochs) — "
                    "ระบบกำลังสลับไปยังหน้า **Monitor** เพื่อติดตามผลแบบเรียลไทม์",
                    True,
                )
            except Exception as exc:
                return f"❌ ไม่สามารถเริ่มการเทรนได้: {exc}", False

        model_variant_dd.change(fn=_sync_weight, inputs=[model_variant_dd], outputs=[pretrained_dd])

    return {
        "container": container,
        "btn_start_train": btn_start_train,
        "btn_open_monitor": btn_open_monitor,
        "train_feedback": train_feedback,
        "start_fn": _handle_start_training,
        "start_inputs": [
            dataset_selector,
            model_variant_dd,
            pretrained_dd,
            train_mode,
            run_name_input,
            epochs_num,
            batch_num,
            imgsz_dd,
            device_dd,
            seed_num,
            patience_num,
        ],
        "refresh_fn": _refresh_training_choices,
        "refresh_inputs": [dataset_selector],
        "refresh_outputs": [dataset_selector, device_dd],
    }
