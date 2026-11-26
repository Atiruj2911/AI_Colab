"""Roboflow-style 3-column Annotate & AI Auto-Label workspace.

Layout:
- Left: Classes (add/counts), Layers (boxes in current image + quick delete), Find Objects with AI (single + batch)
- Center: Interactive bounding-box canvas (draw/move/resize/delete) + Save / Mark as Null / Undo
- Right: Always-visible Image Strip (1/N, < >, Split, Filter, thumbnails with checkmarks) — auto-saves on switch
"""

from __future__ import annotations

from typing import Any

import gradio as gr
from gradio_image_annotation import image_annotator

from application.app_controller import AppController
from core.dataset.entries import SPLITS
from utils.validation import format_bytes

MODE_CHOICES = [("ทั้งหมด", "all"), ("ยังไม่ตีกรอบ", "unlabeled"), ("ตีกรอบแล้ว", "labeled")]
PRETRAINED_CHOICES = [
    ("yolo11n.pt (โมเดลเล็ก เร็ว)", "yolo11n.pt"),
    ("yolo11s.pt (โมเดลกลาง)", "yolo11s.pt"),
]


def _model_choices(controller: AppController) -> list[tuple[str, str]]:
    trained = [
        (f"⭐ {c.name} ({format_bytes(c.size_bytes)})", c.path)
        for c in controller.list_checkpoints()
    ]
    return trained + PRETRAINED_CHOICES


def _list_entries(controller: AppController, ds: str | None, split: str, mode: str) -> list[dict[str, Any]]:
    if not ds:
        return []
    try:
        return controller.annotation_service.list_images(ds, split, mode, limit=500)
    except Exception:
        return []


def _format_classes_md(controller: AppController, ds: str | None) -> str:
    if not ds:
        return "_ยังไม่ได้เลือก Dataset_"
    try:
        rows = controller.annotation_service.list_classes(ds)
    except Exception:
        return "_ไม่พบข้อมูลคลาส_"
    if not rows:
        return "_ยังไม่มีคลาส_"
    return "\n".join(f"- `{r['id']}` **{r['name']}** ({r['count']})" for r in rows)


def _format_layers(boxes: list[dict[str, Any]]) -> tuple[str, list[tuple[str, int]]]:
    if not boxes:
        return "_ยังไม่มีกรอบในภาพนี้_", []
    lines = []
    choices: list[tuple[str, int]] = []
    for idx, b in enumerate(boxes):
        w = int(round(b["xmax"] - b["xmin"]))
        h = int(round(b["ymax"] - b["ymin"]))
        label = b.get("label", "?")
        lines.append(f"{idx + 1}. **{label}** ({w}×{h} px)")
        choices.append((f"#{idx + 1} {label} ({w}×{h})", idx))
    return "\n".join(lines), choices


def _gallery_items(entries: list[dict[str, Any]]) -> list[tuple[str, str]]:
    items = []
    for e in entries:
        if e["has_label"]:
            tag = f"✅ {e['box_count']}" if e["box_count"] > 0 else "✅ Null"
        else:
            tag = "⬜"
        items.append((str(e["image_path"]), f"{tag} {e['key']}"))
    return items


def _load_image_state(controller: AppController, ds: str | None, entries: list[dict[str, Any]], index: int):
    if not ds or not entries:
        return (
            0,
            "0 / 0",
            "ไม่มีภาพในเงื่อนไขที่เลือก",
            gr.update(value=None, label_list=[]),
            "_ยังไม่มีกรอบในภาพนี้_",
            gr.update(choices=[], value=None),
        )
    idx = max(0, min(index, len(entries) - 1))
    e = entries[idx]
    data = controller.annotation_service.load_annotator_value(ds, e["split"], e["key"])
    if data["has_label"]:
        status = f"✅ {data['box_count']} กรอบ" if data["box_count"] > 0 else "✅ Null (ภาพว่าง)"
    else:
        status = "⬜ ยังไม่ตีกรอบ"
    counter = f"**{idx + 1} / {len(entries)}**"
    header = f"`{e['split']}/{e['key']}` · {data['width']}×{data['height']} px · {status}"
    layers_md, layer_choices = _format_layers(data["value"]["boxes"])
    return (
        idx,
        counter,
        header,
        gr.update(value=data["value"], label_list=data["label_list"]),
        layers_md,
        gr.update(choices=layer_choices, value=(layer_choices[0][1] if layer_choices else None)),
    )


def build_annotate_tab(controller: AppController, shared_ds_dd: gr.Dropdown) -> dict[str, Any]:
    init_ds = shared_ds_dd.value
    init_entries = _list_entries(controller, init_ds, "train", "all")
    init_idx, init_counter, init_header, init_ann_upd, init_layers_md, init_layer_dd = _load_image_state(
        controller, init_ds, init_entries, 0
    )
    init_value = init_ann_upd.get("value") if isinstance(init_ann_upd, dict) else None
    init_labels = init_ann_upd.get("label_list", []) if isinstance(init_ann_upd, dict) else []

    entries_state = gr.State(init_entries)
    index_state = gr.State(init_idx)

    # ======================================================== 3-Column Roboflow Layout
    with gr.Row(equal_height=False):
        # -------------------- LEFT: Classes, Layers, AI Auto-Label
        with gr.Column(scale=2, min_width=210):
            gr.Markdown("#### 🏷️ Classes")
            with gr.Row(elem_classes=["compact-row"]):
                txt_new_class = gr.Textbox(placeholder="เพิ่มคลาส เช่น person", show_label=False, scale=4)
                btn_add_class = gr.Button("➕", size="sm", scale=1)
            classes_md = gr.Markdown(_format_classes_md(controller, init_ds))

            with gr.Accordion("📋 Layers (กรอบในภาพนี้)", open=False):
                layers_md = gr.Markdown(init_layers_md)
                with gr.Row(elem_classes=["compact-row"]):
                    layer_dd = gr.Dropdown(
                        label="เลือกกรอบเพื่อลบ",
                        choices=(init_layer_dd.get("choices", []) if isinstance(init_layer_dd, dict) else []),
                        value=(init_layer_dd.get("value") if isinstance(init_layer_dd, dict) else None),
                        scale=4,
                    )
                    btn_del_layer = gr.Button("🗑️", size="sm", variant="stop", scale=1)

            gr.Markdown("---")
            gr.Markdown("#### ✨ Find Objects with AI")
            model_dd = gr.Dropdown(
                label="โมเดล",
                choices=_model_choices(controller),
                value=_model_choices(controller)[0][1],
            )
            custom_pt = gr.File(label="หรืออัปโหลด .pt เอง", file_types=[".pt"], type="filepath")
            conf_sl = gr.Slider(label="Confidence", minimum=0.05, maximum=0.95, step=0.05, value=0.25)
            chk_auto_add_cls = gr.Checkbox(label="เพิ่มคลาสใหม่อัตโนมัติ", value=True)
            btn_ai_single = gr.Button("✨ ทำนายภาพนี้ (ทดลอง)", variant="secondary", size="sm")

            with gr.Accordion("🤖 ตีกรอบทั้งชุด (Batch)", open=False):
                auto_scope = gr.Radio(
                    label="ขอบเขต",
                    choices=[("เฉพาะภาพที่ยังไม่ตีกรอบ", "unlabeled"), ("ทุกภาพใน Split นี้", "all")],
                    value="unlabeled",
                )
                chk_auto_overwrite = gr.Checkbox(label="เขียนทับ Label เดิม", value=False)
                btn_ai_batch = gr.Button("▶ ตีกรอบทั้งชุด", variant="primary", size="sm")
            auto_msg = gr.Markdown("")

        # -------------------- CENTER: Large Canvas + Compact Action Bar
        with gr.Column(scale=8, min_width=560):
            img_info_md = gr.Markdown(init_header)
            annotator = image_annotator(
                value=init_value,
                label="ลาก=วาดกรอบ · คลิก=ย้าย/ย่อขยาย · Delete=ลบ (บันทึกอัตโนมัติเมื่อเปลี่ยนภาพ)",
                label_list=init_labels,
                sources=[],
                height=620,
                use_default_label=True,
                enable_keyboard_shortcuts=True,
                show_clear_button=False,
                show_download_button=False,
            )
            with gr.Row(elem_classes=["compact-row"]):
                btn_save = gr.Button("💾 บันทึก", variant="primary", size="sm", scale=2)
                btn_null = gr.Button("⬜ Mark as Null", variant="secondary", size="sm", scale=2)
                btn_undo = gr.Button("↩️ ย้อนกลับ", variant="secondary", size="sm", scale=1)
            save_msg = gr.Markdown("")

            with gr.Accordion("✏️ เปลี่ยนชื่อ / ลบภาพนี้", open=False):
                with gr.Row(elem_classes=["compact-row"]):
                    txt_rename = gr.Textbox(label="ชื่อใหม่ (ไม่ใส่นามสกุล)", scale=4)
                    btn_rename = gr.Button("เปลี่ยนชื่อ", size="sm", scale=1)
                with gr.Row(elem_classes=["compact-row"]):
                    chk_del = gr.Checkbox(label="ยืนยันการลบภาพและ Label นี้", value=False, scale=4)
                    btn_del = gr.Button("🗑️ ลบภาพนี้", variant="stop", size="sm", scale=1)
                file_msg = gr.Markdown("")

        # -------------------- RIGHT: Always-visible Image Strip (Roboflow style)
        with gr.Column(scale=2, min_width=195):
            gr.Markdown("#### 🖼️ Images")
            with gr.Row(elem_classes=["compact-row"]):
                btn_prev = gr.Button("◀", size="sm", scale=1)
                counter_md = gr.Markdown(init_counter)
                btn_next = gr.Button("▶", size="sm", scale=1)
            split_rd = gr.Dropdown(choices=list(SPLITS), value="train", label="Split")
            mode_rd = gr.Dropdown(choices=MODE_CHOICES, value="all", label="กรองภาพ")
            thumb_gallery = gr.Gallery(
                value=_gallery_items(init_entries),
                columns=2,
                height=520,
                allow_preview=False,
                object_fit="contain",
                show_label=False,
            )

    # ================================================================ helpers & handlers
    def _autosave_if_changed(ds, entries, idx, value) -> str:
        if not ds or not entries or not value:
            return ""
        e = entries[max(0, min(idx, len(entries) - 1))]
        try:
            res = controller.annotation_service.save_annotator_value(
                ds, e["split"], e["key"], value, only_if_changed=True
            )
            if res["changed"]:
                return f"💾 บันทึกอัตโนมัติ `{e['key']}` ({res['box_count']} กรอบ)"
        except Exception:
            pass
        return ""

    def _reload_all(ds, split, mode, keep_index=0):
        entries = _list_entries(controller, ds, split, mode)
        idx, counter, header, ann_upd, l_md, l_dd = _load_image_state(controller, ds, entries, keep_index)
        return (
            entries,
            idx,
            counter,
            header,
            ann_upd,
            l_md,
            l_dd,
            _format_classes_md(controller, ds),
            _gallery_items(entries),
            gr.update(choices=_model_choices(controller)),
        )

    def _switch_image(ds, split, mode, entries, cur_idx, value, target_idx):
        if not ds or not entries:
            return entries, 0, "0 / 0", "ไม่มีภาพ", gr.update(value=None), "_ไม่มีกรอบ_", gr.update(choices=[], value=None), _format_classes_md(controller, ds), [], ""
        note = _autosave_if_changed(ds, entries, cur_idx, value)
        # Refresh list if auto-save changed label status
        new_entries = _list_entries(controller, ds, split, mode) if note else entries
        if not new_entries:
            new_entries = _list_entries(controller, ds, split, "all")
        idx, counter, header, ann_upd, l_md, l_dd = _load_image_state(controller, ds, new_entries, target_idx)
        return new_entries, idx, counter, header, ann_upd, l_md, l_dd, _format_classes_md(controller, ds), _gallery_items(new_entries), note

    def on_prev(ds, split, mode, entries, idx, value):
        target = (idx - 1) % max(len(entries), 1)
        return _switch_image(ds, split, mode, entries, idx, value, target)

    def on_next(ds, split, mode, entries, idx, value):
        target = (idx + 1) % max(len(entries), 1)
        return _switch_image(ds, split, mode, entries, idx, value, target)

    def on_gallery_select(ds, split, mode, entries, cur_idx, value, evt: gr.SelectData):
        return _switch_image(ds, split, mode, entries, cur_idx, value, evt.index)

    def on_save(ds, split, mode, entries, idx, value):
        if not ds or not entries:
            return entries, idx, "0 / 0", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "⚠️ ไม่มีภาพ"
        e = entries[max(0, min(idx, len(entries) - 1))]
        try:
            res = controller.annotation_service.save_annotator_value(ds, e["split"], e["key"], value, only_if_changed=False)
            new_entries = _list_entries(controller, ds, split, mode)
            new_idx, counter, header, ann_upd, l_md, l_dd = _load_image_state(controller, ds, new_entries, idx)
            extra = f" · เพิ่มคลาส: {', '.join(res['added_classes'])}" if res["added_classes"] else ""
            msg = f"✅ บันทึก `{e['key']}` แล้ว ({res['box_count']} กรอบ){extra}"
            return new_entries, new_idx, counter, header, ann_upd, l_md, l_dd, _format_classes_md(controller, ds), _gallery_items(new_entries), msg
        except Exception as exc:
            return entries, idx, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), f"❌ {exc}"

    def on_mark_null(ds, split, mode, entries, idx):
        if not ds or not entries:
            return entries, idx, "0 / 0", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "⚠️ ไม่มีภาพ"
        e = entries[max(0, min(idx, len(entries) - 1))]
        try:
            controller.annotation_service.mark_as_null(ds, e["split"], e["key"])
            new_entries = _list_entries(controller, ds, split, mode)
            target = idx if mode == "unlabeled" else (idx + 1 if idx + 1 < len(new_entries) else 0)
            new_idx, counter, header, ann_upd, l_md, l_dd = _load_image_state(controller, ds, new_entries, target)
            return new_entries, new_idx, counter, header, ann_upd, l_md, l_dd, _format_classes_md(controller, ds), _gallery_items(new_entries), f"⬜ บันทึก `{e['key']}` เป็น Null (ภาพว่าง) แล้ว"
        except Exception as exc:
            return entries, idx, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), f"❌ {exc}"

    def on_undo(ds, split, mode, entries, idx):
        if not ds or not entries:
            return entries, idx, "0 / 0", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "⚠️ ไม่มีภาพ"
        e = entries[max(0, min(idx, len(entries) - 1))]
        try:
            msg = controller.annotation_service.undo_last_save(ds, e["split"], e["key"])
            new_entries = _list_entries(controller, ds, split, mode)
            new_idx, counter, header, ann_upd, l_md, l_dd = _load_image_state(controller, ds, new_entries, idx)
            return new_entries, new_idx, counter, header, ann_upd, l_md, l_dd, _format_classes_md(controller, ds), _gallery_items(new_entries), f"✅ {msg}"
        except Exception as exc:
            return entries, idx, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), f"❌ {exc}"

    def on_del_layer(ds, split, mode, entries, idx, value, layer_idx):
        if not ds or not entries or not value or layer_idx is None:
            return entries, idx, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "⚠️ กรุณาเลือกกรอบที่จะลบ"
        boxes = list(value.get("boxes") or [])
        li = int(layer_idx)
        if 0 <= li < len(boxes):
            boxes.pop(li)
        new_val = {"image": value.get("image"), "boxes": boxes}
        return on_save(ds, split, mode, entries, idx, new_val)

    def on_add_class(ds, new_name, entries, idx):
        if not ds:
            return "", _format_classes_md(controller, None), gr.update(), "⚠️ กรุณาเลือก Dataset ก่อน"
        try:
            controller.annotation_service.add_class(ds, new_name)
            _, _, _, ann_upd, _, _ = _load_image_state(controller, ds, entries, idx)
            return "", _format_classes_md(controller, ds), ann_upd, f"✅ เพิ่มคลาส `{new_name.strip()}` แล้ว"
        except Exception as exc:
            return new_name, _format_classes_md(controller, ds), gr.update(), f"❌ {exc}"

    def on_ai_single(ds, split, mode, entries, idx, model_choice, uploaded_pt, conf, auto_add):
        if not ds or not entries:
            return entries, idx, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "⚠️ ไม่มีภาพ"
        e = entries[max(0, min(idx, len(entries) - 1))]
        weights = uploaded_pt if uploaded_pt else model_choice
        try:
            res = controller.annotation_service.run_autolabel_single(
                dataset_id=ds,
                split=e["split"],
                key=e["key"],
                weights_path=weights,
                conf=float(conf),
                auto_add_classes=bool(auto_add),
            )
            new_entries = _list_entries(controller, ds, split, mode)
            new_idx, counter, header, ann_upd, l_md, l_dd = _load_image_state(controller, ds, new_entries, idx)
            msg = f"✨ พบ **{res['box_count']} กรอบ** ในภาพนี้ — แก้ไข/ลบกรอบบนภาพต่อได้เลย"
            return new_entries, new_idx, counter, header, ann_upd, l_md, l_dd, _format_classes_md(controller, ds), _gallery_items(new_entries), msg
        except Exception as exc:
            return entries, idx, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), f"❌ {exc}"

    def on_ai_batch(ds, split, model_choice, uploaded_pt, conf, scope, auto_add, allow_overwrite, idx):
        if not ds:
            return [], 0, "0 / 0", "⚠️ กรุณาเลือก Dataset", gr.update(), "", gr.update(), "", [], "⚠️ กรุณาเลือก Dataset"
        weights = uploaded_pt if uploaded_pt else model_choice
        try:
            summary = controller.annotation_service.run_autolabel(
                dataset_id=ds,
                weights_path=weights,
                conf=float(conf),
                iou=0.45,
                imgsz=640,
                scope=scope,
                split_filter=split,
                auto_add_classes=bool(auto_add),
                apply_immediately=True,
                allow_overwrite=bool(allow_overwrite),
            )
            new_entries = _list_entries(controller, ds, split, "all")
            new_idx, counter, header, ann_upd, l_md, l_dd = _load_image_state(controller, ds, new_entries, idx)
            msg = (
                f"✅ **ตีกรอบทั้งชุดเสร็จแล้ว!** สแกน {summary['images_scanned']} ภาพ · "
                f"เขียน Label {summary.get('written', 0)} ภาพ ({summary['total_boxes']} กรอบ) "
                "— ไล่ตรวจ/แก้กรอบในหน้านี้ได้เลย"
            )
            return new_entries, new_idx, counter, header, ann_upd, l_md, l_dd, _format_classes_md(controller, ds), _gallery_items(new_entries), msg
        except Exception as exc:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), f"❌ {exc}"

    def on_rename(ds, split, mode, entries, idx, new_stem):
        if not ds or not entries:
            return entries, idx, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "", "⚠️ ไม่มีภาพ"
        e = entries[max(0, min(idx, len(entries) - 1))]
        try:
            new_key = controller.annotation_service.rename_image(ds, e["split"], e["key"], new_stem)
            new_entries = _list_entries(controller, ds, split, mode)
            new_idx, counter, header, ann_upd, l_md, l_dd = _load_image_state(controller, ds, new_entries, idx)
            return new_entries, new_idx, counter, header, ann_upd, l_md, l_dd, _gallery_items(new_entries), "", f"✅ เปลี่ยนชื่อเป็น `{new_key}` แล้ว"
        except Exception as exc:
            return entries, idx, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), new_stem, f"❌ {exc}"

    def on_delete(ds, split, mode, entries, idx, confirm):
        if not ds or not entries:
            return entries, idx, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), False, "⚠️ ไม่มีภาพ"
        e = entries[max(0, min(idx, len(entries) - 1))]
        try:
            controller.annotation_service.delete_image(ds, e["split"], e["key"], confirm)
            new_entries = _list_entries(controller, ds, split, mode)
            new_idx, counter, header, ann_upd, l_md, l_dd = _load_image_state(controller, ds, new_entries, idx)
            return new_entries, new_idx, counter, header, ann_upd, l_md, l_dd, _gallery_items(new_entries), False, f"🗑️ ลบ `{e['key']}` แล้ว"
        except Exception as exc:
            return entries, idx, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), confirm, f"❌ {exc}"

    # ---- wiring
    reload_outputs = [
        entries_state, index_state, counter_md, img_info_md, annotator,
        layers_md, layer_dd, classes_md, thumb_gallery, model_dd,
    ]
    shared_ds_dd.change(fn=_reload_all, inputs=[shared_ds_dd, split_rd, mode_rd], outputs=reload_outputs)
    split_rd.change(fn=_reload_all, inputs=[shared_ds_dd, split_rd, mode_rd], outputs=reload_outputs)
    mode_rd.change(fn=_reload_all, inputs=[shared_ds_dd, split_rd, mode_rd], outputs=reload_outputs)

    nav_outputs = [
        entries_state, index_state, counter_md, img_info_md, annotator,
        layers_md, layer_dd, classes_md, thumb_gallery, save_msg,
    ]
    btn_prev.click(fn=on_prev, inputs=[shared_ds_dd, split_rd, mode_rd, entries_state, index_state, annotator], outputs=nav_outputs)
    btn_next.click(fn=on_next, inputs=[shared_ds_dd, split_rd, mode_rd, entries_state, index_state, annotator], outputs=nav_outputs)
    thumb_gallery.select(fn=on_gallery_select, inputs=[shared_ds_dd, split_rd, mode_rd, entries_state, index_state, annotator], outputs=nav_outputs)

    btn_save.click(fn=on_save, inputs=[shared_ds_dd, split_rd, mode_rd, entries_state, index_state, annotator], outputs=nav_outputs)
    btn_null.click(fn=on_mark_null, inputs=[shared_ds_dd, split_rd, mode_rd, entries_state, index_state], outputs=nav_outputs)
    btn_undo.click(fn=on_undo, inputs=[shared_ds_dd, split_rd, mode_rd, entries_state, index_state], outputs=nav_outputs)
    btn_del_layer.click(fn=on_del_layer, inputs=[shared_ds_dd, split_rd, mode_rd, entries_state, index_state, annotator, layer_dd], outputs=nav_outputs)

    btn_add_class.click(
        fn=on_add_class,
        inputs=[shared_ds_dd, txt_new_class, entries_state, index_state],
        outputs=[txt_new_class, classes_md, annotator, save_msg],
    )

    ai_outputs = [
        entries_state, index_state, counter_md, img_info_md, annotator,
        layers_md, layer_dd, classes_md, thumb_gallery, auto_msg,
    ]
    btn_ai_single.click(
        fn=on_ai_single,
        inputs=[shared_ds_dd, split_rd, mode_rd, entries_state, index_state, model_dd, custom_pt, conf_sl, chk_auto_add_cls],
        outputs=ai_outputs,
    )
    btn_ai_batch.click(
        fn=on_ai_batch,
        inputs=[shared_ds_dd, split_rd, model_dd, custom_pt, conf_sl, auto_scope, chk_auto_add_cls, chk_auto_overwrite, index_state],
        outputs=ai_outputs,
    )

    file_outputs = [
        entries_state, index_state, counter_md, img_info_md, annotator,
        layers_md, layer_dd, thumb_gallery, txt_rename, file_msg,
    ]
    btn_rename.click(fn=on_rename, inputs=[shared_ds_dd, split_rd, mode_rd, entries_state, index_state, txt_rename], outputs=file_outputs)

    del_outputs = [
        entries_state, index_state, counter_md, img_info_md, annotator,
        layers_md, layer_dd, thumb_gallery, chk_del, file_msg,
    ]
    btn_del.click(fn=on_delete, inputs=[shared_ds_dd, split_rd, mode_rd, entries_state, index_state, chk_del], outputs=del_outputs)

    return {"reload_fn": _reload_all, "split_rd": split_rd, "mode_rd": mode_rd}
