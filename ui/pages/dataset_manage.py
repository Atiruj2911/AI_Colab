"""Dataset > จัดการ Dataset tab (Phase 2): label table editor, AI auto-label, diagnostics, split, classes, import/export."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import gradio as gr

from application.app_controller import AppController
from core.dataset.entries import SPLITS
from ui.components.common import render_planned_notice
from utils.validation import format_bytes

EDITOR_HEADERS = ["คลาส", "x1", "y1", "x2", "y2"]
EDITOR_DATATYPES = ["str", "number", "number", "number", "number"]
CLASS_HEADERS = ["id", "ชื่อคลาส", "จำนวน Box"]
MODE_CHOICES = [("ทั้งหมด", "all"), ("ยังไม่มี Label", "unlabeled"), ("มี Label แล้ว", "labeled")]


def _ds_choices(controller: AppController) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for ds in controller.list_datasets():
        if ds.total_images <= 0:
            continue
        icon = "✅" if ds.is_valid else "⚠️"
        choices.append((f"{icon} {ds.dataset_name} — {ds.total_images} ภาพ [{ds.dataset_id}]", ds.dataset_id))
    return choices


def _weight_choices(controller: AppController) -> list[tuple[str, str]]:
    return [(f"{c.name} ({format_bytes(c.size_bytes)})", c.path) for c in controller.list_checkpoints()]


def _gallery(controller: AppController, ds: str | None, split: str, mode: str):
    if not ds:
        return [], []
    rows = controller.annotation_service.list_images(ds, split, mode)
    items = []
    keys = []
    for r in rows:
        status = f"มี Label · {r['box_count']} กรอบ" if r["has_label"] else "ยังไม่มี Label"
        items.append((str(r["image_path"]), f"{r['key']} [{status}]"))
        keys.append((r["split"], r["key"]))
    return items, keys


def _editor_outputs(controller: AppController, ds: str, split: str, key: str):
    data = controller.annotation_service.load_editor(ds, split, key)
    md = (
        f"**{split}/{key}** · ขนาด {data['width']}×{data['height']} px · "
        f"{len(data['rows'])} กรอบ · {'มี Label' if data['has_label'] else 'ยังไม่มี Label'}"
    )
    if data["problems"]:
        md += "\n\n⚠️ พบบรรทัด label ที่อ่านไม่ได้ (จะถูกละเว้นเมื่อบันทึกใหม่): " + "; ".join(data["problems"][:3])
    return {"ds": ds, "split": split, "key": key}, md, data["preview"], data["rows"]


def _class_rows(controller: AppController, ds: str | None) -> list[list[Any]]:
    if not ds:
        return []
    return [[c["id"], c["name"], c["count"]] for c in controller.annotation_service.list_classes(ds)]


def _diag_markdown(d: dict[str, Any]) -> str:
    def _listing(items: list[str], limit: int = 15) -> str:
        if not items:
            return "_ไม่พบ_"
        shown = "\n".join(f"- `{html.escape(str(i))}`" for i in items[:limit])
        more = f"\n- ... และอีก {len(items) - limit} รายการ" if len(items) > limit else ""
        return shown + more

    lines = [f"### ผลการวินิจฉัย Dataset ({d['total_images']} ภาพ)", ""]
    lines.append(f"**ภาพซ้ำ (เนื้อไฟล์เหมือนกัน):** {len(d['duplicates'])} กลุ่ม")
    for group in d["duplicates"][:10]:
        lines.append("- " + " = ".join(f"`{g}`" for g in group))
    lines.append(f"\n**ภาพเสีย/เปิดไม่ได้:** {len(d['corrupt'])} ภาพ\n{_listing(d['corrupt'])}")
    lines.append(f"\n**ภาพที่ยังไม่มี Label:** {len(d['missing_labels'])} ภาพ\n{_listing(d['missing_labels'])}")
    lines.append(f"\n**ไฟล์ Label กำพร้า (ไม่มีภาพคู่):** {len(d['orphan_labels'])} ไฟล์\n{_listing(d['orphan_labels'])}")
    lines.append(
        f"\n**Label รูปแบบผิด:** {len(d['invalid_labels'])} ไฟล์"
        + "".join(f"\n- `{a}` — {b}" for a, b in d["invalid_labels"][:10])
    )
    lines.append(f"\n**กรอบที่ class_id ไม่อยู่ใน data.yaml:** {d['invalid_class_boxes']} กรอบ")
    lines.append(f"**Label ว่าง (ภาพ background):** {d['empty_labels']} ไฟล์")
    lines.append(f"**คลาสที่ไม่มีตัวอย่างเลย:** {', '.join(d['unused_classes']) or 'ไม่มี'}")
    lines.append("\n**การกระจายคลาส (จำนวนกรอบ) ต่อ Split:**")
    if d["class_counts"]:
        lines.append("")
        lines.append("| Split | คลาส | จำนวนกรอบ |")
        lines.append("|---|---|---:|")
        for split in SPLITS:
            for name, n in sorted(d["class_counts"].get(split, {}).items()):
                lines.append(f"| {split} | {html.escape(name)} | {n:,} |")
    else:
        lines.append("_ยังไม่มี Box_")
    return "\n".join(lines)


def build_manage_tab(controller: AppController) -> dict[str, Any]:
    svc = controller.annotation_service
    ds_choices = _ds_choices(controller)
    default_ds = ds_choices[0][1] if ds_choices else None
    init_items, init_keys = _gallery(controller, default_ds, "train", "all")

    gr.HTML(
        render_planned_notice(
            "Phase 2 — Dataset & Annotation (MVP: ตารางแก้ไขกรอบ)",
            "ตัวแก้ไขแบบลากวาด/ย้าย/ปรับขนาดกรอบบนภาพยังไม่มี — ตอนนี้แก้พิกัดและคลาสผ่านตาราง "
            "ทุกการเขียน Label มีการสำรองไว้ให้ย้อนกลับได้ 1 ครั้ง",
        )
    )

    with gr.Row():
        ds_dd = gr.Dropdown(label="Dataset", choices=ds_choices, value=default_ds, scale=5)
        split_rd = gr.Radio(choices=list(SPLITS), value="train", label="Split", scale=2)
        mode_rd = gr.Radio(choices=MODE_CHOICES, value="all", label="กรองภาพ", scale=3)

    current = gr.State(None)
    gallery_keys = gr.State(init_keys)

    with gr.Tabs():
        # ------------------------------------------------------------ editor
        with gr.Tab("1) แก้ไข Label"):
            with gr.Row():
                with gr.Column(scale=4):
                    gallery = gr.Gallery(
                        label="ภาพใน Split (คลิกเพื่อเลือกภาพ)",
                        value=init_items,
                        columns=4,
                        height=430,
                        allow_preview=False,
                        object_fit="contain",
                    )
                with gr.Column(scale=6):
                    selected_md = gr.Markdown("เลือกภาพจากรายการด้านซ้าย")
                    editor_img = gr.Image(label="ภาพพร้อมกรอบ", type="pil", height=300, interactive=False)
                    boxes_df = gr.Dataframe(
                        headers=EDITOR_HEADERS,
                        datatype=EDITOR_DATATYPES,
                        type="array",
                        row_count=0,
                        column_count=5,
                        interactive=True,
                        label="กรอบ Bounding Box (พิกัดเป็น pixel ของภาพต้นฉบับ: x1,y1 มุมซ้ายบน / x2,y2 มุมขวาล่าง)",
                    )
                    overwrite_chk = gr.Checkbox(label="ยืนยันการเขียนทับ Label เดิม", value=False)
                    with gr.Row():
                        btn_save = gr.Button("💾 บันทึก Label", variant="primary")
                        btn_undo = gr.Button("↩️ ย้อนกลับการบันทึกล่าสุด", variant="secondary")
                    editor_msg = gr.Markdown("")

            with gr.Accordion("เปลี่ยนชื่อ / ลบภาพนี้", open=False):
                with gr.Row():
                    txt_new_name = gr.Textbox(label="ชื่อใหม่ (ไม่ใส่นามสกุล)", scale=4)
                    btn_rename = gr.Button("เปลี่ยนชื่อ", scale=1)
                with gr.Row():
                    chk_delete = gr.Checkbox(label="ยืนยันการลบภาพและ Label นี้ (ย้อนกลับไม่ได้)", scale=4)
                    btn_delete = gr.Button("🗑️ ลบภาพนี้", variant="stop", scale=1)
                manage_msg = gr.Markdown("")

        # ------------------------------------------------------------ AI auto-label
        with gr.Tab("2) AI Auto-Label"):
            gr.Markdown(
                "ใช้โมเดลที่เทรนแล้วเสนอกรอบให้ภาพ **ระบบยังไม่เขียน Label จนกว่าคุณจะกดยอมรับ** "
                "และชื่อคลาสในโมเดลต้องตรงกับชื่อคลาสใน Dataset"
            )
            with gr.Row():
                with gr.Column(scale=5):
                    weights_dd = gr.Dropdown(label="โมเดลสำหรับทำนาย (.pt)", choices=_weight_choices(controller))
                    btn_refresh_weights = gr.Button("🔄 รีเฟรชรายการโมเดล", size="sm")
                    conf_sl = gr.Slider(label="Confidence", minimum=0.05, maximum=0.95, step=0.01, value=0.25)
                    iou_sl = gr.Slider(label="IoU (NMS)", minimum=0.05, maximum=0.95, step=0.01, value=0.45)
                    imgsz_dd = gr.Dropdown(label="Image Size", choices=[320, 416, 512, 640, 768, 960, 1280], value=640)
                    scope_rd = gr.Radio(
                        label="ขอบเขต",
                        choices=[
                            ("เฉพาะภาพที่ยังไม่มี Label (ปลอดภัย)", "unlabeled"),
                            ("ทุกภาพ (จะเขียนทับ Label เดิมเมื่อยืนยัน)", "all"),
                        ],
                        value="unlabeled",
                    )
                    auto_split_dd = gr.Dropdown(label="เฉพาะ Split", choices=["all", *SPLITS], value="all")
                    btn_predict = gr.Button("▶ ทำนายและแสดงผลเสนอ", variant="primary")
                with gr.Column(scale=7):
                    auto_md = gr.Markdown("ยังไม่มีผลเสนอ")
                    proposal_gallery = gr.Gallery(label="ตัวอย่างผลเสนอ (กรอบสีคือผลที่โมเดลทำนาย)", columns=3, height=400, allow_preview=True)
                    chk_accept_overwrite = gr.Checkbox(
                        label="ยืนยันการเขียนทับ Label เดิม (ใช้เฉพาะเมื่อเลือกขอบเขต 'ทุกภาพ')", value=False
                    )
                    with gr.Row():
                        btn_accept = gr.Button("✅ ยอมรับผลเสนอ", variant="primary")
                        btn_discard = gr.Button("ทิ้งผลเสนอ", variant="secondary")

        # ------------------------------------------------------------ diagnostics & split
        with gr.Tab("3) วินิจฉัย & แบ่ง Dataset"):
            with gr.Row():
                with gr.Column(scale=6):
                    btn_diag = gr.Button("🔍 ตรวจวินิจฉัย Dataset นี้", variant="primary")
                    diag_md = gr.Markdown("กดปุ่มเพื่อตรวจ: ภาพซ้ำ, ภาพเสีย, Label ขาด/กำพร้า/ผิดรูปแบบ, การกระจายคลาส")
                with gr.Column(scale=6):
                    gr.Markdown(
                        "#### แบ่ง Train / Val / Test ด้วย Seed\n"
                        "ระบบจะ**สร้าง Dataset ใหม่**ทุกครั้ง (ไม่แก้ไข Dataset เดิม) "
                        "ควรแบ่งก่อนทำ Augmentation เสมอเพื่อป้องกัน data leakage"
                    )
                    with gr.Row():
                        train_pct = gr.Number(label="Train %", value=80, minimum=0, maximum=100)
                        val_pct = gr.Number(label="Val %", value=20, minimum=0, maximum=100)
                        test_pct = gr.Number(label="Test %", value=0, minimum=0, maximum=100)
                    with gr.Row():
                        seed_num = gr.Number(label="Random Seed", value=42, precision=0)
                        new_split_name = gr.Textbox(label="ชื่อ Dataset ใหม่ (ไม่บังคับ)")
                    chk_drop_dup = gr.Checkbox(label="ตัดภาพซ้ำออก (เก็บภาพแรกของแต่ละกลุ่ม)", value=True)
                    btn_split = gr.Button("✂️ สร้าง Dataset ที่แบ่งแล้ว", variant="primary")
                    split_md = gr.Markdown("")

        # ------------------------------------------------------------ classes
        with gr.Tab("4) คลาส"):
            gr.Markdown("แก้ชื่อคลาสในตารางแล้วกด **บันทึกชื่อคลาส** ระบบจะอัปเดต data.yaml ให้ (id ของคลาสไม่เปลี่ยน)")
            classes_df = gr.Dataframe(
                headers=CLASS_HEADERS,
                datatype=["number", "str", "number"],
                type="array",
                column_count=3,
                row_count=0,
                interactive=True,
                value=_class_rows(controller, default_ds),
                label="คลาสใน Dataset",
            )
            with gr.Row():
                btn_save_names = gr.Button("💾 บันทึกชื่อคลาส", variant="primary")
            with gr.Row():
                txt_new_class = gr.Textbox(label="เพิ่มคลาสใหม่", scale=4)
                btn_add_class = gr.Button("➕ เพิ่มคลาส", scale=1)
            with gr.Row():
                num_remove_id = gr.Number(label="id ที่จะลบ (ได้เฉพาะคลาสที่ id สูงสุดและไม่มี Box)", precision=0, scale=4)
                btn_remove_class = gr.Button("🗑️ ลบคลาส", variant="stop", scale=1)
            class_msg = gr.Markdown("")

        # ------------------------------------------------------------ import / export
        with gr.Tab("5) Export / Import Label"):
            gr.Markdown(
                "**Export:** ได้ ZIP ที่มี `data.yaml`, `classes.txt` และ `labels/<split>/...`\n\n"
                "**Import:** รับ ZIP รูปแบบเดียวกัน จับคู่ตามชื่อไฟล์ภาพ และไม่เขียนทับ Label เดิมถ้าไม่ยืนยัน"
            )
            with gr.Row():
                with gr.Column():
                    btn_export = gr.Button("📦 Export Label เป็น ZIP", variant="primary")
                    export_file = gr.File(label="ไฟล์ Export", interactive=False)
                    export_msg = gr.Markdown("")
                with gr.Column():
                    import_file = gr.File(label="เลือก ZIP ของ Label ที่ต้องการนำเข้า", file_types=[".zip"], type="filepath")
                    chk_import_overwrite = gr.Checkbox(label="ยืนยันการเขียนทับ Label เดิม", value=False)
                    btn_import = gr.Button("📥 Import Label", variant="secondary")
                    import_msg = gr.Markdown("")

    # ================================================================ handlers
    gallery_outputs = [gallery, gallery_keys]
    editor_outputs = [current, selected_md, editor_img, boxes_df, editor_msg]

    def _refresh_choices():
        return gr.update(choices=_ds_choices(controller)), gr.update(choices=_weight_choices(controller))

    def on_ds_or_filter_change(ds, split, mode):
        items, keys = _gallery(controller, ds, split, mode)
        return items, keys, None, "เลือกภาพจากรายการด้านซ้าย", None, [], "", _class_rows(controller, ds)

    def on_select(ds, keys, evt: gr.SelectData):
        if not ds or not keys:
            return None, "", None, [], "ไม่พบรายการภาพ"
        split, key = keys[evt.index]
        try:
            return (*_editor_outputs(controller, ds, split, key), "")
        except Exception as exc:
            return None, "", None, [], f"❌ เปิดภาพไม่สำเร็จ: {exc}"

    def on_save(cur, rows, overwrite, ds, split, mode):
        if not cur:
            return None, [], "⚠️ กรุณาเลือกภาพก่อนบันทึก", *_gallery(controller, ds, split, mode)
        try:
            res = controller.annotation_service.save_boxes(cur["ds"], cur["split"], cur["key"], rows, overwrite)
            _, _, preview, new_rows = _editor_outputs(controller, cur["ds"], cur["split"], cur["key"])
            msg = f"✅ บันทึกแล้ว · {res['box_count']} กรอบ" + (" (ไม่มีการเปลี่ยนแปลง)" if not res["changed"] else "")
            return preview, new_rows, msg, *_gallery(controller, ds, split, mode)
        except Exception as exc:
            return gr.update(), gr.update(), f"❌ บันทึกไม่สำเร็จ: {exc}", gr.update(), gr.update()

    def on_undo(cur, ds, split, mode):
        if not cur:
            return gr.update(), gr.update(), "⚠️ กรุณาเลือกภาพก่อน", gr.update(), gr.update()
        try:
            msg = controller.annotation_service.undo_last_save(cur["ds"], cur["split"], cur["key"])
            _, _, preview, rows = _editor_outputs(controller, cur["ds"], cur["split"], cur["key"])
            return preview, rows, f"✅ {msg}", *_gallery(controller, ds, split, mode)
        except Exception as exc:
            return gr.update(), gr.update(), f"❌ {exc}", gr.update(), gr.update()

    # rename/delete outputs: current, selected_md, editor_img, boxes_df, editor_msg, <extra>, manage_msg, gallery, keys
    def _manage_error(msg: str):
        return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), msg, gr.update(), gr.update())

    def on_rename(cur, new_name, ds, split, mode):
        if not cur:
            return _manage_error("⚠️ กรุณาเลือกภาพก่อน")
        try:
            new_key = controller.annotation_service.rename_image(cur["ds"], cur["split"], cur["key"], new_name)
            new_cur, md, preview, rows = _editor_outputs(controller, cur["ds"], cur["split"], new_key)
            return new_cur, md, preview, rows, "", "", f"✅ เปลี่ยนชื่อเป็น `{new_key}` แล้ว", *_gallery(controller, ds, split, mode)
        except Exception as exc:
            return _manage_error(f"❌ {exc}")

    def on_delete(cur, confirm, ds, split, mode):
        if not cur:
            return _manage_error("⚠️ กรุณาเลือกภาพก่อน")
        try:
            controller.annotation_service.delete_image(cur["ds"], cur["split"], cur["key"], confirm)
            items, keys = _gallery(controller, ds, split, mode)
            return None, "ลบภาพแล้ว เลือกภาพใหม่จากรายการ", None, [], "", False, f"🗑️ ลบ `{cur['split']}/{cur['key']}` แล้ว", items, keys
        except Exception as exc:
            return _manage_error(f"❌ {exc}")

    def on_predict(ds, weights, conf, iou, imgsz, scope, split_filter):
        if not ds:
            return "⚠️ กรุณาเลือก Dataset", []
        try:
            summary = controller.annotation_service.run_autolabel(
                dataset_id=ds,
                weights_path=weights,
                conf=float(conf),
                iou=float(iou),
                imgsz=int(imgsz),
                scope=scope,
                split_filter=split_filter,
            )
            previews = controller.annotation_service.build_proposal_previews(ds)
            md = (
                f"**ผลเสนอ:** ภาพที่สแกน {summary['images_scanned']} · มีผลตรวจจับ {summary['images_with_detections']} ภาพ · "
                f"รวม {summary['total_boxes']} กรอบ"
                + (f" · ข้าม {summary['unknown_class_boxes']} กรอบที่ชื่อคลาสไม่ตรงกับ Dataset" if summary["unknown_class_boxes"] else "")
                + "\n\nยังไม่ได้เขียน Label — กด **ยอมรับผลเสนอ** เพื่อบันทึก"
            )
            return md, previews
        except Exception as exc:
            return f"❌ Auto-Label ไม่สำเร็จ: {exc}", []

    def on_accept(ds, overwrite, split, mode):
        if not ds:
            return "⚠️ กรุณาเลือก Dataset", [], gr.update(), gr.update()
        try:
            res = controller.annotation_service.accept_proposals(ds, overwrite)
            md = (
                f"✅ เขียน Label แล้ว {res['written']} ภาพ · ข้าม (มี Label เดิม) {res['skipped_existing']} · "
                f"ข้าม (ไม่พบกรอบ) {res['skipped_no_detection']}"
            )
            return md, [], *_gallery(controller, ds, split, mode)
        except Exception as exc:
            return f"❌ {exc}", gr.update(), gr.update(), gr.update()

    def on_discard(ds):
        if ds:
            controller.annotation_service.discard_proposals(ds)
        return "ทิ้งผลเสนอแล้ว", []

    def on_diag(ds):
        if not ds:
            return "⚠️ กรุณาเลือก Dataset"
        try:
            return _diag_markdown(controller.annotation_service.diagnostics(ds))
        except Exception as exc:
            return f"❌ วินิจฉัยไม่สำเร็จ: {exc}"

    def on_split(ds, tp, vp, sp, seed, drop_dup, name):
        if not ds:
            return "⚠️ กรุณาเลือก Dataset", gr.update()
        try:
            new_report, counts = controller.annotation_service.create_split_dataset(
                dataset_id=ds,
                train_pct=float(tp or 0),
                val_pct=float(vp or 0),
                test_pct=float(sp or 0),
                seed=int(seed or 0),
                drop_duplicates=bool(drop_dup),
                new_name=name or None,
            )
            md = (
                f"✅ สร้าง Dataset ใหม่ **{new_report.dataset_name}** `[{new_report.dataset_id}]`\n\n"
                f"- Train {counts['train']} · Val {counts['val']} · Test {counts['test']} ภาพ\n"
                f"- ภาพ background (ไม่มี Label): {counts['background']}\n"
                f"- ตัดออก: ภาพเสีย {counts['excluded_corrupt']} · ภาพซ้ำ {counts['excluded_duplicate']}\n"
                f"- ผลตรวจสอบ: {'ผ่าน' if new_report.is_valid else 'มี error — ดูที่แท็บตรวจสอบ Label'}"
            )
            return md, gr.update(choices=_ds_choices(controller), value=new_report.dataset_id)
        except Exception as exc:
            return f"❌ แบ่ง Dataset ไม่สำเร็จ: {exc}", gr.update()

    def on_save_names(ds, rows):
        if not ds:
            return _class_rows(controller, None), "⚠️ กรุณาเลือก Dataset"
        try:
            new_names = {int(r[0]): str(r[1]) for r in (rows or []) if r and r[0] not in (None, "")}
            changed = controller.annotation_service.rename_classes(ds, new_names)
            return _class_rows(controller, ds), f"✅ บันทึกแล้ว ({changed} คลาสเปลี่ยนชื่อ)"
        except Exception as exc:
            return _class_rows(controller, ds), f"❌ {exc}"

    def on_add_class(ds, name):
        try:
            controller.annotation_service.add_class(ds, name)
            return _class_rows(controller, ds), "✅ เพิ่มคลาสแล้ว", ""
        except Exception as exc:
            return _class_rows(controller, ds), f"❌ {exc}", name

    def on_remove_class(ds, cid):
        if cid is None:
            return _class_rows(controller, ds), "⚠️ กรุณาระบุ id ของคลาส"
        try:
            controller.annotation_service.remove_last_class(ds, int(cid))
            return _class_rows(controller, ds), "✅ ลบคลาสแล้ว"
        except Exception as exc:
            return _class_rows(controller, ds), f"❌ {exc}"

    def on_export(ds):
        if not ds:
            return None, "⚠️ กรุณาเลือก Dataset"
        try:
            path = controller.annotation_service.export_labels(ds)
            return str(path), f"✅ Export แล้ว: `{Path(path).name}`"
        except Exception as exc:
            return None, f"❌ {exc}"

    def on_import(ds, zip_path, overwrite, split, mode):
        if not ds:
            return "⚠️ กรุณาเลือก Dataset", gr.update(), gr.update()
        try:
            res = controller.annotation_service.import_labels(ds, zip_path, overwrite)
            md = (
                f"✅ นำเข้า {res['imported']} ไฟล์ · เนื้อหาเหมือนเดิม {res['unchanged']} · ข้าม (มี Label เดิมต่างกัน) {res['skipped_existing']} · "
                f"ไม่ถูกต้อง {res['invalid']} · ไม่พบภาพคู่ {res['unmatched']}"
            )
            return md, *_gallery(controller, ds, split, mode)
        except Exception as exc:
            return f"❌ {exc}", gr.update(), gr.update()

    # ---- wiring
    ds_dd.focus(fn=_refresh_choices, outputs=[ds_dd, weights_dd])
    btn_refresh_weights.click(fn=_refresh_choices, outputs=[ds_dd, weights_dd])

    filter_inputs = [ds_dd, split_rd, mode_rd]
    filter_outputs = [gallery, gallery_keys, current, selected_md, editor_img, boxes_df, editor_msg, classes_df]
    ds_dd.change(fn=on_ds_or_filter_change, inputs=filter_inputs, outputs=filter_outputs)
    split_rd.change(fn=on_ds_or_filter_change, inputs=filter_inputs, outputs=filter_outputs)
    mode_rd.change(fn=on_ds_or_filter_change, inputs=filter_inputs, outputs=filter_outputs)

    gallery.select(fn=on_select, inputs=[ds_dd, gallery_keys], outputs=editor_outputs)

    btn_save.click(
        fn=on_save,
        inputs=[current, boxes_df, overwrite_chk, ds_dd, split_rd, mode_rd],
        outputs=[editor_img, boxes_df, editor_msg, *gallery_outputs],
    )
    btn_undo.click(
        fn=on_undo,
        inputs=[current, ds_dd, split_rd, mode_rd],
        outputs=[editor_img, boxes_df, editor_msg, *gallery_outputs],
    )
    btn_rename.click(
        fn=on_rename,
        inputs=[current, txt_new_name, ds_dd, split_rd, mode_rd],
        outputs=[current, selected_md, editor_img, boxes_df, editor_msg, txt_new_name, manage_msg, *gallery_outputs],
    )
    btn_delete.click(
        fn=on_delete,
        inputs=[current, chk_delete, ds_dd, split_rd, mode_rd],
        outputs=[current, selected_md, editor_img, boxes_df, editor_msg, chk_delete, manage_msg, *gallery_outputs],
    )

    btn_predict.click(
        fn=on_predict,
        inputs=[ds_dd, weights_dd, conf_sl, iou_sl, imgsz_dd, scope_rd, auto_split_dd],
        outputs=[auto_md, proposal_gallery],
    )
    btn_accept.click(
        fn=on_accept,
        inputs=[ds_dd, chk_accept_overwrite, split_rd, mode_rd],
        outputs=[auto_md, proposal_gallery, *gallery_outputs],
    )
    btn_discard.click(fn=on_discard, inputs=[ds_dd], outputs=[auto_md, proposal_gallery])

    btn_diag.click(fn=on_diag, inputs=[ds_dd], outputs=[diag_md])
    btn_split.click(
        fn=on_split,
        inputs=[ds_dd, train_pct, val_pct, test_pct, seed_num, chk_drop_dup, new_split_name],
        outputs=[split_md, ds_dd],
    )

    btn_save_names.click(fn=on_save_names, inputs=[ds_dd, classes_df], outputs=[classes_df, class_msg])
    btn_add_class.click(fn=on_add_class, inputs=[ds_dd, txt_new_class], outputs=[classes_df, class_msg, txt_new_class])
    btn_remove_class.click(fn=on_remove_class, inputs=[ds_dd, num_remove_id], outputs=[classes_df, class_msg])

    btn_export.click(fn=on_export, inputs=[ds_dd], outputs=[export_file, export_msg])
    btn_import.click(
        fn=on_import,
        inputs=[ds_dd, import_file, chk_import_overwrite, split_rd, mode_rd],
        outputs=[import_msg, *gallery_outputs],
    )

    return {"ds_dd": ds_dd, "split_rd": split_rd, "mode_rd": mode_rd}
