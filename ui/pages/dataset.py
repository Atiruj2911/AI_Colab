"""Page 2: Dataset — clean 3-step flow (Import -> Annotate & Auto-Label -> Validate & Split)."""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr

from application.app_controller import AppController
from core.dataset.entries import SPLITS
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
        status = "Ready" if ds.is_valid else "Draft"
        label = f"{ds.dataset_name} ({ds.total_images} images, {len(ds.class_names)} classes — {status})"
        choices.append((label, ds.dataset_id))
    return choices


def _split_counts(ds: DatasetValidationReport) -> str:
    parts = []
    for name in ("train", "val", "test"):
        stats = ds.splits.get(name)
        parts.append(f"{name[0].upper()} {stats.image_count if stats else 0}")
    return " / ".join(parts)


def _modified(ds: DatasetValidationReport) -> str:
    path = Path(ds.dataset_root)
    if not path.exists():
        return "—"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def _render_library(controller: AppController, query: str, status: str) -> str:
    needle = (query or "").strip().lower()
    rows: list[str] = []
    for ds in controller.list_datasets():
        if ds.total_images <= 0:
            continue
        state = "Ready" if ds.is_valid else "Draft"
        if status in {"Ready", "Draft"} and state != status:
            continue
        if needle and needle not in ds.dataset_name.lower():
            continue
        annotated = sum(sp.label_count for sp in ds.splits.values()) or ds.total_labels
        rows.append(
            "<tr>"
            f"<td>{html.escape(ds.dataset_name)}</td>"
            f"<td>{ds.total_images:,}</td>"
            f"<td>{annotated:,}</td>"
            f"<td>{len(ds.class_names)}</td>"
            f"<td>{html.escape(_split_counts(ds))}</td>"
            f"<td>{_modified(ds)}</td>"
            f"<td>{state}</td>"
            "</tr>"
        )
    body = "".join(rows) if rows else (
        '<tr><td colspan="7" class="empty-note">No datasets match this filter. Use Import Dataset to add images or a YOLO zip.</td></tr>'
    )
    return (
        '<div class="studio-card" style="padding:0;overflow:auto;">'
        '<table class="data-table"><thead><tr>'
        "<th>Dataset</th><th>Images</th><th>Annotated</th><th>Classes</th><th>Train / Val / Test</th><th>Modified</th><th>Status</th>"
        f"</tr></thead><tbody>{body}</tbody></table></div>"
        '<div class="empty-note" style="margin-top:8px;">Open, rename, class edits, and diagnostics use the dataset selected above. '
        "Duplicate and delete are not available.</div>"
    )


def _format_summary(report: DatasetValidationReport | None) -> tuple[str, str, str]:
    if report is None:
        return (
            render_dataset_summary_card(None),
            render_validation_issues_markdown(None),
            "ยังไม่มีข้อมูลโครงสร้างโฟลเดอร์",
        )
    return (
        render_dataset_summary_card(report),
        render_validation_issues_markdown(report),
        f"```text\n{report.structure_tree}\n```",
    )


def _class_rows(controller: AppController, ds: str | None) -> list[list[Any]]:
    if not ds:
        return []
    try:
        return [[c["id"], c["name"], c["count"]] for c in controller.annotation_service.list_classes(ds)]
    except Exception:
        return []


def _diag_markdown(d: dict[str, Any]) -> str:
    def _listing(items: list[str], limit: int = 12) -> str:
        if not items:
            return "_ไม่พบ_"
        shown = "\n".join(f"- `{html.escape(str(i))}`" for i in items[:limit])
        more = f"\n- ... และอีก {len(items) - limit} รายการ" if len(items) > limit else ""
        return shown + more

    lines = [
        "| Check | Count | Severity |",
        "|---|---:|---|",
        f"| Duplicate groups | {len(d['duplicates'])} | {'Warning' if d['duplicates'] else 'OK'} |",
        f"| Corrupted images | {len(d['corrupt'])} | {'Error' if d['corrupt'] else 'OK'} |",
        f"| Missing labels | {len(d['missing_labels'])} | {'Warning' if d['missing_labels'] else 'OK'} |",
        f"| Orphan label files | {len(d['orphan_labels'])} | {'Warning' if d['orphan_labels'] else 'OK'} |",
        f"| Invalid label files | {len(d['invalid_labels'])} | {'Error' if d['invalid_labels'] else 'OK'} |",
        f"| Classes with no boxes | {len(d['unused_classes'])} | {'Warning' if d['unused_classes'] else 'OK'} |",
        "",
        f"Images scanned: **{d['total_images']}**.",
    ]
    if d["unused_classes"]:
        lines.append("Unused classes: " + ", ".join(f"`{html.escape(name)}`" for name in d["unused_classes"]))
    if d["corrupt"]:
        lines.append(f"\n**Corrupted images:**\n{_listing(d['corrupt'])}")
    if d["class_counts"]:
        lines.append("\n| Split | คลาส | จำนวนกรอบ |\n|---|---|---:|")
        for split in SPLITS:
            for name, n in sorted(d["class_counts"].get(split, {}).items()):
                lines.append(f"| {split} | {html.escape(name)} | {n:,} |")
    return "\n".join(lines)


def _parse_class_csv(text: str) -> list[str]:
    if not text:
        return []
    return [c.strip() for c in text.replace("\n", ",").split(",") if c.strip()]


def build_dataset_page(controller: AppController) -> dict[str, Any]:
    initial_choices = _dataset_choices(controller)
    initial_id = initial_choices[0][1] if initial_choices else None
    initial_report = controller.get_dataset(initial_id) if initial_id else None
    init_summary, init_issues, init_tree = _format_summary(initial_report)

    with gr.Column(visible=False) as container:
        gr.HTML(
            render_page_header(
                "Datasets",
                "Review the library or import images and YOLO labels.",
            )
        )

        with gr.Row(elem_classes=["compact-row"]):
            dataset_dropdown = gr.Dropdown(
                label="Dataset",
                choices=initial_choices,
                value=initial_id,
                interactive=True,
                scale=6,
            )
            btn_open_annotate = gr.Button("Open", variant="primary", scale=1)

        with gr.Tabs(selected="tab_library" if initial_choices else "tab_import"):
            with gr.Tab("Dataset Library", id="tab_library"):
                with gr.Row(elem_classes=["compact-row"]):
                    library_query = gr.Textbox(placeholder="Search datasets", show_label=False, scale=5)
                    library_status = gr.Dropdown(
                        choices=["All", "Ready", "Draft"],
                        value="All",
                        label="Status",
                        scale=2,
                    )
                library_html = gr.HTML(value=_render_library(controller, "", "All"))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=6):
                        summary_html = gr.HTML(value=init_summary)
                        with gr.Row(elem_classes=["compact-row"]):
                            btn_revalidate = gr.Button("Re-validate", variant="secondary", size="sm", scale=2)
                            txt_rename_ds = gr.Textbox(placeholder="New dataset name", show_label=False, scale=3)
                            btn_rename_ds = gr.Button("Rename", size="sm", scale=1)
                        rename_ds_msg = gr.Markdown("")
                        issues_md = gr.Markdown(value=init_issues)
                    with gr.Column(scale=6):
                        gr.Markdown("**Classes**")
                        classes_df = gr.Dataframe(
                            headers=["ID", "Class Name", "Boxes"],
                            datatype=["number", "str", "number"],
                            type="array",
                            column_count=3,
                            row_count=0,
                            interactive=True,
                            value=_class_rows(controller, initial_id),
                        )
                        with gr.Row():
                            btn_save_names = gr.Button("Save Class Names", size="sm")
                            num_remove_id = gr.Number(label="Unused class ID", precision=0)
                            btn_remove_class = gr.Button("Delete Class", variant="stop", size="sm")
                        class_msg = gr.Markdown("")

                with gr.Accordion("Diagnostics and labels", open=False):
                    with gr.Row():
                        with gr.Column(scale=5):
                            btn_diag = gr.Button("Run Diagnostics", variant="secondary")
                            diag_md = gr.Markdown("")
                        with gr.Column(scale=7):
                            with gr.Row():
                                with gr.Column():
                                    btn_export_lbl = gr.Button("Export Labels (ZIP)", size="sm")
                                    export_lbl_file = gr.File(label="Exported Labels", interactive=False)
                                    export_lbl_msg = gr.Markdown("")
                                with gr.Column():
                                    import_lbl_file = gr.File(label="Import Labels (.zip)", file_types=[".zip"], type="filepath")
                                    chk_import_ow = gr.Checkbox(label="Overwrite existing labels", value=False)
                                    btn_import_lbl = gr.Button("Import Labels", size="sm")
                                    import_lbl_msg = gr.Markdown("")
                    with gr.Accordion("Dataset structure", open=False):
                        structure_md = gr.Markdown(value=init_tree, elem_classes=["tree-box"])

            with gr.Tab("Import Dataset", id="tab_import"):
                import_mode = gr.Radio(
                    label="Import Mode",
                    choices=[
                        ("Images Only (no labels yet)", "images_only"),
                        ("Images + YOLO Labels (.txt)", "with_labels"),
                    ],
                    value="images_only",
                )

                # ---- Mode 1 panel
                with gr.Column(visible=True) as mode1_box:
                    gr.Markdown("Upload image files or a `.zip` archive to start annotating.")
                    m1_files = gr.File(
                        label="Upload Images (.jpg, .png, .webp) or .zip",
                        file_count="multiple",
                        type="filepath",
                    )
                    with gr.Row():
                        m1_name = gr.Textbox(label="Dataset Name (optional)", placeholder="e.g. factory_parts_v1")
                        m1_classes = gr.Textbox(
                            label="Initial Classes (optional, comma-separated)",
                            placeholder="Leave blank to add classes while annotating",
                        )
                    btn_m1_create = gr.Button("Import Dataset", variant="primary")
                    m1_msg = gr.Markdown("")

                # ---- Mode 2 panel
                with gr.Column(visible=False) as mode2_box:
                    gr.Markdown(
                        "Upload a `.zip` or image + `.txt` files. If the folder structure is non-standard, you can map image and label folders below."
                    )
                    m2_files = gr.File(
                        label="Upload YOLO Dataset (.zip or files)",
                        file_count="multiple",
                        type="filepath",
                    )
                    m2_name = gr.Textbox(label="Dataset Name (optional)", placeholder="e.g. ppe_labeled_v1")
                    btn_m2_upload = gr.Button("Import Dataset", variant="primary")
                    m2_msg = gr.Markdown("")

                    # Custom folder mapping fallback (appears when standard YOLO format is not detected)
                    stage_state = gr.State(None)
                    with gr.Column(visible=False) as mapping_box:
                        gr.Markdown("### Folder Mapping")
                        with gr.Row():
                            map_img_dir = gr.Dropdown(label="Images Folder", choices=[])
                            map_lbl_dir = gr.Dropdown(label="Labels Folder (.txt)", choices=[])
                        map_classes = gr.Textbox(
                            label="Class Names (comma-separated, order 0, 1, 2...)",
                            placeholder="e.g. board, screw",
                        )
                        map_preview_md = gr.Markdown("")
                        btn_map_commit = gr.Button("Confirm Mapping", variant="primary")

                with gr.Accordion("Development", open=False):
                    gr.Markdown("Creates a small synthetic dataset for trying the workflow.")
                    btn_demo = gr.Button("Create Demo Dataset", variant="secondary", size="sm")

        # ================================================================ handlers
        def _switch_import_mode(mode: str):
            return gr.update(visible=(mode == "images_only")), gr.update(visible=(mode == "with_labels"))

        def _handle_mode1(files, name, classes_csv):
            if not files:
                return "⚠️ กรุณาเลือกไฟล์ภาพหรือ ZIP ก่อน", gr.update()
            try:
                report = controller.dataset_service.ingest_images_only(
                    uploaded_files=files,
                    dataset_name=name,
                    class_names=_parse_class_csv(classes_csv),
                )
                controller.session_service.update_session(active_dataset_id=report.dataset_id)
                choices = _dataset_choices(controller)
                msg = (
                    f"✅ นำเข้าภาพ **{report.total_images} ภาพ** เรียบร้อย (`{report.dataset_name}`)\n\n"
                    "กด **Open Annotator** ด้านบนเพื่อเริ่มตีกรอบ"
                )
                return msg, gr.update(choices=choices, value=report.dataset_id)
            except Exception as exc:
                return f"❌ นำเข้าไม่สำเร็จ: {exc}", gr.update()

        def _handle_mode2(files, name):
            if not files:
                return "⚠️ กรุณาเลือกไฟล์ ZIP หรือไฟล์ Dataset ก่อน", gr.update(), gr.update(visible=False), None, gr.update(), gr.update(), "", ""
            try:
                report = controller.upload_dataset(files, dataset_name=name)
                choices = _dataset_choices(controller)
                msg = (
                    f"✅ อัปโหลด **{report.dataset_name}** ({report.total_images} ภาพ) สำเร็จ — "
                    + ("โครงสร้างถูกต้องพร้อมใช้งาน!" if report.is_valid else "พบข้อผิดพลาดบางจุด ดูได้ที่แท็บ **3) ตรวจสอบ & แบ่ง Train/Val**")
                )
                return msg, gr.update(choices=choices, value=report.dataset_id), gr.update(visible=False), None, gr.update(), gr.update(), "", ""
            except Exception as exc:
                # Standard YOLO structure not found -> stage for manual folder mapping!
                try:
                    staged = controller.dataset_service.stage_custom_upload(files)
                except Exception as stage_exc:
                    return f"❌ อัปโหลดไม่สำเร็จ: {stage_exc}", gr.update(), gr.update(visible=False), None, gr.update(), gr.update(), "", ""

                img_choices = [(f"{rel} ({cnt} ภาพ)", rel) for rel, cnt in staged["image_dirs"]]
                lbl_choices = [(f"{rel} ({cnt} ไฟล์ .txt)", rel) for rel, cnt in staged["label_dirs"]] + [("(ไม่มี Label)", "(ไม่มี Label)")]
                if not img_choices:
                    return f"❌ ไม่พบไฟล์ภาพในไฟล์ที่อัปโหลด ({exc})", gr.update(), gr.update(visible=False), None, gr.update(), gr.update(), "", ""

                def_img = img_choices[0][1]
                def_lbl = lbl_choices[0][1]
                prev = controller.dataset_service.preview_custom_mapping(staged["stage_id"], staged["root_dir"], def_img, def_lbl)
                prev_md = _format_mapping_preview(prev)
                cls_str = ", ".join(staged["suggested_classes"])
                info = "ℹ️ โครงสร้างไฟล์ไม่ตรงตาม YOLO มาตรฐาน — กรุณาเลือกโฟลเดอร์ภาพและ Label ด้านล่างแล้วยืนยัน"
                return (
                    info,
                    gr.update(),
                    gr.update(visible=True),
                    staged,
                    gr.update(choices=img_choices, value=def_img),
                    gr.update(choices=lbl_choices, value=def_lbl),
                    cls_str,
                    prev_md,
                )

        def _format_mapping_preview(p: dict[str, Any]) -> str:
            sample = ", ".join(f"`{s}`" for s in p["sample_matched"]) if p["sample_matched"] else "_ไม่มี_"
            return (
                f"**ตัวอย่างผลจับคู่:** ภาพทั้งหมด **{p['total_images']}** · ไฟล์ Label **{p['total_labels']}** · "
                f"จับคู่ได้ **{p['matched']}** ภาพ · ภาพที่ไม่มี Label {p['unmatched_images']} · Label กำพร้า {p['orphan_labels']}\n\n"
                f"ตัวอย่างชื่อไฟล์ที่จับคู่ได้: {sample}"
            )

        def _handle_map_preview(staged, img_dir, lbl_dir):
            if not staged or img_dir is None:
                return ""
            try:
                p = controller.dataset_service.preview_custom_mapping(staged["stage_id"], staged["root_dir"], img_dir, lbl_dir)
                return _format_mapping_preview(p)
            except Exception as exc:
                return f"❌ {exc}"

        def _handle_map_commit(staged, custom_name, img_dir, lbl_dir, classes_csv):
            if not staged:
                return "⚠️ ไม่มีไฟล์ที่รอจับคู่", gr.update(), gr.update(visible=False), None
            try:
                report = controller.dataset_service.commit_custom_mapping(
                    stage_id=staged["stage_id"],
                    root_dir=staged["root_dir"],
                    dataset_name=custom_name or staged["default_name"],
                    image_rel_dir=img_dir,
                    label_rel_dir=lbl_dir,
                    class_names=_parse_class_csv(classes_csv),
                )
                choices = _dataset_choices(controller)
                msg = (
                    f"✅ สร้าง Dataset **{report.dataset_name}** ({report.total_images} ภาพ) เรียบร้อยแล้ว! "
                    "ไปที่แท็บ **2) ตีกรอบ & Auto-Label** เพื่อตรวจแก้ หรือ **3) ตรวจสอบ & แบ่ง Train/Val** เพื่อแบ่งชุดข้อมูล"
                )
                return msg, gr.update(choices=choices, value=report.dataset_id), gr.update(visible=False), None
            except Exception as exc:
                return f"❌ สร้าง Dataset ไม่สำเร็จ: {exc}", gr.update(), gr.update(), staged

        def _handle_create_demo():
            try:
                report = controller.create_demo_dataset()
                choices = _dataset_choices(controller)
                return gr.update(choices=choices, value=report.dataset_id), f"✅ สร้าง Demo Dataset (`{report.dataset_name}`) แล้ว"
            except Exception as exc:
                return gr.update(), f"❌ สร้าง Demo ไม่สำเร็จ: {exc}"

        def _handle_ds_change(ds_id: str | None):
            report = controller.get_dataset(ds_id) if ds_id else None
            if report:
                controller.session_service.update_session(active_dataset_id=report.dataset_id)
            s_html, i_md, t_md = _format_summary(report)
            return s_html, i_md, t_md, _class_rows(controller, ds_id)

        def _handle_revalidate(ds_id: str | None):
            if not ds_id:
                return _format_summary(None)
            report = controller.dataset_service.revalidate_dataset(ds_id)
            return _format_summary(report)

        def _handle_split(ds_id, tp, vp, sp, seed, drop_dup, name):
            if not ds_id:
                return "⚠️ กรุณาเลือก Dataset ก่อน", gr.update()
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
                md = (
                    f"✅ สร้าง Dataset ใหม่ **{new_report.dataset_name}** พร้อมนำไปเทรนแล้ว!\n\n"
                    f"- Train {counts['train']} · Val {counts['val']} · Test {counts['test']} ภาพ\n"
                    f"- ตัดภาพเสีย {counts['excluded_corrupt']} · ภาพซ้ำ {counts['excluded_duplicate']}"
                )
                return md, gr.update(choices=_dataset_choices(controller), value=new_report.dataset_id)
            except Exception as exc:
                return f"❌ แบ่ง Dataset ไม่สำเร็จ: {exc}", gr.update()

        def _handle_diag(ds_id):
            if not ds_id:
                return "⚠️ กรุณาเลือก Dataset ก่อน"
            try:
                return _diag_markdown(controller.annotation_service.diagnostics(ds_id))
            except Exception as exc:
                return f"❌ {exc}"

        def _handle_save_names(ds_id, rows):
            if not ds_id:
                return _class_rows(controller, None), "⚠️ กรุณาเลือก Dataset"
            try:
                new_names = {int(r[0]): str(r[1]) for r in (rows or []) if r and r[0] not in (None, "")}
                changed = controller.annotation_service.rename_classes(ds_id, new_names)
                return _class_rows(controller, ds_id), f"✅ บันทึกชื่อคลาสแล้ว ({changed} รายการเปลี่ยน)"
            except Exception as exc:
                return _class_rows(controller, ds_id), f"❌ {exc}"

        def _handle_remove_class(ds_id, cid):
            if not ds_id or cid is None:
                return _class_rows(controller, ds_id), "⚠️ ระบุ id ของคลาสที่ต้องการลบ"
            try:
                controller.annotation_service.remove_unused_class(ds_id, int(cid))
                return _class_rows(controller, ds_id), "Class deleted and remaining IDs re-indexed."
            except Exception as exc:
                return _class_rows(controller, ds_id), f"❌ {exc}"

        def _handle_export_lbl(ds_id):
            if not ds_id:
                return None, "⚠️ กรุณาเลือก Dataset"
            try:
                path = controller.annotation_service.export_labels(ds_id)
                return str(path), f"✅ Export แล้ว: `{Path(path).name}`"
            except Exception as exc:
                return None, f"❌ {exc}"

        def _handle_import_lbl(ds_id, zip_path, ow):
            if not ds_id:
                return "⚠️ กรุณาเลือก Dataset"
            try:
                res = controller.annotation_service.import_labels(ds_id, zip_path, ow)
                return (
                    f"✅ นำเข้า {res['imported']} · เหมือนเดิม {res['unchanged']} · "
                    f"ข้าม {res['skipped_existing']} · ผิดรูปแบบ {res['invalid']} · ไม่พบภาพคู่ {res['unmatched']}"
                )
            except Exception as exc:
                return f"❌ {exc}"

        def _handle_rename_ds(ds_id, new_name):
            if not ds_id:
                return "⚠️ กรุณาเลือก Dataset", gr.update(), ""
            try:
                rep = controller.dataset_service.rename_dataset(ds_id, new_name)
                return f"✅ เปลี่ยนชื่อเป็น **`{rep.dataset_name}`** แล้ว", gr.update(choices=_dataset_choices(controller), value=ds_id), ""
            except Exception as exc:
                return f"❌ {exc}", gr.update(), new_name

        def _handle_preview_split(ds_id, tp, vp, sp, drop_dup):
            if not ds_id:
                return "⚠️ กรุณาเลือก Dataset ก่อน"
            try:
                c = controller.dataset_service.preview_split(ds_id, float(tp or 0), float(vp or 0), float(sp or 0), bool(drop_dup))
                return (
                    f"👁️ **Preview Split ({c['total']} ภาพ):** "
                    f"Train **{c['train']}** · Val **{c['val']}** · Test **{c['test']}** "
                    f"(ตัดภาพเสีย {c['excluded_corrupt']}, ภาพซ้ำ {c['excluded_duplicate']}) "
                    "— กด **Confirm & สร้าง Split Dataset** เพื่อสร้างจริง"
                )
            except Exception as exc:
                return f"❌ {exc}"

        def _paint_library(query: str, status: str):
            return _render_library(controller, query, status)

        def _refresh_page(current_ds_id: str | None, query: str, status: str):
            choices = _dataset_choices(controller)
            valid_ids = {cid for _, cid in choices}
            selected = current_ds_id if current_ds_id in valid_ids else (choices[0][1] if choices else None)
            report = controller.get_dataset(selected) if selected else None
            s_html, i_md, t_md = _format_summary(report)
            return (
                gr.update(choices=choices, value=selected),
                s_html,
                i_md,
                t_md,
                _class_rows(controller, selected),
                _render_library(controller, query, status),
            )

        # ---- wiring
        import_mode.change(fn=_switch_import_mode, inputs=[import_mode], outputs=[mode1_box, mode2_box])
        btn_m1_create.click(
            fn=_handle_mode1,
            inputs=[m1_files, m1_name, m1_classes],
            outputs=[m1_msg, dataset_dropdown],
        ).then(fn=_paint_library, inputs=[library_query, library_status], outputs=[library_html])
        btn_m2_upload.click(
            fn=_handle_mode2,
            inputs=[m2_files, m2_name],
            outputs=[m2_msg, dataset_dropdown, mapping_box, stage_state, map_img_dir, map_lbl_dir, map_classes, map_preview_md],
        ).then(fn=_paint_library, inputs=[library_query, library_status], outputs=[library_html])
        map_img_dir.change(fn=_handle_map_preview, inputs=[stage_state, map_img_dir, map_lbl_dir], outputs=[map_preview_md])
        map_lbl_dir.change(fn=_handle_map_preview, inputs=[stage_state, map_img_dir, map_lbl_dir], outputs=[map_preview_md])
        btn_map_commit.click(
            fn=_handle_map_commit,
            inputs=[stage_state, m2_name, map_img_dir, map_lbl_dir, map_classes],
            outputs=[m2_msg, dataset_dropdown, mapping_box, stage_state],
        ).then(fn=_paint_library, inputs=[library_query, library_status], outputs=[library_html])
        btn_demo.click(fn=_handle_create_demo, outputs=[dataset_dropdown, m1_msg]).then(
            fn=_paint_library, inputs=[library_query, library_status], outputs=[library_html],
        )
        library_query.change(fn=_paint_library, inputs=[library_query, library_status], outputs=[library_html])
        library_status.change(fn=_paint_library, inputs=[library_query, library_status], outputs=[library_html])

        dataset_dropdown.change(fn=_handle_ds_change, inputs=[dataset_dropdown], outputs=[summary_html, issues_md, structure_md, classes_df])
        btn_revalidate.click(fn=_handle_revalidate, inputs=[dataset_dropdown], outputs=[summary_html, issues_md, structure_md])
        btn_rename_ds.click(fn=_handle_rename_ds, inputs=[dataset_dropdown, txt_rename_ds], outputs=[rename_ds_msg, dataset_dropdown, txt_rename_ds])
        btn_diag.click(fn=_handle_diag, inputs=[dataset_dropdown], outputs=[diag_md])
        btn_save_names.click(fn=_handle_save_names, inputs=[dataset_dropdown, classes_df], outputs=[classes_df, class_msg])
        btn_remove_class.click(fn=_handle_remove_class, inputs=[dataset_dropdown, num_remove_id], outputs=[classes_df, class_msg])
        btn_export_lbl.click(fn=_handle_export_lbl, inputs=[dataset_dropdown], outputs=[export_lbl_file, export_lbl_msg])
        btn_import_lbl.click(fn=_handle_import_lbl, inputs=[dataset_dropdown, import_lbl_file, chk_import_ow], outputs=[import_lbl_msg])

    return {
        "container": container,
        "dataset_dropdown": dataset_dropdown,
        "btn_open_annotate": btn_open_annotate,
        "refresh_fn": _refresh_page,
        "refresh_inputs": [dataset_dropdown, library_query, library_status],
        "refresh_outputs": [dataset_dropdown, summary_html, issues_md, structure_md, classes_df, library_html],
    }
