"""Page 7: Sessions & Recovery (current session, reconnect, run artifacts download, runtime-loss notice)."""

from __future__ import annotations

import html
from typing import Any

import gradio as gr

from application.app_controller import AppController
from ui.components.common import (
    render_page_header,
    render_planned_notice,
    render_status_badge,
    render_temp_storage_warning,
)
from utils.validation import format_bytes, format_duration


def _render_current_session_card(controller: AppController) -> str:
    session = controller.session_service.get_or_create_session()
    snap = controller.get_monitor_snapshot()
    job = snap["job"]

    if job is None:
        return (
            '<div class="studio-card">'
            "<div style='display:flex;justify-content:space-between;align-items:center;'>"
            "<h4 style='margin:0;'>เซสชันปัจจุบัน</h4>"
            + render_status_badge(None)
            + "</div>"
            f"<div style='margin-top:10px;font-size:13px;line-height:1.9;color:#cbd5f5;'>"
            f"<strong>Session ID:</strong> <code>{html.escape(str(session.get('session_id', '-')))}</code><br>"
            f"<strong>สร้างเมื่อ:</strong> {html.escape(str(session.get('created_at', '-'))[:19])}<br>"
            f"<strong>Active Dataset:</strong> <code>{html.escape(str(session.get('active_dataset_id') or '-'))}</code><br>"
            "<strong>งานเทรนล่าสุด:</strong> ยังไม่มีงานเทรน"
            "</div></div>"
        )

    pct = max(0.0, min(100.0, float(job.progress_pct)))
    alive_text = "🟢 Worker ยังทำงานอยู่ (สามารถ Reconnect ได้)" if snap["alive"] else "⚪ Worker ไม่ได้ทำงานอยู่"
    best_map = f"{job.best_map50_95:.4f}" if job.best_map50_95 is not None else "-"

    return (
        '<div class="studio-card">'
        "<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;'>"
        "<h4 style='margin:0;'>เซสชันปัจจุบัน</h4>"
        + render_status_badge(job.status)
        + "</div>"
        f"<div style='margin-top:10px;font-size:13px;line-height:1.9;color:#dce6ff;'>"
        f"<strong>Session ID:</strong> <code>{html.escape(job.session_id)}</code><br>"
        f"<strong>Run:</strong> <code>{html.escape(job.run_name)}</code> ({html.escape(job.model_variant)})<br>"
        f"<strong>เริ่มเมื่อ:</strong> {html.escape(job.started_at[:19].replace('T', ' '))} UTC<br>"
        f"<strong>สถานะ:</strong> Epoch {job.current_epoch} / {job.total_epochs} | Elapsed {format_duration(job.elapsed_seconds)} | Best mAP50-95: <strong>{best_map}</strong><br>"
        f"<strong>Worker Status:</strong> {alive_text} (PID: <code>{job.pid or '-'}</code>)"
        "</div>"
        f'<div class="progress-track"><div class="progress-fill" style="width:{pct:.1f}%;"></div></div>'
        f"<div style='font-size:12px;color:#93a4cc;text-align:right;'>{pct:.1f}%</div>"
        "</div>"
    )


def _runs_table_rows(controller: AppController) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for run in controller.list_runs():
        rows.append(
            [
                run.run_id,
                run.run_name,
                run.model_variant,
                run.status,
                f"{run.current_epoch}/{run.total_epochs}",
                round(run.best_map50_95, 4) if run.best_map50_95 is not None else "-",
                format_duration(run.elapsed_seconds),
                run.started_at[:19].replace("T", " "),
            ]
        )
    return rows


def _run_dropdown_choices(controller: AppController) -> list[tuple[str, str]]:
    return [(f"{r.run_name} [{r.status}] ({r.run_id})", r.run_id) for r in controller.list_runs()]


def _run_artifact_paths(controller: AppController, run_id: str | None) -> tuple[str, list[str] | None]:
    if not run_id:
        return "เลือก Run เพื่อดูไฟล์ Checkpoint, Config และ Log ที่พร้อมดาวน์โหลด", None
    artifacts = controller.list_run_artifacts(run_id)
    if not artifacts:
        return f"ยังไม่พบไฟล์ artifact ใน Run `{run_id}`", None
    lines = [f"- **{a.name}** (`{a.kind}`) — {format_bytes(a.size_bytes)}" for a in artifacts]
    return "\n".join(lines), [a.path for a in artifacts]


def build_sessions_page(controller: AppController) -> dict[str, Any]:
    run_choices = _run_dropdown_choices(controller)
    first_run = run_choices[0][1] if run_choices else None
    art_md_init, art_files_init = _run_artifact_paths(controller, first_run)

    with gr.Column(visible=False) as container:
        gr.HTML(
            render_page_header(
                "Sessions & Recovery",
                "ตรวจสอบเซสชันปัจจุบัน เชื่อมต่อกลับไปยังงานเทรนที่กำลังรัน และดาวน์โหลด Checkpoints/Config เก็บลงเครื่อง",
            )
        )

        gr.HTML(render_temp_storage_warning(compact=False))

        with gr.Tabs():
            with gr.Tab("เซสชันปัจจุบัน"):
                with gr.Row():
                    with gr.Column(scale=7):
                        session_card_html = gr.HTML(value=_render_current_session_card(controller))
                    with gr.Column(scale=5):
                        with gr.Group(elem_classes=["studio-card"]):
                            gr.Markdown("#### การเชื่อมต่อและการกู้คืน")
                            btn_reconnect_live = gr.Button(
                                "🔄 Reconnect ไปยังงานเทรนปัจจุบัน (ไม่เริ่มเทรนซ้ำ)",
                                variant="primary",
                            )
                            btn_resume_planned = gr.Button(
                                "Resume from Last Checkpoint (เพิ่มเติมใน Phase 4)",
                                interactive=False,
                            )
                            btn_finetune_planned = gr.Button(
                                "Start Fine-tuning from Best (เพิ่มเติมใน Phase 4)",
                                interactive=False,
                            )
                            reconnect_msg = gr.Markdown(value="")

                gr.Markdown("#### ดาวน์โหลด Checkpoints, Config และ Log ของ Run")
                with gr.Row():
                    run_select_dd = gr.Dropdown(
                        label="เลือก Run เพื่อดาวน์โหลดไฟล์สำคัญ",
                        choices=run_choices,
                        value=first_run,
                        scale=5,
                    )
                    btn_refresh_sessions = gr.Button("🔄 รีเฟรชข้อมูลเซสชัน", variant="secondary", scale=2)
                artifact_list_md = gr.Markdown(value=art_md_init)
                artifact_files = gr.File(
                    label="ไฟล์ทั้งหมดของ Run นี้ (best.pt / last.pt / train_config.json / data.yaml / train.log)",
                    value=art_files_init,
                    interactive=False,
                )

            with gr.Tab("ประวัติการเทรน (Runs ใน Runtime นี้)"):
                runs_table = gr.Dataframe(
                    headers=["Run ID", "Run Name", "Model", "Status", "Epoch", "Best mAP50-95", "Elapsed", "Started (UTC)"],
                    value=_runs_table_rows(controller),
                    interactive=False,
                )
                gr.HTML(
                    render_planned_notice(
                        "Phase 5 — Experiment Comparison",
                        "การเปรียบเทียบหลายการทดลอง (Experiment Comparison) และ Import/Export Project Config เต็มรูปแบบอยู่ในแผน Phase 5",
                    )
                )

        def _refresh_sessions(selected_run: str | None):
            choices = _run_dropdown_choices(controller)
            ids = {rid for _, rid in choices}
            target = selected_run if selected_run in ids else (choices[0][1] if choices else None)
            art_md, art_paths = _run_artifact_paths(controller, target)
            return (
                _render_current_session_card(controller),
                gr.update(choices=choices, value=target),
                art_md,
                art_paths,
                _runs_table_rows(controller),
            )

        def _on_select_run(run_id: str | None):
            return _run_artifact_paths(controller, run_id)

        def _on_reconnect_click():
            snap = controller.get_monitor_snapshot()
            job = snap["job"]
            if job is None:
                return "ℹ️ ไม่พบงานเทรนที่กำลังทำงานอยู่ใน Runtime นี้", False
            if snap["alive"]:
                return (
                    f"✅ เชื่อมต่อกับงานที่กำลังรันอยู่แล้ว: `{job.run_name}` (PID `{job.pid}`, Epoch {job.current_epoch}/{job.total_epochs}) — กำลังเปิดหน้า Monitor",
                    True,
                )
            return (
                f"ℹ️ งานล่าสุด `{job.run_name}` อยู่ในสถานะ **{job.status}** (ไม่ได้กำลังรัน) — คุณสามารถดูผลหรือดาวน์โหลด checkpoint ได้ด้านล่าง",
                True,
            )

        session_refresh_outputs = [
            session_card_html,
            run_select_dd,
            artifact_list_md,
            artifact_files,
            runs_table,
        ]

        run_select_dd.change(fn=_on_select_run, inputs=[run_select_dd], outputs=[artifact_list_md, artifact_files])
        btn_refresh_sessions.click(fn=_refresh_sessions, inputs=[run_select_dd], outputs=session_refresh_outputs)

    return {
        "container": container,
        "btn_reconnect_live": btn_reconnect_live,
        "reconnect_msg": reconnect_msg,
        "reconnect_fn": _on_reconnect_click,
        "refresh_fn": _refresh_sessions,
        "refresh_inputs": [run_select_dd],
        "refresh_outputs": session_refresh_outputs,
    }
