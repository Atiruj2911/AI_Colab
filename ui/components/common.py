"""Reusable HTML/Markdown UI components with clear non-color-only status indicators."""

from __future__ import annotations

import html
from typing import Any

from core.schemas import DatasetValidationReport, JobState
from utils.validation import format_bytes, format_duration

JOB_STATUS_DISPLAY: dict[str, tuple[str, str, str]] = {
    "queued": ("⏳", "รอเริ่มเทรน (Queued)", "badge-info"),
    "starting": ("🚀", "กำลังเตรียมโมเดล (Starting)", "badge-info"),
    "running": ("🟢", "กำลังเทรน (Running)", "badge-running"),
    "stopping": ("🛑", "กำลังหยุดอย่างปลอดภัย (Stopping)", "badge-warning"),
    "completed": ("✅", "เทรนเสร็จสมบูรณ์ (Completed)", "badge-success"),
    "stopped": ("⏹️", "หยุดโดยผู้ใช้ (Stopped)", "badge-warning"),
    "failed": ("❌", "เกิดข้อผิดพลาด (Failed)", "badge-danger"),
}


def render_status_badge(status: str | None) -> str:
    if not status:
        return '<span class="status-badge badge-muted">⚪ ยังไม่มีงานเทรน (Idle)</span>'
    icon, label, cls = JOB_STATUS_DISPLAY.get(status, ("ℹ️", status, "badge-muted"))
    return f'<span class="status-badge {cls}">{icon} {html.escape(label)}</span>'


def render_page_header(title: str, subtitle: str = "", right_html: str = "") -> str:
    sub_html = f'<div class="page-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    return (
        '<div class="page-header">'
        f'<div><h2 class="page-title">{html.escape(title)}</h2>{sub_html}</div>'
        f"<div>{right_html}</div>"
        "</div>"
    )


def render_kpi_cards(items: list[tuple[str, str, str, bool]]) -> str:
    """Render responsive KPI cards. Each item: (label, value, subtext, is_accent)."""
    cards: list[str] = []
    for label, value, sub, accent in items:
        val_cls = "kpi-value kpi-value-accent" if accent else "kpi-value"
        sub_html = f'<div class="kpi-sub">{html.escape(sub)}</div>' if sub else ""
        cards.append(
            '<div class="kpi-card">'
            f'<div class="kpi-label">{html.escape(label)}</div>'
            f'<div class="{val_cls}">{html.escape(value)}</div>'
            f"{sub_html}"
            "</div>"
        )
    return f'<div class="kpi-grid">{"".join(cards)}</div>'


def render_temp_storage_warning(compact: bool = False) -> str:
    if compact:
        return (
            '<div class="callout-warning">'
            "<strong>⚠️ ข้อจำกัด Colab Temporary Storage:</strong> "
            "ไฟล์ทั้งหมดอยู่ในดิสก์ชั่วคราวของ Runtime นี้ — หากแท็บเบราว์เซอร์หลุดแต่ Runtime ยังทำงานอยู่ "
            "สามารถกด <strong>Reconnect</strong> กลับมาดูผลต่อได้ แต่หาก <strong>Colab Runtime ถูกยุติ (Terminated/Disconnect Runtime)</strong> "
            "ไฟล์และ checkpoint จะหายทั้งหมด กรุณาดาวน์โหลด <code>best.pt</code> / <code>last.pt</code> เก็บลงเครื่องเมื่อได้ผลที่ต้องการ"
            "</div>"
        )
    return (
        '<div class="callout-warning">'
        "<strong>⚠️ ข้อสำคัญเรื่องพื้นที่ชั่วคราวและการกู้คืน (Colab Temporary Storage):</strong><br>"
        "• ระบบนี้บันทึก Dataset, Checkpoints (<code>best.pt</code>, <code>last.pt</code>) และผลลัพธ์ไว้ใน Temporary Disk ของ Colab Runtime โดยไม่บังคับเชื่อม Google Drive<br>"
        "• <strong>กรณี Browser หลุด/รีเฟรช แต่ Runtime ยังมีชีวิต:</strong> งานเทรนยังทำงานต่อเบื้องหลัง และสามารถกด <strong>Reconnect</strong> กลับมาติดตามได้โดยไม่เริ่มเทรนซ้ำ<br>"
        "• <strong>กรณี Colab Runtime ถูกยุติ (Idle timeout / ครบโควต้า / Restart Runtime):</strong> ข้อมูลใน Runtime จะถูกลบทั้งหมด ไม่สามารถกู้คืนจากในเครื่อง Colab ได้ ต้องใช้ไฟล์ checkpoint ที่ผู้ใช้ดาวน์โหลดเก็บไว้ภายนอกแล้วเท่านั้น"
        "</div>"
    )


def render_planned_notice(phase_label: str, description: str) -> str:
    return (
        '<div class="callout-info">'
        f"<strong>🗓️ ฟีเจอร์ในแผน ({html.escape(phase_label)}):</strong> "
        f"{html.escape(description)}"
        "</div>"
    )


def render_dataset_summary_card(report: DatasetValidationReport | None) -> str:
    if report is None:
        return (
            '<div class="studio-card">'
            "<div style='color:#93a4cc;font-size:14px;'>ยังไม่ได้เลือกหรืออัปโหลด Dataset — "
            "อัปโหลดไฟล์ ZIP (YOLO Format) หรือกด <strong>สร้าง Demo Dataset</strong> เพื่อเริ่มต้น</div>"
            "</div>"
        )

    if report.is_valid:
        badge = '<span class="status-badge badge-success">✅ โครงสร้างถูกต้อง (YOLO Detection)</span>'
    else:
        badge = f'<span class="status-badge badge-danger">❌ พบข้อผิดพลาด {len(report.errors)} รายการ</span>'

    warn_badge = ""
    if report.warnings:
        warn_badge = f' <span class="status-badge badge-warning">⚠️ คำเตือน {len(report.warnings)} รายการ</span>'

    split_rows = []
    for sp_name, sp in report.splits.items():
        split_rows.append(
            f"<tr>"
            f"<td style='padding:6px 8px;'><strong>{html.escape(sp_name)}</strong></td>"
            f"<td style='padding:6px 8px;text-align:right;'>{sp.image_count:,}</td>"
            f"<td style='padding:6px 8px;text-align:right;'>{sp.label_count:,}</td>"
            f"<td style='padding:6px 8px;text-align:right;'>{sp.matched_pairs:,}</td>"
            f"<td style='padding:6px 8px;text-align:right;'>{sp.total_boxes:,}</td>"
            f"</tr>"
        )

    classes_html = ", ".join(
        f"<code>{cid}: {html.escape(name)}</code> ({report.class_counts.get(name, 0):,})"
        for cid, name in sorted(report.class_names.items())
    ) or "<em>ไม่พบรายชื่อคลาส</em>"

    return (
        '<div class="studio-card">'
        f"<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:10px;'>"
        f"<div style='font-size:17px;font-weight:700;'>📦 {html.escape(report.dataset_name)}</div>"
        f"<div>{badge}{warn_badge}</div>"
        "</div>"
        + render_kpi_cards(
            [
                ("รูปภาพทั้งหมด", f"{report.total_images:,}", "Images", False),
                ("ไฟล์ Label (.txt)", f"{report.total_labels:,}", f"{report.total_boxes:,} boxes", False),
                ("จำนวน Class", f"{len(report.class_names):,}", "จาก data.yaml", True),
            ]
        )
        + f"<div style='margin:8px 0;font-size:13px;color:#cbd5f5;'><strong>คลาส (จำนวน Box):</strong> {classes_html}</div>"
        + (
            "<table style='width:100%;border-collapse:collapse;font-size:13px;margin-top:8px;'>"
            "<thead><tr style='border-bottom:1px solid #25365e;color:#93a4cc;'>"
            "<th style='text-align:left;padding:6px 8px;'>Split</th>"
            "<th style='text-align:right;padding:6px 8px;'>Images</th>"
            "<th style='text-align:right;padding:6px 8px;'>Labels</th>"
            "<th style='text-align:right;padding:6px 8px;'>Matched</th>"
            "<th style='text-align:right;padding:6px 8px;'>Boxes</th>"
            "</tr></thead><tbody>"
            + "".join(split_rows)
            + "</tbody></table>"
            if split_rows
            else ""
        )
        + "</div>"
    )


def render_validation_issues_markdown(report: DatasetValidationReport | None) -> str:
    if report is None:
        return "ยังไม่มีรายงานการตรวจสอบ"

    lines: list[str] = []
    if not report.errors and not report.warnings:
        lines.append("✅ **ผ่านการตรวจสอบทั้งหมด** ไม่พบ Error หรือ Warning — พร้อมนำไปเทรน YOLO11 ได้ทันที")
        return "\n".join(lines)

    if report.errors:
        lines.append(f"### ❌ ข้อผิดพลาดที่ต้องแก้ไข ({len(report.errors)} รายการ)")
        for err in report.errors:
            lines.append(f"- **[{err.code}]** {err.message}")
            if err.hint:
                lines.append(f"  - 💡 *คำแนะนำ:* {err.hint}")

    if report.warnings:
        lines.append(f"\n### ⚠️ คำเตือน ({len(report.warnings)} รายการ)")
        for warn in report.warnings:
            lines.append(f"- **[{warn.code}]** {warn.message}")
            if warn.hint:
                lines.append(f"  - 💡 *คำแนะนำ:* {warn.hint}")

    return "\n".join(lines)


def render_job_sidebar_pill(job: JobState | None, alive: bool) -> str:
    if job is None:
        return (
            '<div style="padding:10px;border-radius:10px;background:#10192e;border:1px solid #1f2e52;font-size:12px;">'
            "<div style='color:#93a4cc;margin-bottom:4px;'>สถานะงานเทรน</div>"
            + render_status_badge(None)
            + "</div>"
        )
    sub = f"Run: {job.run_name} • Ep {job.current_epoch}/{job.total_epochs}"
    if alive and job.pid:
        sub += f" • PID {job.pid}"
    return (
        '<div style="padding:10px;border-radius:10px;background:#10192e;border:1px solid #263966;font-size:12px;">'
        "<div style='color:#93a4cc;margin-bottom:4px;'>สถานะงานล่าสุด</div>"
        + render_status_badge(job.status)
        + f"<div style='color:#cbd5f5;margin-top:6px;word-break:break-word;'>{html.escape(sub)}</div>"
        "</div>"
    )


def render_runtime_status_panel(
    job: JobState | None,
    gpu: dict[str, Any],
    disk: dict[str, Any],
    heartbeat_stale: bool = False,
) -> str:
    if job is None:
        return (
            '<div class="studio-card">'
            "<h4 style='margin-top:0;'>สถานะการเทรนและทรัพยากร</h4>"
            "<div style='color:#93a4cc;font-size:13px;'>ยังไม่มีงานเทรนในเซสชันนี้ — ไปที่หน้า <strong>Training</strong> เพื่อเริ่มเทรน</div>"
            f"<div style='margin-top:10px;font-size:12.5px;color:#cbd5f5;'>🖥️ <strong>GPU:</strong> {html.escape(str(gpu.get('summary', '-')))}</div>"
            f"<div style='margin-top:4px;font-size:12.5px;color:#cbd5f5;'>💾 <strong>Storage:</strong> {html.escape(str(disk.get('summary', '-')))}</div>"
            "</div>"
        )

    best_map_str = f"{job.best_map50_95:.4f}" if job.best_map50_95 is not None else "รอผล Validation"
    best_ready = "✅ พร้อมดาวน์โหลด" if job.best_checkpoint_path else "⏳ รอบันทึก"
    last_ready = "✅ พร้อมดาวน์โหลด" if job.last_checkpoint_path else "⏳ รอบันทึก"
    hb_note = (
        '<div class="callout-warning">⚠️ ไม่ได้รับ Heartbeat นานผิดปกติ — Worker อาจกำลังประมวลผลหนักหรือค้าง กรุณาตรวจ Training Log</div>'
        if heartbeat_stale
        else ""
    )
    err_note = (
        f'<div class="callout-danger">❌ <strong>ข้อผิดพลาด:</strong> {html.escape(job.error_message)}</div>'
        if job.error_message
        else ""
    )

    return (
        '<div class="studio-card">'
        "<h4 style='margin:0 0 10px 0;color:#f8faff;'>สถานะการเทรน (Live)</h4>"
        f"<div style='font-size:13px;line-height:1.9;color:#dce6ff;'>"
        f"• <strong>Run:</strong> <code>{html.escape(job.run_name)}</code> ({html.escape(job.model_variant)})<br>"
        f"• <strong>Best Model (best.pt):</strong> {best_ready} (mAP50-95: <strong>{best_map_str}</strong>)<br>"
        f"• <strong>Last Checkpoint (last.pt):</strong> {last_ready}<br>"
        f"• <strong>Worker PID:</strong> <code>{job.pid or '-'}</code> | <strong>Elapsed:</strong> {format_duration(job.elapsed_seconds)}<br>"
        f"• <strong>GPU / VRAM:</strong> {html.escape(str(gpu.get('summary', '-')))}<br>"
        f"• <strong>Storage:</strong> {html.escape(str(disk.get('summary', '-')))}<br>"
        f"• <strong>Auto Save:</strong> บันทึก best.pt + last.pt อัตโนมัติ (ปรับเงื่อนไขละเอียดใน Phase 4)"
        "</div>"
        + hb_note
        + err_note
        + "</div>"
    )
