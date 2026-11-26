"""Reusable HTML/Markdown UI components with clear non-color-only status indicators."""

from __future__ import annotations

import html
from typing import Any

from core.schemas import DatasetValidationReport, JobState
from utils.validation import format_bytes, format_duration

def _icon(paths: str, size: int = 14) -> str:
    """Inline Lucide-style stroke icon. Decorative; the adjacent text carries the meaning."""
    return (
        f'<svg class="status-icon" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        f'aria-hidden="true">{paths}</svg>'
    )


_ICON = {
    "clock": _icon('<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>'),
    "loader": _icon('<path d="M21 12a9 9 0 1 1-6.2-8.6"/>'),
    "activity": _icon('<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'),
    "square": _icon('<rect width="14" height="14" x="5" y="5" rx="1"/>'),
    "check": _icon('<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>'),
    "pause": _icon('<rect width="4" height="14" x="6" y="5"/><rect width="4" height="14" x="14" y="5"/>'),
    "alert": _icon('<path d="m21.7 18-9-16a2 2 0 0 0-3.4 0l-9 16A2 2 0 0 0 2 21h18a2 2 0 0 0 1.7-3z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'),
    "minus": _icon('<circle cx="12" cy="12" r="10"/><line x1="8" y1="12" x2="16" y2="12"/>'),
    "info": _icon('<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>'),
}

JOB_STATUS_DISPLAY: dict[str, tuple[str, str, str]] = {
    "queued": ("clock", "Queued", "badge-info"),
    "starting": ("loader", "Starting", "badge-info"),
    "running": ("activity", "Running", "badge-running"),
    "stopping": ("square", "Stopping", "badge-warning"),
    "completed": ("check", "Completed", "badge-success"),
    "stopped": ("pause", "Stopped", "badge-warning"),
    "failed": ("alert", "Failed", "badge-danger"),
}


def render_status_badge(status: str | None) -> str:
    if not status:
        return f'<span class="status-badge badge-muted">{_ICON["minus"]} Idle</span>'
    key, label, cls = JOB_STATUS_DISPLAY.get(status, ("info", status, "badge-muted"))
    return f'<span class="status-badge {cls}">{_ICON.get(key, _ICON["info"])} {html.escape(label)}</span>'


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
            "<strong>Colab temporary disk.</strong> "
            "Files live only on this runtime. Reconnect recovers a live job after a browser refresh. "
            "If the runtime is terminated, checkpoints are gone — download <code>best.pt</code> and <code>last.pt</code>."
            "</div>"
        )
    return (
        '<div class="callout-warning">'
        "<strong>Colab temporary storage.</strong> "
        "Datasets and checkpoints stay on this runtime disk. A browser refresh can reconnect to a live job. "
        "Terminating the runtime deletes everything that was not downloaded."
        "</div>"
    )


def render_planned_notice(phase_label: str, description: str) -> str:
    return (
        '<div class="callout-info">'
        f"<strong>ฟีเจอร์ในแผน ({html.escape(phase_label)}):</strong> "
        f"{html.escape(description)}"
        "</div>"
    )


def render_dataset_summary_card(report: DatasetValidationReport | None) -> str:
    if report is None:
        return (
            '<div class="studio-card">'
            '<div class="empty-note">No dataset selected. Import images or a YOLO dataset to continue.</div>'
            "</div>"
        )

    if report.is_valid:
        badge = f'<span class="status-badge badge-success">{_ICON["check"]} Ready</span>'
    else:
        badge = f'<span class="status-badge badge-danger">{_ICON["alert"]} {len(report.errors)} errors</span>'

    warn_badge = ""
    if report.warnings:
        warn_badge = f' <span class="status-badge badge-warning">{_ICON["alert"]} {len(report.warnings)} warnings</span>'

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
    ) or "<em>No classes</em>"

    return (
        '<div class="studio-card">'
        f"<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:8px;'>"
        f"<div style='font-size:16px;font-weight:600;'>{html.escape(report.dataset_name)}</div>"
        f"<div>{badge}{warn_badge}</div>"
        "</div>"
        + render_kpi_cards(
            [
                ("Images", f"{report.total_images:,}", "Total", False),
                ("Labels", f"{report.total_labels:,}", f"{report.total_boxes:,} boxes", False),
                ("Classes", f"{len(report.class_names):,}", "From data.yaml", False),
            ]
        )
        + f"<div style='margin:8px 0;font-size:13px;'><strong>Classes:</strong> {classes_html}</div>"
        + (
            "<table class='data-table'>"
            "<thead><tr>"
            "<th>Split</th><th>Images</th><th>Labels</th><th>Matched</th><th>Boxes</th>"
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
        return "No validation report yet."

    lines: list[str] = []
    if not report.errors and not report.warnings:
        lines.append("**Validation passed.** No errors or warnings.")
        return "\n".join(lines)

    if report.errors:
        lines.append(f"### Errors ({len(report.errors)})")
        for err in report.errors:
            lines.append(f"- **[{err.code}]** {err.message}")
            if err.hint:
                lines.append(f"  - {err.hint}")

    if report.warnings:
        lines.append(f"\n### Warnings ({len(report.warnings)})")
        for warn in report.warnings:
            lines.append(f"- **[{warn.code}]** {warn.message}")
            if warn.hint:
                lines.append(f"  - {warn.hint}")

    return "\n".join(lines)


def render_job_sidebar_pill(job: JobState | None, alive: bool) -> str:
    if job is None:
        return (
            '<div class="nav-status-card">'
            '<div class="nav-status-label">Runtime</div>'
            + render_status_badge(None)
            + "</div>"
        )
    sub = f"{job.run_name} · Ep {job.current_epoch}/{job.total_epochs}"
    if alive and job.pid:
        sub += f" · PID {job.pid}"
    elif job.status in {"queued", "starting", "running"} and not alive:
        sub += " · worker not running"
    return (
        '<div class="nav-status-card">'
        '<div class="nav-status-label">Runtime</div>'
        + render_status_badge(job.status)
        + f'<div class="nav-status-sub">{html.escape(sub)}</div>'
        + "</div>"
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
