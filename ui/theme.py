"""Dark Navy / Indigo Gradio theme and custom CSS inspired by ui_reference/."""

from __future__ import annotations

import gradio as gr


def build_studio_theme() -> gr.Theme:
    """Create the default Dark Navy / Indigo theme."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), gr.themes.GoogleFont("Noto Sans Thai"), "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ).set(
        body_background_fill="#080d1a",
        body_background_fill_dark="#080d1a",
        body_text_color="#e5ecff",
        body_text_color_dark="#e5ecff",
        body_text_color_subdued="#94a3c8",
        body_text_color_subdued_dark="#94a3c8",
        background_fill_primary="#0f172a",
        background_fill_primary_dark="#0f172a",
        background_fill_secondary="#131d35",
        background_fill_secondary_dark="#131d35",
        block_background_fill="#10192e",
        block_background_fill_dark="#10192e",
        block_border_color="#233255",
        block_border_color_dark="#233255",
        block_label_text_color="#cbd5f5",
        block_label_text_color_dark="#cbd5f5",
        block_title_text_color="#f1f5ff",
        block_title_text_color_dark="#f1f5ff",
        input_background_fill="#0b1224",
        input_background_fill_dark="#0b1224",
        input_border_color="#28385e",
        input_border_color_dark="#28385e",
        button_primary_background_fill="linear-gradient(90deg, #6344f5 0%, #7c5cff 100%)",
        button_primary_background_fill_dark="linear-gradient(90deg, #6344f5 0%, #7c5cff 100%)",
        button_primary_background_fill_hover="linear-gradient(90deg, #7456ff 0%, #8c70ff 100%)",
        button_primary_background_fill_hover_dark="linear-gradient(90deg, #7456ff 0%, #8c70ff 100%)",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="#182442",
        button_secondary_background_fill_dark="#182442",
        button_secondary_border_color="#2c3e68",
        button_secondary_border_color_dark="#2c3e68",
        button_secondary_text_color="#e2e8ff",
        button_secondary_text_color_dark="#e2e8ff",
        table_even_background_fill="#0f172a",
        table_even_background_fill_dark="#0f172a",
        table_odd_background_fill="#16213c",
        table_odd_background_fill_dark="#16213c",
        table_row_focus="#1f2f54",
        table_row_focus_dark="#1f2f54",
        table_border_color="#233255",
        table_border_color_dark="#233255",
    )


STUDIO_CSS = """
/* Force consistent dark-navy studio canvas — 100% full width across ALL pages */
.gradio-container,
.gradio-container > .main,
.gradio-container > .main > .wrap,
.gradio-container .contain {
    max-width: 100% !important;
    width: 100% !important;
}
.gradio-container {
    background: radial-gradient(circle at top right, #161f42 0%, #080d1a 48%, #070b15 100%) !important;
    color: #e5ecff !important;
    padding: 10px 18px !important;
    margin: 0 !important;
}

/* Fixed sidebar + fluid 100% main content shell (identical width on every page) */
.studio-shell {
    width: 100% !important;
    flex-wrap: nowrap !important;
    align-items: flex-start !important;
    gap: 16px !important;
}
.studio-main {
    flex: 1 1 auto !important;
    width: 100% !important;
    min-width: 0 !important;
}

/* Fix inline <code> contrast across all markdown blocks */
.gradio-container code {
    background: #1b294b !important;
    color: #c7d2fe !important;
    border: 1px solid #2c3f6e !important;
    padding: 1px 6px !important;
    border-radius: 5px !important;
    font-size: 0.9em !important;
}

/* Keep small action/nav button rows on one clean line */
.compact-row {
    flex-wrap: nowrap !important;
    align-items: center !important;
    gap: 6px !important;
}
.compact-row > * {
    min-width: 0 !important;
}

/* Fix gr.File list contrast & prevent huge vertical stretching on multi-file upload */
.file-preview-holder, .file-preview {
    max-height: 240px !important;
    overflow-y: auto !important;
    background: #0f172a !important;
}
table.file-preview tr,
table.file-preview tr:nth-child(odd),
table.file-preview tr:nth-child(even) {
    background: #0f172a !important;
    color: #e5ecff !important;
}
table.file-preview tr:nth-child(odd) {
    background: #16213c !important;
}
table.file-preview td,
table.file-preview td a {
    color: #e5ecff !important;
}
table.file-preview td.download a {
    color: #a5b4fc !important;
}

/* Sidebar */
.studio-sidebar {
    flex: 0 0 210px !important;
    width: 210px !important;
    min-width: 210px !important;
    max-width: 210px !important;
    background: #0c1326 !important;
    border: 1px solid #1e2c4d !important;
    border-radius: 16px !important;
    padding: 14px 12px !important;
    min-height: 78vh;
}

.studio-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 8px 14px 8px;
    border-bottom: 1px solid #1e2c4d;
    margin-bottom: 10px;
}

.studio-brand-icon {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    background: linear-gradient(135deg, #6d4aff 0%, #3b82f6 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    color: #ffffff;
    box-shadow: 0 0 18px rgba(109, 74, 255, 0.45);
    flex-shrink: 0;
}

.studio-brand-title {
    font-size: 16px;
    font-weight: 700;
    color: #f8faff;
    line-height: 1.2;
}

.studio-brand-sub {
    font-size: 11px;
    color: #8b9dc7;
}

.nav-btn button {
    justify-content: flex-start !important;
    text-align: left !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-weight: 500 !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    color: #b7c4e6 !important;
    transition: all 0.15s ease !important;
}

.nav-btn button:hover {
    background: #172240 !important;
    color: #ffffff !important;
    border-color: #283a66 !important;
}

.nav-btn-active button {
    justify-content: flex-start !important;
    text-align: left !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-weight: 600 !important;
    background: linear-gradient(90deg, #5d3ef0 0%, #7c5cff 100%) !important;
    color: #ffffff !important;
    border: 1px solid #8d73ff !important;
    box-shadow: 0 4px 16px rgba(109, 74, 255, 0.32) !important;
}

/* Main workspace cards */
.studio-card {
    background: #10192e !important;
    border: 1px solid #223257 !important;
    border-radius: 14px !important;
    padding: 16px !important;
}

.page-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 12px;
}

.page-title {
    font-size: 26px;
    font-weight: 700;
    color: #f8faff;
    margin: 0;
}

.page-subtitle {
    font-size: 13px;
    color: #93a4cc;
    margin-top: 2px;
}

/* Hero Banner (Home) */
.hero-box {
    background: linear-gradient(135deg, #131c3a 0%, #1c1949 55%, #111b36 100%);
    border: 1px solid #2e3e6e;
    border-radius: 16px;
    padding: 26px 28px;
    margin-bottom: 14px;
}

.hero-title {
    font-size: 32px;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 6px 0;
}

.hero-sub {
    font-size: 15px;
    color: #b7c6eb;
    margin: 0 0 18px 0;
}

.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.feature-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    border-radius: 12px;
    background: #16213f;
    border: 1px solid #2d3f6e;
    color: #eef3ff;
    font-size: 13px;
    font-weight: 600;
}

.feature-pill-accent {
    background: linear-gradient(135deg, #6242f5 0%, #835dff 100%);
    border-color: #9576ff;
}

/* Status Badges (icon + text + border so color is never the sole cue) */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 12.5px;
    font-weight: 600;
    border: 1px solid transparent;
}

.badge-success { background: rgba(16, 185, 129, 0.16); color: #6ee7b7; border-color: rgba(16, 185, 129, 0.45); }
.badge-running { background: rgba(34, 197, 94, 0.18); color: #86efac; border-color: rgba(34, 197, 94, 0.5); }
.badge-warning { background: rgba(245, 158, 11, 0.16); color: #fcd34d; border-color: rgba(245, 158, 11, 0.45); }
.badge-danger  { background: rgba(239, 68, 68, 0.16); color: #fca5a5; border-color: rgba(239, 68, 68, 0.45); }
.badge-info    { background: rgba(99, 102, 241, 0.18); color: #c7d2fe; border-color: rgba(99, 102, 241, 0.48); }
.badge-muted   { background: rgba(148, 163, 184, 0.14); color: #cbd5e1; border-color: rgba(148, 163, 184, 0.35); }

/* KPI Strip */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
    margin: 10px 0 14px 0;
}

.kpi-card {
    background: #101a31;
    border: 1px solid #24355c;
    border-radius: 12px;
    padding: 13px 15px;
}

.kpi-label {
    font-size: 12px;
    color: #93a4cc;
    margin-bottom: 4px;
}

.kpi-value {
    font-size: 21px;
    font-weight: 700;
    color: #f8faff;
}

.kpi-value-accent {
    color: #6ee7b7;
}

.kpi-sub {
    font-size: 11px;
    color: #7d90bc;
    margin-top: 3px;
}

/* Warning / Info Callouts */
.callout-warning {
    background: rgba(245, 158, 11, 0.11);
    border: 1px solid rgba(245, 158, 11, 0.4);
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 12px 14px;
    color: #fde68a;
    font-size: 13px;
    line-height: 1.5;
    margin: 8px 0;
}

.callout-info {
    background: rgba(99, 102, 241, 0.11);
    border: 1px solid rgba(99, 102, 241, 0.38);
    border-left: 4px solid #818cf8;
    border-radius: 10px;
    padding: 12px 14px;
    color: #dbeafe;
    font-size: 13px;
    line-height: 1.5;
    margin: 8px 0;
}

.callout-danger {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.45);
    border-left: 4px solid #ef4444;
    border-radius: 10px;
    padding: 12px 14px;
    color: #fecaca;
    font-size: 13px;
    line-height: 1.5;
    margin: 8px 0;
}

/* Progress Bar */
.progress-track {
    width: 100%;
    height: 12px;
    background: #0b1224;
    border: 1px solid #26365e;
    border-radius: 999px;
    overflow: hidden;
    margin: 8px 0;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
    border-radius: 999px;
    transition: width 0.3s ease;
}

/* Danger stop button */
.danger-btn button {
    background: linear-gradient(90deg, #dc2626 0%, #ef4444 100%) !important;
    border-color: #f87171 !important;
    color: #ffffff !important;
}

/* Code / tree block */
.tree-box pre {
    background: #091022 !important;
    border: 1px solid #223257 !important;
    border-radius: 10px !important;
    padding: 12px !important;
    color: #cbd5f5 !important;
    font-size: 12.5px !important;
    overflow-x: auto;
}
"""
