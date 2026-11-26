"""Professional Light Workstation theme.

Design tokens live here and are applied to Gradio plus the shared CSS classes.
Page HTML still contains the old dark-palette inline colors; a scoped bridge
remaps those exact values so this step stays readable without editing pages.
Annotation and augmentation documents inside iframes are not affected.
"""

from __future__ import annotations

import gradio as gr

# Core palette
PAPER = "#F3F6FA"
SURFACE = "#FFFFFF"
SUNKEN = "#F1F5F9"
INK = "#1E293B"
GRAPHITE = "#64748B"
LINE = "#E2E8F0"
COBALT = "#2563EB"
COBALT_HOVER = "#1D4ED8"
COBALT_SOFT = "#EFF6FF"
STAGE = "#2A2E35"
RUNNING = "#0F7F63"
WARNING = "#A86400"
DANGER = "#C42F2A"
DANGER_HOVER = "#A82824"
ON_ACCENT = "#FFFFFF"

_COBALT_HUE = gr.themes.Color(
    c50="#EFF6FF",
    c100="#DBEAFE",
    c200="#BFDBFE",
    c300="#93C5FD",
    c400="#60A5FA",
    c500=COBALT,
    c600=COBALT_HOVER,
    c700="#163F94",
    c800="#123377",
    c900="#0E275A",
    c950="#091833",
    name="cobalt",
)


def build_studio_theme() -> gr.Theme:
    """Light workstation theme. Dark slots are pinned to the same values so the OS theme cannot flip the app back to navy."""
    theme = gr.themes.Base(
        primary_hue=_COBALT_HUE,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=[
            gr.themes.GoogleFont("Inter"),
            gr.themes.GoogleFont("Noto Sans Thai"),
            "sans-serif",
        ],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
    )
    light_values = {
        "body_background_fill": PAPER,
        "body_text_color": INK,
        "body_text_color_subdued": GRAPHITE,
        "background_fill_primary": SURFACE,
        "background_fill_secondary": SUNKEN,
        "block_background_fill": SURFACE,
        "block_border_color": LINE,
        "block_label_background_fill": SURFACE,
        "block_label_text_color": INK,
        "block_title_text_color": INK,
        "block_label_text_size": "13px",
        "block_title_text_size": "15px",
        "block_radius": "8px",
        "input_background_fill": SURFACE,
        "input_border_color": LINE,
        "input_border_color_focus": COBALT,
        "input_radius": "6px",
        "border_color_primary": LINE,
        "border_color_accent": COBALT,
        "color_accent": COBALT,
        "link_text_color": COBALT,
        "link_text_color_hover": COBALT_HOVER,
        "link_text_color_active": COBALT_HOVER,
        "button_primary_background_fill": COBALT,
        "button_primary_background_fill_hover": COBALT_HOVER,
        "button_primary_text_color": ON_ACCENT,
        "button_primary_text_color_hover": ON_ACCENT,
        "button_primary_border_color": COBALT,
        "button_primary_border_color_hover": COBALT_HOVER,
        "button_secondary_background_fill": SURFACE,
        "button_secondary_background_fill_hover": SUNKEN,
        "button_secondary_text_color": INK,
        "button_secondary_text_color_hover": INK,
        "button_secondary_border_color": LINE,
        "button_secondary_border_color_hover": "#C5CDD6",
        "button_cancel_background_fill": DANGER,
        "button_cancel_background_fill_hover": DANGER_HOVER,
        "button_cancel_text_color": ON_ACCENT,
        "button_cancel_text_color_hover": ON_ACCENT,
        "button_cancel_border_color": DANGER_HOVER,
        "button_cancel_border_color_hover": DANGER_HOVER,
        "button_large_padding": "8px 16px",
        "button_large_text_size": "14px",
        "button_large_text_weight": "600",
        "button_large_radius": "6px",
        "button_medium_padding": "8px 16px",
        "button_medium_text_size": "14px",
        "button_medium_text_weight": "600",
        "button_medium_radius": "6px",
        "button_small_padding": "4px 12px",
        "button_small_text_size": "13px",
        "button_small_text_weight": "600",
        "button_small_radius": "6px",
        "shadow_drop": "none",
        "shadow_drop_lg": "none",
        "block_shadow": "none",
        "checkbox_label_text_color": INK,
        "checkbox_background_color": SURFACE,
        "checkbox_border_color": LINE,
        "checkbox_label_background_fill_selected": COBALT,
        "slider_color": COBALT,
        "table_even_background_fill": SURFACE,
        "table_odd_background_fill": SUNKEN,
        "table_row_focus": COBALT_SOFT,
        "table_border_color": LINE,
        "stat_background_fill": SUNKEN,
        "panel_background_fill": SURFACE,
        "panel_border_color": LINE,
        "loader_color": COBALT,
    }
    payload: dict[str, str] = {}
    for key, value in light_values.items():
        if hasattr(theme, key):
            payload[key] = value
        dark_key = f"{key}_dark"
        if hasattr(theme, dark_key):
            payload[dark_key] = value
    return theme.set(**payload)


STUDIO_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Thai:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

.gradio-container {{
    --paper: {PAPER};
    --surface: {SURFACE};
    --sunken: {SUNKEN};
    --ink: {INK};
    --graphite: {GRAPHITE};
    --line: {LINE};
    --cobalt: {COBALT};
    --cobalt-hover: {COBALT_HOVER};
    --cobalt-soft: {COBALT_SOFT};
    --stage: {STAGE};
    --running: {RUNNING};
    --warning: {WARNING};
    --danger: {DANGER};
    --danger-hover: {DANGER_HOVER};
    --on-accent: {ON_ACCENT};
    --font-sans: "Inter", "Noto Sans Thai", sans-serif;
    --font-mono: "IBM Plex Mono", ui-monospace, monospace;
    --space-1: 4px;
    --space-2: 8px;
    --space-3: 12px;
    --space-4: 16px;
    --space-5: 24px;
    --space-6: 32px;
    --radius-sm: 4px;
    --radius-md: 6px;
    --radius-lg: 8px;
    --btn-sm: 34px;
    --btn-md: 38px;
    --header-h: 56px;
    --sidebar-w: 216px;
    max-width: 100% !important;
    width: 100% !important;
    background: var(--paper) !important;
    color: var(--ink) !important;
    font-family: var(--font-sans) !important;
    font-size: 14px !important;
    padding: 20px 24px !important;
    margin: 0 !important;
}}
.gradio-container > .main,
.gradio-container > .main > .wrap,
.gradio-container .contain {{
    max-width: 100% !important;
    width: 100% !important;
}}
.gradio-container button,
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container label {{
    font-family: var(--font-sans) !important;
}}
.gradio-container button {{ cursor: pointer; }}
.gradio-container button:disabled,
.gradio-container button[disabled] {{
    opacity: 0.45 !important;
    cursor: not-allowed !important;
}}
.gradio-container button:focus-visible,
.gradio-container input:focus-visible,
.gradio-container textarea:focus-visible,
.gradio-container select:focus-visible {{
    outline: 2px solid var(--cobalt) !important;
    outline-offset: 2px !important;
}}

/* Gradio 6 default button class is "lg". Map it to the md size.
   True 44px page CTAs need an opt-in class, added with the page pass. */
.gradio-container button.lg {{
    min-height: var(--btn-md) !important;
    padding: 8px 16px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    border-radius: var(--radius-md) !important;
}}
.gradio-container button.sm {{
    min-height: var(--btn-sm) !important;
    padding: 4px 12px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    border-radius: var(--radius-md) !important;
}}
.gradio-container button.primary:hover:not(:disabled) {{
    filter: brightness(1.06);
}}
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container textarea {{
    min-height: 38px;
    font-size: 14px;
}}
/* Slider values live in a short box beside the label. The 38px field
   height pushes that number down onto the track. */
.gradio-container .tab-like-container input[type="number"] {{
    min-height: 0 !important;
    height: 100% !important;
    padding: 0 8px !important;
    line-height: 26px !important;
    font-size: 13px !important;
}}
.gradio-container .tab-like-container {{
    height: 28px !important;
    align-items: center !important;
}}
.gradio-container .tab-like-container button {{
    min-height: 0 !important;
    height: 28px !important;
    padding: 0 8px !important;
    border-radius: 0 !important;
}}
.gradio-container .wrap > .head {{
    flex-wrap: nowrap !important;
    align-items: center !important;
    gap: 8px !important;
}}
.gradio-container .slider_input_container {{
    align-items: center !important;
}}

/* Danger: Gradio 6 puts variant="stop" on the button as class "stop". */
.gradio-container button.stop {{
    background: var(--danger) !important;
    border-color: var(--danger-hover) !important;
    color: var(--on-accent) !important;
}}
.gradio-container button.stop:hover:not(:disabled) {{
    background: var(--danger-hover) !important;
    border-color: var(--danger-hover) !important;
    color: var(--on-accent) !important;
}}
.gradio-container button.stop:disabled {{
    background: #F3D6D4 !important;
    border-color: #E7C4C2 !important;
    color: #8A4A46 !important;
    opacity: 1 !important;
    cursor: not-allowed !important;
}}

.gradio-container .prose,
.gradio-container .markdown {{
    color: var(--ink);
    font-size: 14px;
    line-height: 1.5;
}}
.gradio-container .prose h3,
.gradio-container .markdown h3 {{
    font-size: 18px;
    font-weight: 600;
    color: var(--ink);
    letter-spacing: 0;
    margin-top: 24px;
}}
.gradio-container code,
.gradio-container pre,
.tree-box pre {{
    font-family: var(--font-mono) !important;
}}
.gradio-container code {{
    background: var(--sunken) !important;
    color: var(--ink) !important;
    border: 1px solid var(--line) !important;
    padding: 1px 6px !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.92em !important;
}}

.studio-shell {{
    width: 100% !important;
    flex-wrap: nowrap !important;
    align-items: flex-start !important;
    gap: var(--space-4) !important;
}}
.studio-main {{
    flex: 1 1 auto !important;
    width: 100% !important;
    min-width: 0 !important;
}}
.compact-row {{
    flex-wrap: nowrap !important;
    align-items: center !important;
    gap: 8px !important;
}}
.compact-row button.lg {{
    align-self: center !important;
    height: var(--btn-md) !important;
}}
.compact-row button.sm {{
    align-self: center !important;
    height: var(--btn-sm) !important;
}}
.gradio-container button[role="tab"] {{
    min-height: 36px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}}
.data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}
.data-table th {{
    text-align: left;
    padding: 8px 10px;
    color: var(--graphite);
    font-weight: 600;
    border-bottom: 1px solid var(--line);
    white-space: nowrap;
}}
.data-table td {{
    padding: 8px 10px;
    border-bottom: 1px solid var(--line);
    color: var(--ink);
    vertical-align: middle;
}}
.data-table tr:last-child td {{ border-bottom: none; }}
.dash-grid {{
    display: grid;
    grid-template-columns: minmax(0, 1.3fr) minmax(280px, 0.7fr);
    gap: 16px;
    align-items: start;
}}
.kv-list {{ font-size: 13px; line-height: 1.7; color: var(--ink); }}
.kv-list span {{ color: var(--graphite); }}
.empty-note {{ color: var(--graphite); font-size: 13px; line-height: 1.5; }}
@media (max-width: 1100px) {{
    .dash-grid {{ grid-template-columns: 1fr; }}
}}
.compact-row > * {{ min-width: 0 !important; }}

.file-preview-holder, .file-preview {{
    max-height: 240px !important;
    overflow-y: auto !important;
    background: var(--surface) !important;
}}
table.file-preview tr,
table.file-preview tr:nth-child(odd),
table.file-preview tr:nth-child(even) {{
    background: var(--surface) !important;
    color: var(--ink) !important;
}}
table.file-preview tr:nth-child(odd) {{ background: var(--sunken) !important; }}
table.file-preview td,
table.file-preview td a {{ color: var(--ink) !important; }}
table.file-preview td.download a {{ color: var(--cobalt) !important; }}

.studio-sidebar {{
    flex: 0 0 var(--sidebar-w) !important;
    width: var(--sidebar-w) !important;
    min-width: var(--sidebar-w) !important;
    max-width: var(--sidebar-w) !important;
    display: flex !important;
    flex-direction: column !important;
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-lg) !important;
    padding: 12px !important;
    min-height: calc(100vh - 40px);
}}
.nav-status {{ margin-top: auto !important; padding-top: 12px !important; }}
.nav-status-card {{
    padding: 10px;
    border-radius: var(--radius-md);
    background: var(--sunken);
    border: 1px solid var(--line);
    font-size: 12px;
}}
.nav-status-label {{ color: var(--graphite); margin-bottom: 6px; font-weight: 600; }}
.nav-status-sub {{ color: var(--ink); margin-top: 6px; word-break: break-word; line-height: 1.4; }}
.studio-brand {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 8px 12px 8px;
    border-bottom: 1px solid var(--line);
    margin-bottom: 10px;
}}
.studio-brand-icon {{
    width: 36px;
    height: 36px;
    border-radius: var(--radius-lg);
    background: var(--cobalt);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--on-accent);
    flex-shrink: 0;
}}
.studio-brand-title {{
    font-size: 15px;
    font-weight: 600;
    color: var(--ink);
    line-height: 1.25;
}}
.studio-brand-sub {{
    font-size: 12px;
    color: var(--graphite);
}}

/* Gradio 6 puts nav classes on the <button>, not a wrapper. */
.studio-sidebar button.nav-btn,
.studio-sidebar button.nav-btn-active {{
    width: 100% !important;
    min-height: 34px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    text-align: left !important;
    border-radius: var(--radius-md) !important;
    padding: 6px 12px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    color: var(--ink) !important;
    box-shadow: none !important;
}}
.studio-sidebar button.nav-btn:hover {{
    background: var(--sunken) !important;
    border-color: var(--line) !important;
    color: var(--ink) !important;
}}
.studio-sidebar button.nav-btn-active {{
    background: var(--cobalt-soft) !important;
    color: var(--cobalt) !important;
    border-color: transparent !important;
    font-weight: 600 !important;
    box-shadow: inset 3px 0 0 var(--cobalt) !important;
}}
.studio-sidebar button.nav-btn::before,
.studio-sidebar button.nav-btn-active::before {{
    content: "";
    width: 16px;
    height: 16px;
    flex: 0 0 16px;
    margin-right: 8px;
    background-color: currentColor;
    -webkit-mask: var(--ico) center / 16px 16px no-repeat;
    mask: var(--ico) center / 16px 16px no-repeat;
}}
#nav-home {{ --ico: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z'/%3E%3Cpolyline points='9 22 9 12 15 12 15 22'/%3E%3C/svg%3E"); }}
#nav-dataset {{ --ico: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2'%3E%3Cellipse cx='12' cy='5' rx='9' ry='3'/%3E%3Cpath d='M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5'/%3E%3Cpath d='M3 12c0 1.66 4 3 9 3s9-1.34 9-3'/%3E%3C/svg%3E"); }}
#nav-annotate {{ --ico: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 20h9'/%3E%3Cpath d='M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z'/%3E%3C/svg%3E"); }}
#nav-prepare {{ --ico: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linejoin='round'%3E%3Cpolygon points='12 2 2 7 12 12 22 7 12 2'/%3E%3Cpolyline points='2 17 12 22 22 17'/%3E%3Cpolyline points='2 12 12 17 22 12'/%3E%3C/svg%3E"); }}
#nav-training {{ --ico: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='22 12 18 12 15 21 9 3 6 12 2 12'/%3E%3C/svg%3E"); }}
#nav-models {{ --ico: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z'/%3E%3Cpath d='m3.3 7 8.7 5 8.7-5'/%3E%3Cpath d='M12 22V12'/%3E%3C/svg%3E"); }}
#nav-settings {{ --ico: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z'/%3E%3Ccircle cx='12' cy='12' r='3'/%3E%3C/svg%3E"); }}

.studio-card {{
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-lg) !important;
    padding: 16px !important;
    color: var(--ink);
}}
.page-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
    min-height: var(--header-h);
    margin-bottom: 12px;
}}
.page-title {{
    font-size: 24px;
    font-weight: 600;
    color: var(--ink);
    margin: 0;
    line-height: 1.3;
}}
.page-subtitle {{
    font-size: 13px;
    color: var(--graphite);
    margin-top: 2px;
    line-height: 1.5;
}}

.hero-box {{
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    padding: 16px 20px;
    margin-bottom: 14px;
}}
.hero-title {{
    font-size: 24px;
    font-weight: 600;
    color: var(--ink);
    margin: 0 0 6px 0;
}}
.hero-sub {{
    font-size: 14px;
    color: var(--graphite);
    margin: 0 0 16px 0;
    line-height: 1.5;
    max-width: 72ch;
}}
.pill-row {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.feature-pill {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border-radius: var(--radius-md);
    background: var(--surface);
    border: 1px solid var(--line);
    color: var(--ink);
    font-size: 13px;
    font-weight: 600;
}}
.feature-pill-accent {{
    background: var(--cobalt);
    border-color: var(--cobalt);
    color: var(--on-accent);
}}

.status-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 2px 10px;
    min-height: 22px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid transparent;
}}
.badge-success {{ background: #E5F6F0; color: var(--running); border-color: #B7E4D4; }}
.badge-running {{ background: #E5F6F0; color: var(--running); border-color: #B7E4D4; }}
.badge-warning {{ background: #FBF3E4; color: var(--warning); border-color: #F0DDB4; }}
.badge-danger  {{ background: #FBEDEC; color: var(--danger); border-color: #F3C9C6; }}
.badge-info    {{ background: var(--cobalt-soft); color: var(--cobalt); border-color: #C9D8F4; }}
.badge-muted   {{ background: var(--sunken); color: var(--graphite); border-color: var(--line); }}
.status-icon {{ display: block; flex: 0 0 auto; }}

.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
    margin: 8px 0 12px 0;
}}
.kpi-card {{
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    padding: 12px 14px;
    min-height: 64px;
}}
.kpi-label {{ font-size: 12px; color: var(--graphite); margin-bottom: 4px; }}
.kpi-value {{
    font-size: 20px;
    font-weight: 600;
    color: var(--ink);
    font-variant-numeric: tabular-nums;
}}
.kpi-value-accent {{ color: var(--running); }}
.kpi-sub {{ font-size: 12px; color: var(--graphite); margin-top: 2px; }}

.callout-warning {{
    background: #FBF3E4;
    border: 1px solid #F0DDB4;
    border-left: 3px solid var(--warning);
    border-radius: var(--radius-md);
    padding: 12px 14px;
    color: #6B4408;
    font-size: 13px;
    line-height: 1.5;
    margin: 8px 0;
}}
.callout-info {{
    background: var(--cobalt-soft);
    border: 1px solid #C9D8F4;
    border-left: 3px solid var(--cobalt);
    border-radius: var(--radius-md);
    padding: 12px 14px;
    color: var(--ink);
    font-size: 13px;
    line-height: 1.5;
    margin: 8px 0;
}}
.callout-danger {{
    background: #FBEDEC;
    border: 1px solid #F3C9C6;
    border-left: 3px solid var(--danger);
    border-radius: var(--radius-md);
    padding: 12px 14px;
    color: #7A1E1B;
    font-size: 13px;
    line-height: 1.5;
    margin: 8px 0;
}}
.progress-track {{
    width: 100%;
    height: 8px;
    background: var(--sunken);
    border: 1px solid var(--line);
    border-radius: 999px;
    overflow: hidden;
    margin: 8px 0;
}}
.progress-fill {{
    height: 100%;
    background: var(--running);
    border-radius: 999px;
}}
.tree-box pre {{
    background: var(--sunken) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-lg) !important;
    padding: 12px !important;
    color: var(--ink) !important;
    font-size: 13px !important;
    overflow-x: auto;
}}

/* Page HTML from the navy pass still sets these exact colors inline.
   Remap them until shared components use the variables above.
   Iframe documents keep their own styles. #080d1a stays a stage color. */
.studio-shell [style*="color:#f8faff"],
.studio-shell [style*="color:#f1f5ff"],
.studio-shell [style*="color:#e5ecff"],
.studio-shell [style*="color:#e2e8f0"],
.studio-shell [style*="color:#dce6ff"],
.studio-shell [style*="color:#cbd5f5"],
.studio-shell [style*="color:#eef3ff"] {{ color: var(--ink) !important; }}
.studio-shell [style*="color:#93a4cc"],
.studio-shell [style*="color:#94a3c8"],
.studio-shell [style*="color:#8fa1c7"],
.studio-shell [style*="color:#b7c4e6"],
.studio-shell [style*="color:#b7c6eb"],
.studio-shell [style*="color:#7d90bc"] {{ color: var(--graphite) !important; }}
.studio-shell [style*="color:#6ee7b7"],
.studio-shell [style*="color:#4ade80"],
.studio-shell [style*="color:#86efac"] {{ color: var(--running) !important; }}
.studio-shell [style*="color:#93c5fd"] {{ color: var(--cobalt) !important; }}
.studio-shell [style*="color:#fca5a5"],
.studio-shell [style*="color:#fecaca"] {{ color: var(--danger) !important; }}
.studio-shell [style*="background:#10192e"],
.studio-shell [style*="background:#101a31"],
.studio-shell [style*="background:#0f172a"],
.studio-shell [style*="background:#0c1326"],
.studio-shell [style*="background:#16213c"],
.studio-shell [style*="background:#16213f"],
.studio-shell [style*="background:#0b1224"],
.studio-shell [style*="background:#1a2748"],
.studio-shell [style*="background:#091022"] {{
    background: var(--sunken) !important;
    border-color: var(--line) !important;
}}
.studio-shell [style*="background:#080d1a"] {{ background: var(--stage) !important; }}
.studio-shell [style*="1px solid #233255"],
.studio-shell [style*="1px solid #25365e"],
.studio-shell [style*="1px solid #24355c"],
.studio-shell [style*="1px solid #263966"],
.studio-shell [style*="1px solid #1f2e52"],
.studio-shell [style*="1px solid #2b3f6b"] {{ border-color: var(--line) !important; }}

@media (max-width: 1100px) {{
    .studio-shell {{ flex-wrap: wrap !important; }}
    .studio-sidebar {{
        display: flex !important;
        flex-flow: row wrap !important;
        align-items: center !important;
        gap: 8px !important;
        flex: 1 1 100% !important;
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
        min-height: 0 !important;
    }}
    .studio-brand {{
        flex: 1 1 180px;
        border-bottom: 0;
        margin-bottom: 0;
        padding-bottom: 0;
    }}
    .studio-sidebar button.nav-btn,
    .studio-sidebar button.nav-btn-active {{
        width: auto !important;
        flex: 1 1 120px !important;
    }}
    .studio-main .row:not(.compact-row) {{ flex-wrap: wrap !important; }}
    .studio-main .row:not(.compact-row) > * {{
        flex: 1 1 280px !important;
        min-width: 0 !important;
    }}
}}
@media (prefers-reduced-motion: reduce) {{
    .gradio-container *, .gradio-container *::before, .gradio-container *::after {{
        transition: none !important;
        animation: none !important;
    }}
}}
"""
