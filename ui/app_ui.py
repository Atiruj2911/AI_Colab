"""Main Gradio Blocks application shell (Phase 1 — 4-page clean navigation + mounted Annotate API)."""

from __future__ import annotations

import inspect
from typing import Any

import gradio as gr

from application.app_controller import AppController
from config.defaults import APP_NAME, DEFAULT_MONITOR_REFRESH_SECONDS
from ui.annotate_api import create_annotate_router
from ui.components.common import render_job_sidebar_pill
from ui.pages.annotate import build_annotate_page
from ui.pages.dataset import build_dataset_page
from ui.pages.home import build_home_page
from ui.pages.prepare import build_prepare_page
from ui.pages.training_workspace import build_training_workspace_page
from ui.theme import STUDIO_CSS, build_studio_theme

# Linear Phase 1 workflow navigation: Home -> Dataset -> Annotate -> Prepare -> Training
PAGE_KEYS = [
    ("home", "Home"),
    ("dataset", "Dataset"),
    ("annotate", "Annotate"),
    ("prepare", "Prepare"),
    ("training", "Training"),
]


def _blocks_accepts(param_name: str) -> bool:
    return param_name in inspect.signature(gr.Blocks.__init__).parameters


def _launch_accepts(param_name: str) -> bool:
    return param_name in inspect.signature(gr.Blocks.launch).parameters


def build_app(controller: AppController) -> tuple[gr.Blocks, gr.Theme, str]:
    """Construct the AI Training Studio Gradio Blocks UI."""
    theme = build_studio_theme()
    blocks_kwargs: dict[str, Any] = {"title": APP_NAME}
    if _blocks_accepts("fill_width"):
        blocks_kwargs["fill_width"] = True
    if _blocks_accepts("theme"):
        blocks_kwargs["theme"] = theme
    if _blocks_accepts("css"):
        blocks_kwargs["css"] = STUDIO_CSS

    with gr.Blocks(**blocks_kwargs) as demo:
        active_page_state = gr.State(value="home")

        with gr.Row(equal_height=False, elem_classes=["studio-shell"]):
            # Left Fixed Sidebar
            with gr.Column(scale=0, min_width=210, elem_classes=["studio-sidebar"]):
                gr.HTML(
                    f"""
                    <div class="studio-brand">
                        <div class="studio-brand-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg></div>
                        <div>
                            <div class="studio-brand-title">{APP_NAME}</div>
                            <div class="studio-brand-sub">YOLO11 Studio</div>
                        </div>
                    </div>
                    """
                )

                nav_buttons: dict[str, gr.Button] = {}
                for key, label in PAGE_KEYS:
                    cls = ["nav-btn-active"] if key == "home" else ["nav-btn"]
                    nav_buttons[key] = gr.Button(label, elem_classes=cls)

                initial_snap = controller.get_monitor_snapshot()
                sidebar_job_html = gr.HTML(
                    value=render_job_sidebar_pill(initial_snap["job"], initial_snap["alive"]),
                )

            # Right Main Content Area
            with gr.Column(scale=1, min_width=0, elem_classes=["studio-main"]):
                pages: dict[str, dict[str, Any]] = {
                    "home": build_home_page(controller),
                    "dataset": build_dataset_page(controller),
                    "annotate": build_annotate_page(controller),
                    "prepare": build_prepare_page(controller),
                    "training": build_training_workspace_page(controller),
                }

        # Live polling timer: updates sidebar pill + Training Monitor (Left side ONLY, NEVER touches Testing!)
        poll_timer = gr.Timer(value=DEFAULT_MONITOR_REFRESH_SECONDS, active=True)

        def _switch_page(target_key: str):
            vis_updates = [gr.update(visible=(key == target_key)) for key, _ in PAGE_KEYS]
            btn_updates = [
                gr.update(elem_classes=["nav-btn-active"] if key == target_key else ["nav-btn"])
                for key, _ in PAGE_KEYS
            ]
            snap = controller.get_monitor_snapshot()
            pill = render_job_sidebar_pill(snap["job"], snap["alive"])
            return [target_key, pill, *vis_updates, *btn_updates]

        switch_outputs = [
            active_page_state,
            sidebar_job_html,
            *[pages[key]["container"] for key, _ in PAGE_KEYS],
            *[nav_buttons[key] for key, _ in PAGE_KEYS],
        ]

        # Wire Sidebar navigation + page-specific refresh on entry
        for key, _ in PAGE_KEYS:
            evt = nav_buttons[key].click(
                fn=lambda k=key: _switch_page(k),
                outputs=switch_outputs,
            )
            page_info = pages[key]
            if "refresh_fn" in page_info and "refresh_outputs" in page_info:
                evt.then(
                    fn=page_info["refresh_fn"],
                    inputs=page_info.get("refresh_inputs", []),
                    outputs=page_info["refresh_outputs"],
                )

        # Home workflow navigation shortcuts
        for btn_name, target_key in (
            ("btn_go_dataset", "dataset"),
            ("btn_go_annotate", "annotate"),
            ("btn_go_prepare", "prepare"),
            ("btn_go_training", "training"),
        ):
            evt = pages["home"][btn_name].click(
                fn=lambda k=target_key: _switch_page(k),
                outputs=switch_outputs,
            )
            target_info = pages[target_key]
            if "refresh_fn" in target_info:
                evt.then(
                    fn=target_info["refresh_fn"],
                    inputs=target_info.get("refresh_inputs", []),
                    outputs=target_info["refresh_outputs"],
                )

        # Dataset -> Open Annotate Workspace button
        def _open_annotate_for_dataset(selected_ds: str | None):
            if selected_ds:
                controller.session_service.update_session(active_dataset_id=selected_ds)
            return _switch_page("annotate")

        pages["dataset"]["btn_open_annotate"].click(
            fn=_open_annotate_for_dataset,
            inputs=[pages["dataset"]["dataset_dropdown"]],
            outputs=switch_outputs,
        ).then(
            fn=pages["annotate"]["refresh_fn"],
            inputs=pages["annotate"]["refresh_inputs"],
            outputs=pages["annotate"]["refresh_outputs"],
        )

        # Annotate -> Next: Prepare (Split & Augment)
        pages["annotate"]["btn_go_prepare"].click(
            fn=lambda: _switch_page("prepare"),
            outputs=switch_outputs,
        ).then(
            fn=pages["prepare"]["refresh_fn"],
            inputs=pages["prepare"]["refresh_inputs"],
            outputs=pages["prepare"]["refresh_outputs"],
        )

        # Prepare -> Next: Training
        def _open_training_for_dataset(selected_ds: str | None):
            if selected_ds:
                controller.session_service.update_session(active_dataset_id=selected_ds)
            return _switch_page("training")

        pages["prepare"]["btn_go_training"].click(
            fn=_open_training_for_dataset,
            inputs=[pages["prepare"]["dataset_dropdown"]],
            outputs=switch_outputs,
        ).then(
            fn=pages["training"]["refresh_fn"],
            inputs=pages["training"]["refresh_inputs"],
            outputs=pages["training"]["refresh_outputs"],
        )

        # Live polling tick: updates sidebar job pill + Training Monitor panel ONLY (never auto-updates Testing!)
        def _on_timer_tick(current_page: str, selected_run: str | None):
            snap = controller.get_monitor_snapshot(run_id=selected_run or None)
            pill = render_job_sidebar_pill(snap["job"], snap["alive"])
            if current_page == "training":
                mon_vals = pages["training"]["monitor_refresh_fn"](selected_run)
                return [pill, *mon_vals]
            return [pill, *[gr.update() for _ in pages["training"]["monitor_refresh_outputs"]]]

        poll_timer.tick(
            fn=_on_timer_tick,
            inputs=[active_page_state, pages["training"]["run_selector"]],
            outputs=[sidebar_job_html, *pages["training"]["monitor_refresh_outputs"]],
        )

    return demo, theme, STUDIO_CSS


def launch_app(
    controller: AppController,
    share: bool | None = None,
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    debug: bool = False,
) -> None:
    """Build and launch the Gradio application with mounted Annotation Workspace routes."""
    demo, theme, css = build_app(controller)
    effective_share = controller.config.is_colab if share is None else bool(share)

    launch_kwargs: dict[str, Any] = {
        "share": effective_share,
        "server_name": server_name,
        "server_port": server_port,
        "debug": debug,
        "allowed_paths": [str(controller.workspace.root)],
    }
    if _launch_accepts("theme") and not _blocks_accepts("theme"):
        launch_kwargs["theme"] = theme
    if _launch_accepts("css") and not _blocks_accepts("css"):
        launch_kwargs["css"] = css

    launch_kwargs["prevent_thread_lock"] = True
    fastapi_app, _, _ = demo.queue(default_concurrency_limit=4).launch(**launch_kwargs)
    fastapi_app.include_router(create_annotate_router(controller))
    demo.block_thread()
