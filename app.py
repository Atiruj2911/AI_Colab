"""Entry point for AI Training Studio (Phase 1 — End-to-End MVP)."""

from __future__ import annotations

import argparse
from pathlib import Path

from application.app_controller import AppController
from config.app_config import AppConfig
from infrastructure.monitoring.system_monitor import format_startup_banner
from ui.app_ui import launch_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Training Studio — YOLO11 Gradio App")
    parser.add_argument("--workspace", type=Path, default=None, help="Override runtime workspace root directory")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host address")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument(
        "--share",
        action="store_true",
        default=None,
        help="Force Gradio public share link (auto-enabled on Google Colab)",
    )
    parser.add_argument(
        "--no-share",
        action="store_true",
        help="Disable Gradio public share link even on Google Colab",
    )
    parser.add_argument("--debug", action="store_true", help="Enable Gradio debug mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig.from_env(workspace_override=args.workspace)
    controller = AppController(config=config)

    print(format_startup_banner(controller.workspace.root), flush=True)

    share_flag: bool | None
    if args.no_share:
        share_flag = False
    elif args.share:
        share_flag = True
    else:
        share_flag = None

    launch_app(
        controller=controller,
        share=share_flag,
        server_name=args.host,
        server_port=args.port,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
