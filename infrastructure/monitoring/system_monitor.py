"""Real runtime environment, GPU/VRAM, CPU/RAM, and disk monitoring."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from utils.validation import format_bytes


def get_package_versions() -> dict[str, str]:
    """Collect real installed versions of core runtime packages without upgrading anything."""
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
    }

    # PyTorch & CUDA
    try:
        import torch

        versions["pytorch"] = str(torch.__version__)
        versions["cuda_version"] = str(torch.version.cuda or "CPU only")
        if torch.cuda.is_available():
            versions["gpu"] = torch.cuda.get_device_name(0)
            versions["gpu_count"] = str(torch.cuda.device_count())
        else:
            versions["gpu"] = "ไม่พบ GPU (ใช้ CPU)"
            versions["gpu_count"] = "0"
    except Exception as exc:
        versions["pytorch"] = f"Unavailable ({exc})"
        versions["cuda_version"] = "Unavailable"
        versions["gpu"] = "Unavailable"
        versions["gpu_count"] = "0"

    for key, module_name in (
        ("ultralytics", "ultralytics"),
        ("gradio", "gradio"),
        ("onnx", "onnx"),
        ("onnxruntime", "onnxruntime"),
    ):
        try:
            mod = __import__(module_name)
            versions[key] = str(getattr(mod, "__version__", "installed"))
        except Exception:
            versions[key] = "Not installed"

    return versions


def get_available_devices() -> list[tuple[str, str]]:
    """Return list of (display_label, device_value) based on real hardware."""
    devices: list[tuple[str, str]] = [("Auto", "auto")]
    try:
        import torch

        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(idx)
                devices.append((f"GPU {idx} ({name})", str(idx)))
    except Exception:
        pass

    devices.append(("CPU", "cpu"))
    return devices


def get_gpu_vram_status() -> dict[str, Any]:
    """Query real GPU name, VRAM usage, and utilization where available."""
    result: dict[str, Any] = {
        "available": False,
        "name": "CPU (ไม่พบ GPU)",
        "vram_used_mb": None,
        "vram_total_mb": None,
        "gpu_util_pct": None,
        "summary": "ไม่พบ GPU (กำลังทำงานบน CPU)",
    }

    # Prefer nvidia-smi so it reflects memory used by the separate training subprocess too
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            first_line = proc.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first_line.split(",")]
            if len(parts) >= 4:
                name = parts[0]
                used_mb = float(parts[1])
                total_mb = float(parts[2])
                util_pct = float(parts[3])
                result.update(
                    {
                        "available": True,
                        "name": name,
                        "vram_used_mb": used_mb,
                        "vram_total_mb": total_mb,
                        "gpu_util_pct": util_pct,
                        "summary": f"{name} | VRAM {used_mb / 1024:.2f} / {total_mb / 1024:.2f} GB ({util_pct:.0f}%)",
                    }
                )
                return result
    except Exception:
        pass

    # Fallback to torch.cuda
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_mb = props.total_memory / (1024 * 1024)
            free_b, total_b = torch.cuda.mem_get_info(0)
            used_mb = (total_b - free_b) / (1024 * 1024)
            result.update(
                {
                    "available": True,
                    "name": props.name,
                    "vram_used_mb": used_mb,
                    "vram_total_mb": total_mb,
                    "gpu_util_pct": None,
                    "summary": f"{props.name} | VRAM {used_mb / 1024:.2f} / {total_mb / 1024:.2f} GB",
                }
            )
    except Exception:
        pass

    return result


def get_disk_status(path: Path) -> dict[str, Any]:
    """Return real disk capacity for the partition holding workspace."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target)
    pct = (usage.used / usage.total * 100.0) if usage.total else 0.0
    return {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "used_pct": pct,
        "summary": f"{format_bytes(usage.used)} / {format_bytes(usage.total)} (ว่าง {format_bytes(usage.free)})",
    }


def format_startup_banner(workspace_root: Path) -> str:
    """Create startup diagnostic banner printed on app launch."""
    versions = get_package_versions()
    disk = get_disk_status(workspace_root)
    lines = [
        "=" * 64,
        "AI Training Studio — Runtime Diagnostics",
        "=" * 64,
        f"Python      : {versions.get('python')} ({sys.executable})",
        f"Platform    : {versions.get('platform')}",
        f"PyTorch     : {versions.get('pytorch')}",
        f"CUDA        : {versions.get('cuda_version')}",
        f"GPU         : {versions.get('gpu')}",
        f"Ultralytics : {versions.get('ultralytics')}",
        f"Gradio      : {versions.get('gradio')}",
        f"ONNX        : {versions.get('onnx')} | ONNXRuntime: {versions.get('onnxruntime')}",
        f"Workspace   : {workspace_root}",
        f"Disk        : {disk['summary']}",
        "=" * 64,
    ]
    return "\n".join(lines)
